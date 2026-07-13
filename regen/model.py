import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from regen.recovery import recover_damage


class PretrainedModelMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        device=None,
        token=None,
        filename="pytorch_model.pt",
        config_filename="config.json",
        repo_type="model",
        force_download=False,
        config_overrides=None,
    ):
        source = Path(str(pretrained_model_name_or_path))
        if source.exists():
            model = _load_pretrained_from_local(
                source,
                filename,
                config_filename,
                config_overrides=config_overrides,
            )
        else:
            config_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename=config_filename,
                repo_type=repo_type,
                token=token or os.environ.get("HF_TOKEN"),
                force_download=force_download,
            )
            weights_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename=filename,
                repo_type=repo_type,
                token=token or os.environ.get("HF_TOKEN"),
                force_download=force_download,
            )
            with open(config_path, "r") as f:
                config = json.load(f)
            state_dict = _load_state_dict(weights_path)
            model = _create_model_for_state_dict(config, state_dict, config_overrides)
            model.load_state_dict(state_dict)

        if device is not None:
            model = model.to(device)
        return model

    def save_pretrained(
        self,
        save_directory_or_repo_id,
        push_to_hub=False,
        token=None,
        filename="pytorch_model.pt",
        config_filename="config.json",
        repo_type="model",
        commit_message="Save pretrained model",
    ):
        if push_to_hub:
            with tempfile.TemporaryDirectory() as tmp_dir:
                self._save_pretrained_local(tmp_dir, filename, config_filename)
                api = HfApi()
                repo_id = str(save_directory_or_repo_id)
                try:
                    api.repo_info(repo_id, repo_type=repo_type)
                except RepositoryNotFoundError:
                    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
                return api.upload_folder(
                    folder_path=tmp_dir,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token or os.environ.get("HF_TOKEN"),
                    commit_message=commit_message,
                )
        return self._save_pretrained_local(
            save_directory_or_repo_id,
            filename,
            config_filename,
        )

    def _save_pretrained_local(self, save_directory, filename, config_filename):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        weights_path = save_directory / filename
        config_path = save_directory / config_filename
        torch.save(self.state_dict(), weights_path)
        config = self.get_config()
        config["api_version"] = 1
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)
        return {"weights": str(weights_path), "config": str(config_path)}

    @torch.no_grad()
    def predict(
        self,
        damaged_voxels,
        steps=96,
        class_label=None,
        shape_condition=None,
        return_state=False,
    ):
        self.eval()
        device = next(self.parameters()).device
        damaged_tensor = _voxel_tensor(damaged_voxels, device)
        condition = self.rollout_condition_tensor(
            damaged_tensor,
            class_label=class_label,
            shape_condition=shape_condition,
            device=device,
        )
        state = self.initialize(damaged_tensor)
        for _ in range(steps):
            state = self(state, condition) if condition is not None else self(state)

        damage_logits = self.classify(state)
        probabilities = torch.softmax(damage_logits, dim=-1)
        damage_confidence, damage_labels = torch.max(probabilities, dim=-1)
        class_logits = None
        predicted_class = None
        if getattr(self, "has_class_head", False):
            class_logits = self.classify_shape(state)
            predicted_class = torch.argmax(class_logits, dim=-1)
        return Prediction(
            damage_logits=damage_logits,
            damage_labels=damage_labels,
            damage_confidence=damage_confidence,
            damage_probabilities=probabilities,
            class_logits=class_logits,
            class_label=predicted_class,
            final_state=state if return_state else None,
        )

    def recover(
        self,
        damaged_voxels,
        original_mask=None,
        shape_condition=None,
        steps_per_prediction=96,
        recovery_steps=24,
        confidence_threshold=0.0,
        confidence_window=12,
        confidence_required=6,
        max_additions_per_step=None,
        constrain_to_original=True,
        show_progress=True,
        no_progress_patience=0,
        extra_steps_after_complete=0,
        consensus_min_votes=2,
        single_vote_confidence_threshold=0.99,
        direction_probability_threshold=0.6,
    ):
        recovery_shape_condition = shape_condition
        if recovery_shape_condition is None and original_mask is not None:
            recovery_shape_condition = original_mask

        def predict_fn(current):
            prediction = self.predict(
                current,
                steps=steps_per_prediction,
                shape_condition=recovery_shape_condition,
            )
            return (
                prediction.damage_labels.squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8),
                prediction.damage_confidence.squeeze(0).detach().cpu().numpy(),
                prediction.damage_probabilities.squeeze(0).detach().cpu().numpy(),
            )

        return recover_damage(
            damaged_voxels=np.asarray(damaged_voxels),
            predict_fn=predict_fn,
            original_mask=original_mask,
            recovery_steps=recovery_steps,
            confidence_threshold=confidence_threshold,
            confidence_window=confidence_window,
            confidence_required=confidence_required,
            max_additions_per_step=max_additions_per_step,
            constrain_to_original=constrain_to_original,
            show_progress=show_progress,
            no_progress_patience=no_progress_patience,
            extra_steps_after_complete=extra_steps_after_complete,
            consensus_min_votes=consensus_min_votes,
            single_vote_confidence_threshold=single_vote_confidence_threshold,
            direction_probability_threshold=direction_probability_threshold,
        )

    def rollout_condition_tensor(
        self,
        damaged_tensor,
        class_label=None,
        shape_condition=None,
        device=None,
    ):
        if getattr(self, "use_shape_conditioning", False):
            if shape_condition is None:
                shape_condition = damaged_tensor
            return _voxel_tensor(shape_condition, device or damaged_tensor.device)
        return _label_tensor(
            class_label,
            damaged_tensor.shape[0],
            device or damaged_tensor.device,
            use_labels=getattr(self, "use_class_embeddings", False),
        )

    def train_step(
        self,
        batch,
        optimizer,
        steps=96,
        loss_config=None,
        grad_clip=None,
    ):
        loss_config = loss_config or DamageLossConfig()
        self.train()
        device = next(self.parameters()).device
        structures, damage_directions, labels, original_shapes = _prepare_batch(
            batch,
            device,
        )

        state = self.initialize(structures)
        rollout_condition = (
            original_shapes
            if getattr(self, "use_shape_conditioning", False)
            else labels if getattr(self, "use_class_embeddings", False) else None
        )
        for _ in range(steps):
            state = (
                self(state, rollout_condition)
                if rollout_condition is not None
                else self(state)
            )

        loss, metrics = self.loss_and_metrics(
            state,
            damage_directions,
            labels,
            loss_config,
        )
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()
        return {
            key: value.detach().item() if torch.is_tensor(value) else float(value)
            for key, value in metrics.items()
        }

    def loss_and_metrics(self, final_state, damage_directions, labels, loss_config):
        damage_logits = self.classify(final_state)
        damage_loss = self._damage_loss(
            damage_logits,
            damage_directions,
            final_state,
            loss_config,
        )
        loss = loss_config.damage_loss_weight * damage_loss

        class_loss = None
        class_accuracy = None
        if getattr(self, "has_class_head", False) and labels is not None:
            class_logits = self.classify_shape(final_state)
            class_loss = F.cross_entropy(class_logits, labels)
            class_accuracy = (
                (torch.argmax(class_logits, dim=-1) == labels).float().mean()
            )
            loss = loss + loss_config.class_loss_weight * class_loss

        if not getattr(self, "use_tanh", True):
            loss = loss + loss_config.clipping_loss_weight * self._clipping_loss(
                final_state
            )

        with torch.no_grad():
            pred_classes = torch.argmax(damage_logits, dim=-1)
            correct = (pred_classes == damage_directions).float()
            alive_mask = self._alive_mask(final_state)
            full_accuracy = (correct * alive_mask).sum() / (alive_mask.sum() + 1e-8)
            damaged_mask = ((damage_directions > 0).float() * alive_mask).float()
            damaged_accuracy = (correct * damaged_mask).sum() / (
                damaged_mask.sum() + 1e-8
            )

        metrics = {
            "loss": loss.detach(),
            "damage_loss": damage_loss.detach(),
            "full_accuracy": full_accuracy.detach(),
            "damaged_accuracy": damaged_accuracy.detach(),
        }
        if class_loss is not None:
            metrics["class_loss"] = class_loss.detach()
        if class_accuracy is not None:
            metrics["class_accuracy"] = class_accuracy.detach()
        return loss, metrics

    def _damage_loss(self, damage_logits, damage_directions, final_state, loss_config):
        batch, _, _, _, num_classes = damage_logits.shape
        pred_flat = (
            damage_logits.permute(0, 4, 1, 2, 3)
            .contiguous()
            .view(batch, num_classes, -1)
        )
        targets_flat = damage_directions.view(batch, -1)
        weights = torch.ones(num_classes, device=damage_logits.device)
        weights[1:] = loss_config.damage_class_weight
        cell_loss = nn.CrossEntropyLoss(weight=weights, reduction="none")(
            pred_flat,
            targets_flat,
        )

        if loss_config.damage_loss_type == "focal":
            log_probs = F.log_softmax(pred_flat, dim=1)
            target_log_probs = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
            pt = target_log_probs.exp()
            cell_loss = ((1.0 - pt) ** loss_config.focal_gamma) * cell_loss
        elif loss_config.damage_loss_type != "cross_entropy":
            raise ValueError(
                f"Unsupported damage_loss_type: {loss_config.damage_loss_type}"
            )

        alive_mask_flat = self._alive_mask(final_state).view(batch, -1)
        return (cell_loss * alive_mask_flat).sum() / (alive_mask_flat.sum() + 1e-8)

    def _clipping_loss(self, final_state):
        batch = final_state.shape[0]
        clipped_states = torch.clamp(
            final_state.detach(),
            -self.clip_range,
            self.clip_range,
        )
        clip_loss = (
            F.mse_loss(final_state, clipped_states, reduction="none")
            .sum(dim=-1)
            .view(batch, -1)
        )
        alive_mask_flat = self._alive_mask(final_state).view(batch, -1)
        return (clip_loss * alive_mask_flat).sum() / (alive_mask_flat.sum() + 1e-8)

    def _alive_mask(self, final_state):
        return (final_state[:, :, :, :, 0] > self.alpha_living_threshold).float()


class CellRecoveryModel(PretrainedModelMixin, nn.Module):
    """3D NCA model with native pretrained save/load, prediction, and recovery APIs."""

    def __init__(
        self,
        num_hidden_channels=128,
        num_damage_directions=7,
        num_classes=0,
        use_class_embeddings=False,
        use_shape_conditioning=False,
        shape_condition_channels=8,
        class_channels=0,
        alpha_living_threshold=0.1,
        cell_fire_rate=0.5,
        clip_range=64.0,
        use_tanh=True,
        freeze_perception=True,
        model_type="CellRecoveryModel",
        use_adaptive_pooling=False,
    ):
        super().__init__()
        if use_class_embeddings and use_shape_conditioning:
            raise ValueError(
                "use_class_embeddings and use_shape_conditioning are mutually exclusive."
            )
        self.model_type = model_type
        self.use_class_embeddings = use_class_embeddings
        self.use_shape_conditioning = use_shape_conditioning
        self.shape_condition_channels = shape_condition_channels
        self.num_classes = num_classes
        self.num_damage_directions = num_damage_directions
        self.num_hidden_channels = num_hidden_channels
        self.alpha_living_threshold = alpha_living_threshold
        self.cell_fire_rate = cell_fire_rate
        self.clip_range = clip_range
        self.use_tanh = use_tanh
        self.freeze_perception = freeze_perception
        self.class_channels = class_channels
        self.has_class_head = class_channels > 0
        self.use_adaptive_pooling = use_adaptive_pooling
        self.class_channel_start = 1 + num_hidden_channels
        self.damage_channel_start = self.class_channel_start + class_channels
        self.channel_n = (
            1 + num_hidden_channels + class_channels + num_damage_directions
        )
        self.perception_channels = self.channel_n * 3

        if self.use_class_embeddings:
            self.class_embeddings = nn.Embedding(num_classes, self.channel_n - 1)
        if self.use_shape_conditioning:
            if self.use_adaptive_pooling:
                self.shape_encoder = nn.Sequential(
                    nn.Conv3d(
                        1, shape_condition_channels, kernel_size=3, padding="same"
                    ),
                    nn.Tanh(),
                    nn.AdaptiveAvgPool3d((5, 5, 5)),
                    nn.Flatten(),
                    nn.Linear(shape_condition_channels * 5 * 5 * 5, self.channel_n - 1),
                )
            else:
                self.shape_encoder = nn.Sequential(
                    nn.Conv3d(
                        1, shape_condition_channels, kernel_size=3, padding="same"
                    ),
                    nn.Tanh(),
                    nn.Conv3d(
                        shape_condition_channels,
                        shape_condition_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.Tanh(),
                    nn.Conv3d(
                        shape_condition_channels, self.channel_n - 1, kernel_size=1
                    ),
                )

        self.kernel_mask = torch.tensor(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3, 3)

        self.perceive = nn.Sequential(
            nn.Conv3d(
                self.channel_n,
                self.perception_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
        )
        self.dmodel = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(
                self.perception_channels, self.perception_channels, kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv3d(
                self.perception_channels, self.channel_n - 1, kernel_size=1, bias=False
            ),
        )
        self._init_weights()
        self.reset_diag_kernel()

    def _init_weights(self):
        nn.init.zeros_(self.dmodel[-1].weight)

    def reset_diag_kernel(self):
        conv_layer = self.perceive[0]
        conv_layer.weight.requires_grad = not self.freeze_perception
        kernel_mask = self.kernel_mask.to(conv_layer.weight.device)
        kernel_mask = kernel_mask.repeat(
            conv_layer.out_channels, conv_layer.in_channels, 1, 1, 1
        )
        conv_layer.weight.data *= kernel_mask

    def forward(self, x, y=None):
        gray, state = torch.split(x, [1, self.channel_n - 1], dim=-1)
        update = self.dmodel(self.perceive(x.permute(0, 4, 1, 2, 3))).permute(
            0, 2, 3, 4, 1
        )
        if y is not None:
            update = update + self.condition_embedding(y)

        update_mask = torch.rand_like(x[:, :, :, :, :1]) <= self.cell_fire_rate
        living_mask = gray > self.alpha_living_threshold
        residual_mask = (update_mask & living_mask).float()
        if self.use_tanh:
            state = state + residual_mask * torch.tanh(update)
        else:
            state = state + residual_mask * update
        return torch.cat([gray, state], dim=-1)

    def condition_embedding(self, condition):
        if self.use_shape_conditioning:
            return self.shape_condition_embedding(condition)
        if self.use_class_embeddings:
            return self.class_embeddings(condition).view(
                condition.shape[0],
                1,
                1,
                1,
                self.channel_n - 1,
            )
        raise ValueError("This model does not use rollout conditioning.")

    def shape_condition_embedding(self, shape_condition):
        shape_condition = shape_condition.float()
        if shape_condition.ndim == 3:
            shape_condition = shape_condition.unsqueeze(0)
        if shape_condition.ndim == 4:
            shape_condition = shape_condition.unsqueeze(1)
        elif shape_condition.ndim == 5 and shape_condition.shape[-1] == 1:
            shape_condition = shape_condition.permute(0, 4, 1, 2, 3)
        if shape_condition.ndim != 5 or shape_condition.shape[1] != 1:
            raise ValueError(
                "Expected shape condition as [D,H,W], [B,D,H,W], or [B,1,D,H,W]."
            )
        encoded = self.shape_encoder(shape_condition)
        if self.use_adaptive_pooling:
            return encoded.view(
                shape_condition.shape[0],
                1,
                1,
                1,
                self.channel_n - 1,
            )
        return encoded.permute(0, 2, 3, 4, 1)

    def damage_logits(self, x):
        return x[:, :, :, :, self.damage_channel_start :]

    def class_logits_per_cell(self, x):
        if not self.has_class_head:
            raise AttributeError("This model does not have a class head.")
        return x[
            :,
            :,
            :,
            :,
            self.class_channel_start : self.damage_channel_start,
        ]

    def class_logits(self, x):
        per_cell_logits = self.class_logits_per_cell(x)
        alive_mask = (x[:, :, :, :, :1] > self.alpha_living_threshold).float()
        pooled_logits = (per_cell_logits * alive_mask).sum(dim=(1, 2, 3))
        alive_counts = alive_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        return pooled_logits / alive_counts

    def classify(self, x):
        return self.damage_logits(x)

    def classify_shape(self, x):
        return self.class_logits(x)

    def initialize(self, structure):
        shape = structure.shape
        state = torch.zeros(
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            self.channel_n - 1,
            device=structure.device,
        )
        structure = structure.view(shape[0], shape[1], shape[2], shape[3], 1)
        return torch.cat([structure, state], dim=-1)

    def get_config(self):
        config = {
            "model_type": self.model_type,
            "num_hidden_channels": self.num_hidden_channels,
            "num_classes": self.num_classes,
            "num_damage_directions": self.num_damage_directions,
            "alpha_living_threshold": self.alpha_living_threshold,
            "cell_fire_rate": self.cell_fire_rate,
            "clip_range": self.clip_range,
            "use_tanh": self.use_tanh,
            "freeze_perception": self.freeze_perception,
            "use_class_embeddings": self.use_class_embeddings,
            "use_shape_conditioning": self.use_shape_conditioning,
            "shape_condition_channels": self.shape_condition_channels,
            "use_adaptive_pooling": self.use_adaptive_pooling,
            "class_channels": self.class_channels,
            "channel_n": self.channel_n,
            "perception_channels": self.perception_channels,
        }
        return config


@dataclass
class Prediction:
    damage_logits: torch.Tensor
    damage_labels: torch.Tensor
    damage_confidence: torch.Tensor
    damage_probabilities: torch.Tensor
    class_logits: Optional[torch.Tensor] = None
    class_label: Optional[torch.Tensor] = None
    final_state: Optional[torch.Tensor] = None


@dataclass
class DamageLossConfig:
    damage_class_weight: float = 1.0
    damage_loss_type: str = "cross_entropy"
    focal_gamma: float = 2.0
    damage_loss_weight: float = 1.0
    class_loss_weight: float = 1.0
    clipping_loss_weight: float = 1.0


def create_model_from_config(config: dict):
    model_type = config.get("model_type", "CellRecoveryModel")
    if model_type == "NCA3DDamageDetection":
        config = {**config, "model_type": "CellRecoveryModel", "class_channels": 0}
    elif model_type == "NCA3DCombinedDamageClassifier":
        config = {
            **config,
            "model_type": "CellRecoveryModel",
            "class_channels": config.get("num_classes", 0),
            "use_class_embeddings": False,
        }
    elif model_type != "CellRecoveryModel":
        raise ValueError(f"Unsupported model_type in config: {model_type}")

    return CellRecoveryModel(
        num_hidden_channels=config.get("num_hidden_channels", 128),
        num_damage_directions=config.get("num_damage_directions", 7),
        num_classes=config.get("num_classes", 0),
        use_class_embeddings=config.get("use_class_embeddings", False),
        use_shape_conditioning=config.get("use_shape_conditioning", False),
        shape_condition_channels=config.get("shape_condition_channels", 8),
        class_channels=config.get("class_channels", 0),
        alpha_living_threshold=config.get("alpha_living_threshold", 0.1),
        cell_fire_rate=config.get("cell_fire_rate", 0.5),
        clip_range=config.get("clip_range", 64.0),
        use_tanh=config.get("use_tanh", True),
        freeze_perception=config.get("freeze_perception", True),
        use_adaptive_pooling=config.get("use_adaptive_pooling", False),
    )


def _load_pretrained_from_local(path, filename, config_filename, config_overrides=None):
    if path.is_dir():
        config_path = path / config_filename
        weights_path = path / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Missing pretrained config: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing pretrained weights: {weights_path}")
        with config_path.open("r") as f:
            config = json.load(f)
        state_dict = _load_state_dict(weights_path)
        model = _create_model_for_state_dict(config, state_dict, config_overrides)
        model.load_state_dict(state_dict)
        return model

    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("model_config") or checkpoint.get("config")
    if config is None:
        raise ValueError("Local checkpoint must contain `model_config` or `config`.")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = _create_model_for_state_dict(config, state_dict, config_overrides)
    model.load_state_dict(state_dict)
    return model


def _prepare_batch(batch, device):
    structures, damage_directions, labels = batch[:3]
    original_shapes = batch[3] if len(batch) > 3 else structures
    return (
        structures.float().to(device),
        damage_directions.long().to(device),
        labels.long().to(device),
        original_shapes.float().to(device),
    )


def _voxel_tensor(voxels, device):
    if torch.is_tensor(voxels):
        tensor = voxels.float().to(device)
    else:
        tensor = torch.tensor(voxels, dtype=torch.float32, device=device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected 3D or batched 4D voxels, got {tensor.shape}")
    return tensor


def _label_tensor(class_label, batch_size, device, use_labels):
    if class_label is None or not use_labels:
        return None
    if torch.is_tensor(class_label):
        labels = class_label.long().to(device)
    else:
        labels = torch.tensor(class_label, dtype=torch.long, device=device)
    if labels.ndim == 0:
        labels = labels.repeat(batch_size)
    return labels


def _load_pretrained_weights(model, weights_path):
    state_dict = _load_state_dict(weights_path)
    model.load_state_dict(state_dict)
    return model


def _load_state_dict(weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu")
    return checkpoint.get("model_state_dict", checkpoint)


def _create_model_for_state_dict(config, state_dict, config_overrides=None):
    config = dict(config)
    if config_overrides:
        config.update(config_overrides)

    inferred_adaptive_pooling = infer_adaptive_pooling_from_state_dict(state_dict)
    if inferred_adaptive_pooling is not None:
        config["use_adaptive_pooling"] = inferred_adaptive_pooling
    return create_model_from_config(config)


def infer_adaptive_pooling_from_state_dict(state_dict):
    shape_encoder_weight = state_dict.get("shape_encoder.4.weight")
    if shape_encoder_weight is None:
        return None
    if shape_encoder_weight.ndim == 2:
        return True
    if shape_encoder_weight.ndim == 5:
        return False
    return None
