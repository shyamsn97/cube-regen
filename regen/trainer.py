import os
import time
from importlib import import_module
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from regen.base_trainer import BaseNCA3DTrainer, ReplayBuffer
from regen.dataset import centered_subset_mask
from regen.utils import plot_voxels

__all__ = ["ReplayBuffer", "NCA3DTrainer"]


def wandb_safe_config(value):
    if isinstance(value, dict):
        return {str(key): wandb_safe_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [wandb_safe_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class NCA3DTrainer(BaseNCA3DTrainer):
    """Single trainer for damage-only and combined damage/classification NCAs."""

    def __init__(
        self,
        model,
        dataset=None,
        train_dataset=None,
        val_dataset=None,
        batch_size=8,
        lr=1e-4,
        iterations_per_epoch=100,
        min_steps_per_sample=96,
        max_steps_per_sample=128,
        buffer_size=1000,
        buffer_sampling_prob=0.5,
        grad_clip=1.0,
        gradient_checkpointing=False,
        damage_loss_weight=1.0,
        class_loss_weight=0.0,
        damage_class_weight=1.0,
        damage_loss_type="cross_entropy",
        focal_gamma=2.0,
        device=None,
        wandb_project="nca-3d-damage-detection",
        wandb_run_name=None,
        wandb_watch=True,
        wandb_watch_log="all",
        wandb_watch_log_freq=100,
        wandb_log_gradient_sums=True,
        wandb_gradient_log_freq=1,
        save_dir="./nca_models",
        num_workers=0,
        validate_frequency=1,
        repo_id=None,
        repo_type="model",
        trainer_name=None,
        experiment_config=None,
        recovery_eval_frequency=0,
        recovery_eval_samples=0,
        recovery_eval_iterations=24,
        recovery_eval_prediction_steps=None,
        recovery_eval_start_mode="sample",
        recovery_eval_seed_proportion=0.15,
    ):
        train_dataset = train_dataset if train_dataset is not None else dataset
        if train_dataset is None:
            raise ValueError("NCA3DTrainer requires `dataset` or `train_dataset`.")

        self.dataset = train_dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.has_class_head = bool(getattr(model, "has_class_head", False))
        self.trainer_name = trainer_name or "NCA3DTrainer"
        self.save_dir = save_dir
        self.damage_loss_weight = damage_loss_weight
        self.class_loss_weight = class_loss_weight
        self.damage_class_weight = damage_class_weight
        self.damage_loss_type = damage_loss_type
        self.focal_gamma = focal_gamma
        self.validate_frequency = validate_frequency
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.recovery_eval_frequency = recovery_eval_frequency
        self.recovery_eval_samples = recovery_eval_samples
        self.recovery_eval_iterations = recovery_eval_iterations
        self.recovery_eval_prediction_steps = recovery_eval_prediction_steps
        self.recovery_eval_start_mode = recovery_eval_start_mode
        self.recovery_eval_seed_proportion = recovery_eval_seed_proportion
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        super().__init__(
            model=model,
            batch_size=batch_size,
            lr=lr,
            iterations_per_epoch=iterations_per_epoch,
            min_steps_per_sample=min_steps_per_sample,
            max_steps_per_sample=max_steps_per_sample,
            buffer_size=buffer_size,
            buffer_sampling_prob=buffer_sampling_prob,
            device=device,
            grad_clip=grad_clip,
            gradient_checkpointing=gradient_checkpointing,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_config={
                "trainer": self.trainer_name,
                "optimizer": "Adam",
                "learning_rate": lr,
                "batch_size": batch_size,
                "iterations_per_epoch": iterations_per_epoch,
                "min_steps_per_sample": min_steps_per_sample,
                "max_steps_per_sample": max_steps_per_sample,
                "buffer_size": buffer_size,
                "buffer_sampling_prob": buffer_sampling_prob,
                "grad_clip": grad_clip,
                "gradient_checkpointing": gradient_checkpointing,
                "damage_loss_weight": damage_loss_weight,
                "class_loss_weight": class_loss_weight,
                "damage_class_weight": damage_class_weight,
                "damage_loss_type": damage_loss_type,
                "focal_gamma": focal_gamma,
                "model": model.get_config(),
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset) if val_dataset is not None else 0,
                "num_workers": num_workers,
                "validate_frequency": validate_frequency,
                "save_dir": self.save_dir,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "experiment_config": wandb_safe_config(experiment_config or {}),
                "recovery_eval_frequency": recovery_eval_frequency,
                "recovery_eval_samples": recovery_eval_samples,
                "recovery_eval_iterations": recovery_eval_iterations,
                "recovery_eval_prediction_steps": recovery_eval_prediction_steps,
                "recovery_eval_start_mode": recovery_eval_start_mode,
                "recovery_eval_seed_proportion": recovery_eval_seed_proportion,
            },
            wandb_watch=wandb_watch,
            wandb_watch_log=wandb_watch_log,
            wandb_watch_log_freq=wandb_watch_log_freq,
            wandb_log_gradient_sums=wandb_log_gradient_sums,
            wandb_gradient_log_freq=wandb_gradient_log_freq,
        )

        damage_weights = torch.ones(self.model.num_damage_directions)
        damage_weights[1:] = damage_class_weight
        self.damage_loss_fn = nn.CrossEntropyLoss(
            weight=damage_weights.to(self.device),
            reduction="none",
        )
        # Backwards compatible name used by older damage-only call sites.
        self.loss_fn = self.damage_loss_fn
        self.class_loss_fn = nn.CrossEntropyLoss()

        self.set_train_dataset(train_dataset, num_workers=num_workers)
        self.val_loader = self.make_eval_loader(val_dataset, num_workers=num_workers)
        os.makedirs(self.save_dir, exist_ok=True)

    def replay_samples_per_fresh_batch(self):
        return 2 if not self.has_class_head else 1

    def loss_function(self, first, second, third):
        """Compatibility wrapper for the old damage and combined trainer APIs."""
        if first.shape[-1] == self.model.channel_n:
            return self.loss_and_metrics(first, second, third)

        damage_loss = self.masked_damage_loss(
            self.damage_loss_fn,
            first,
            second,
            third,
            loss_type=self.damage_loss_type,
            focal_gamma=self.focal_gamma,
        )
        if not self.model.use_tanh:
            damage_loss = damage_loss + self.clipping_loss(third)
        return damage_loss

    def loss_and_metrics(self, final_state, damage_directions, labels):
        damage_logits = self.model.classify(final_state)
        damage_loss = self.masked_damage_loss(
            self.damage_loss_fn,
            damage_logits,
            damage_directions,
            final_state,
            loss_type=self.damage_loss_type,
            focal_gamma=self.focal_gamma,
        )
        loss = self.damage_loss_weight * damage_loss

        class_loss = None
        class_accuracy = None
        if self.has_class_head:
            class_logits = self.model.classify_shape(final_state)
            class_loss = self.class_loss_fn(class_logits, labels)
            class_predictions = torch.argmax(class_logits, dim=-1)
            class_accuracy = (class_predictions == labels).float().mean()
            loss = loss + self.class_loss_weight * class_loss

        if not self.model.use_tanh:
            loss = loss + self.clipping_loss(final_state)

        damage_accuracy = self.damage_accuracy(
            damage_logits,
            damage_directions,
            final_state,
        )
        damaged_accuracy = self.damaged_accuracy(
            damage_logits,
            damage_directions,
            final_state,
        )
        metrics = {
            "loss": loss.detach(),
            "damage_loss": damage_loss.detach(),
            "full_accuracy": damage_accuracy.detach(),
            "damage_accuracy": damage_accuracy.detach(),
            "damaged_accuracy": damaged_accuracy.detach(),
            "damage_boundary_accuracy": damaged_accuracy.detach(),
        }
        if class_loss is not None:
            metrics["class_loss"] = class_loss.detach()
        if class_accuracy is not None:
            metrics["class_accuracy"] = class_accuracy.detach()
        metrics.update(
            {
                key: value.detach()
                for key, value in self.damage_diagnostics(
                    damage_logits,
                    damage_directions,
                    final_state,
                ).items()
            }
        )
        return loss, metrics

    def progress_postfix(self, averages):
        postfix = {
            "loss": f"{averages.get('loss', 0.0):.4f}",
            "acc": f"{averages.get('damage_accuracy', 0.0):.4f}",
            "dmg_acc": f"{averages.get('damaged_accuracy', 0.0):.4f}",
        }
        if self.has_class_head:
            postfix["cls_acc"] = f"{averages.get('class_accuracy', 0.0):.4f}"
        return postfix

    @torch.no_grad()
    def validate(self) -> Optional[Dict[str, float]]:
        if self.val_loader is None:
            return None

        self.model.eval()
        totals = {}
        batch_count = 0
        for batch in self.val_loader:
            structures, damage_directions, labels, original_shapes = (
                self.prepare_damage_batch(batch)
            )
            states = self.model.initialize(structures)
            final_state, _ = self.run_nca(
                states,
                self.rollout_condition(labels, original_shapes),
            )
            _, metrics = self.loss_and_metrics(final_state, damage_directions, labels)
            batch_count += 1
            for key, value in self.metric_items(metrics).items():
                totals[key] = totals.get(key, 0.0) + value

        return {key: value / batch_count for key, value in totals.items()}

    @torch.no_grad()
    def evaluate_damage_dataset(self):
        self.model.eval()
        correct = 0
        total = 0
        damaged_correct = 0
        damaged_total = 0
        for batch in DataLoader(self.dataset, batch_size=self.batch_size):
            damage_mask, damage_direction, label, original_shape = (
                self.prepare_damage_batch(batch)
            )
            state = self.model.initialize(damage_mask).to(self.device)
            state, _ = self.run_nca(
                state,
                self.rollout_condition(label, original_shape),
            )
            predictions = self.model.classify(state)
            predicted_labels = torch.argmax(predictions, dim=-1)

            total += damage_direction.numel()
            correct += (predicted_labels == damage_direction).sum().item()
            damaged_mask = damage_direction > 0
            damaged_total += damaged_mask.sum().item()
            damaged_correct += (
                ((predicted_labels == damage_direction) & damaged_mask).sum().item()
            )

        self.model.train()
        return {
            "accuracy": 100 * correct / total if total > 0 else 0.0,
            "damaged_accuracy": (
                100 * damaged_correct / damaged_total if damaged_total > 0 else 0.0
            ),
        }

    def save_model(
        self,
        epoch,
        loss,
        metrics: Optional[Dict[str, float]] = None,
        is_best=False,
    ):
        metrics = metrics or {"loss": loss}
        print(
            f"Saving checkpoint for epoch {epoch} "
            f"(loss={loss:.4f}) to {self.save_dir}"
        )
        checkpoint = self.checkpoint_state(
            epoch,
            {
                "metrics": metrics,
                "model_config": self.model.get_config(),
            },
        )
        latest_path = os.path.join(self.save_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        print(f"Saved local checkpoint: {latest_path}")

        epoch_path = os.path.join(self.save_dir, f"epoch_{epoch}.pt")
        torch.save(checkpoint, epoch_path)
        print(f"Saved local checkpoint: {epoch_path}")

        pretrained_paths = self.model.save_pretrained(self.save_dir)
        print(f"Saved pretrained model: {pretrained_paths}")

        if is_best:
            best_path = os.path.join(self.save_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")

        if self.repo_id:
            print(f"Uploading pretrained model to Hugging Face: {self.repo_id}")
            self.model.save_pretrained(
                self.repo_id,
                push_to_hub=True,
                repo_type=self.repo_type,
                commit_message=f"Save model for epoch {epoch}",
            )
        else:
            print("No Hugging Face repo configured for upload.")

    def train(self, epochs, save_frequency=5, visualization_frequency=10):
        self.init_wandb()
        writer = None
        if not self.has_class_head:
            try:
                tensorboard = import_module("torch.utils.tensorboard")
                writer = tensorboard.SummaryWriter(
                    log_dir=f"{self.save_dir}/tensorboard"
                )
            except ImportError:
                print("TensorBoard is not installed; skipping tensorboard logging.")

        for epoch in range(epochs):
            start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            train_loss = train_metrics["loss"]
            self.train_losses.append(train_loss)

            val_metrics = self._maybe_validate(epoch, epochs)
            if val_metrics is not None:
                self.val_losses.append(val_metrics["loss"])

            eval_metrics = None
            if not self.has_class_head:
                eval_metrics = self.evaluate_damage_dataset()
            recovery_metrics = self._maybe_evaluate_recovery(epoch, epochs)
            if recovery_metrics is not None:
                eval_metrics = eval_metrics or {}
                eval_metrics.update(recovery_metrics)

            elapsed = time.time() - start_time
            self._print_epoch_summary(
                epoch, train_metrics, val_metrics, eval_metrics, elapsed
            )
            self._log_epoch_metrics(
                epoch, train_metrics, val_metrics, eval_metrics, elapsed
            )
            self._write_tensorboard(writer, epoch, train_metrics, eval_metrics, elapsed)

            should_save = save_frequency > 0 and (
                epoch % save_frequency == 0 or epoch == epochs - 1
            )
            if should_save:
                monitor_loss = val_metrics["loss"] if val_metrics else train_loss
                is_best = monitor_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = monitor_loss
                self.save_model(
                    epoch,
                    monitor_loss if self.has_class_head else train_loss,
                    metrics=self._prefixed_metrics(train_metrics, val_metrics),
                    is_best=is_best,
                )

            if (
                writer is not None
                and visualization_frequency > 0
                and (epoch % visualization_frequency == 0 or epoch == epochs - 1)
            ):
                img = self.visualize_results(epoch)
                if self.wandb_initialized:
                    self.log_wandb(
                        {"visualization": self.wandb.Image(img), "epoch": epoch},
                        step=self.global_step,
                    )
                writer.add_image(
                    "Visualization", np.array(img), epoch, dataformats="HWC"
                )

    def _maybe_validate(self, epoch, epochs):
        should_validate = (
            self.val_loader is not None
            and self.validate_frequency > 0
            and (epoch % self.validate_frequency == 0 or epoch == epochs - 1)
        )
        if not should_validate:
            return None
        return self.validate()

    def _maybe_evaluate_recovery(self, epoch, epochs):
        should_evaluate = (
            self.recovery_eval_frequency
            and self.recovery_eval_samples
            and (epoch % self.recovery_eval_frequency == 0 or epoch == epochs - 1)
        )
        if not should_evaluate:
            return None
        return self.evaluate_recovery()

    @torch.no_grad()
    def evaluate_recovery(self):
        self.model.eval()
        sample_count = min(self.recovery_eval_samples, len(self.dataset))
        if sample_count <= 0:
            return None

        prediction_steps = (
            self.recovery_eval_prediction_steps or self.max_steps_per_sample
        )
        totals = {
            "recovery_missing": 0.0,
            "recovery_extra": 0.0,
            "recovery_added": 0.0,
            "recovery_recovered_fraction": 0.0,
        }
        for idx in range(sample_count):
            sample = self.dataset[idx]
            damaged = sample[0]
            original_np = self.dataset.shapes[idx].astype(np.uint8)
            damaged_np = damaged.numpy().astype(np.uint8)
            start_np = self.recovery_eval_start(original_np, damaged_np)
            initial_missing = int(((original_np == 1) & (start_np == 0)).sum())
            trajectory = self.model.recover(
                start_np,
                original_mask=original_np,
                steps_per_prediction=prediction_steps,
                recovery_steps=self.recovery_eval_iterations,
                show_progress=False,
            )
            final_step = trajectory.steps[-1]
            missing = final_step.missing_count or 0
            extra = final_step.extra_count or 0
            recovered = 0.0
            if initial_missing > 0:
                recovered = (initial_missing - missing) / initial_missing
            totals["recovery_missing"] += float(missing)
            totals["recovery_extra"] += float(extra)
            totals["recovery_added"] += float(final_step.total_added_count)
            totals["recovery_recovered_fraction"] += float(recovered)

        self.model.train()
        return {key: value / sample_count for key, value in totals.items()}

    def recovery_eval_start(self, original_np, damaged_np):
        if self.recovery_eval_start_mode == "sample":
            return damaged_np
        if self.recovery_eval_start_mode == "seed":
            return centered_subset_mask(original_np, self.recovery_eval_seed_proportion)
        raise ValueError(
            f"Unsupported recovery_eval_start_mode: {self.recovery_eval_start_mode}"
        )

    def _prefixed_metrics(self, train_metrics, val_metrics):
        metrics = {f"train_{key}": value for key, value in train_metrics.items()}
        if val_metrics:
            metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
        return metrics

    def _print_epoch_summary(
        self,
        epoch,
        train_metrics,
        val_metrics,
        eval_metrics,
        elapsed,
    ):
        parts = [
            f"Epoch {epoch}",
            f"train_loss={train_metrics['loss']:.4f}",
            f"train_full_acc={train_metrics.get('damage_accuracy', 0.0):.3f}",
            f"train_damage_acc={train_metrics.get('damaged_accuracy', 0.0):.3f}",
        ]
        if self.has_class_head:
            parts.append(
                f"train_class_acc={train_metrics.get('class_accuracy', 0.0):.3f}"
            )
        if val_metrics is not None:
            parts.extend(
                [
                    f"val_loss={val_metrics['loss']:.4f}",
                    f"val_full_acc={val_metrics.get('damage_accuracy', 0.0):.3f}",
                    f"val_damage_acc={val_metrics.get('damaged_accuracy', 0.0):.3f}",
                ]
            )
            if self.has_class_head:
                parts.append(
                    f"val_class_acc={val_metrics.get('class_accuracy', 0.0):.3f}"
                )
        if eval_metrics is not None:
            if "accuracy" in eval_metrics:
                parts.extend(
                    [
                        f"eval_acc={eval_metrics['accuracy']:.2f}%",
                        f"eval_damage_acc={eval_metrics['damaged_accuracy']:.2f}%",
                    ]
                )
            if "recovery_recovered_fraction" in eval_metrics:
                parts.extend(
                    [
                        "recovery_recovered="
                        f"{eval_metrics['recovery_recovered_fraction']:.3f}",
                        f"recovery_missing={eval_metrics['recovery_missing']:.1f}",
                        f"recovery_extra={eval_metrics['recovery_extra']:.1f}",
                    ]
                )
        parts.append(f"time={elapsed:.1f}s")
        print(" | ".join(parts))

    def _log_epoch_metrics(
        self,
        epoch,
        train_metrics,
        val_metrics,
        eval_metrics,
        elapsed,
    ):
        metrics = {
            "epoch": epoch,
            "epoch_time": elapsed,
            **{f"train_{key}": value for key, value in train_metrics.items()},
        }
        if val_metrics:
            metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
        if eval_metrics:
            if "accuracy" in eval_metrics:
                metrics.update(
                    {
                        "accuracy": eval_metrics["accuracy"],
                        "damaged_accuracy": eval_metrics["damaged_accuracy"],
                    }
                )
            for key, value in eval_metrics.items():
                if key.startswith("recovery_"):
                    metrics[key] = value
        self.log_wandb(metrics, step=self.global_step)

    def _write_tensorboard(self, writer, epoch, train_metrics, eval_metrics, elapsed):
        if writer is None:
            return
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar(
            "Accuracy/train_alive",
            train_metrics.get("damage_accuracy", 0.0),
            epoch,
        )
        writer.add_scalar(
            "Accuracy/train_damaged",
            train_metrics.get("damaged_accuracy", 0.0),
            epoch,
        )
        if eval_metrics:
            if "accuracy" in eval_metrics:
                writer.add_scalar("Accuracy/train", eval_metrics["accuracy"], epoch)
                writer.add_scalar(
                    "Accuracy/damaged", eval_metrics["damaged_accuracy"], epoch
                )
            for key, value in eval_metrics.items():
                if key.startswith("recovery_"):
                    writer.add_scalar(
                        f"Recovery/{key.removeprefix('recovery_')}",
                        value,
                        epoch,
                    )
        writer.add_scalar("Time/epoch", elapsed, epoch)

    def visualize_results(self, epoch):
        del epoch
        self.model.eval()

        with torch.no_grad():
            damage_mask_tensor, damage_direction_tensor, label, original_shape = (
                self.dataset[0]
            )
            damage_mask_tensor = damage_mask_tensor.unsqueeze(0).to(self.device)
            damage_direction_tensor = damage_direction_tensor.unsqueeze(0).to(
                self.device
            )
            label = label.unsqueeze(0).to(self.device)
            original_shape = original_shape.unsqueeze(0).to(self.device)

            state = self.model.initialize(damage_mask_tensor).to(self.device)
            state, _ = self.run_nca(
                state,
                self.rollout_condition(label, original_shape),
            )
            predictions = self.model.classify(state)
            predictions = torch.argmax(predictions, dim=-1).detach().cpu().numpy()[0]

            damage_mask = (
                damage_mask_tensor.squeeze().detach().cpu().numpy().astype(np.uint8)
            )
            damage_direction = (
                damage_direction_tensor.squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            predictions_np = predictions.astype(np.uint8)

            true_damage_img = plot_voxels(
                live_mask=damage_mask,
                damage_direction=damage_direction,
            )
            predicted_damage_img = plot_voxels(
                live_mask=damage_mask,
                damage_direction=predictions_np,
            )

            combined_width = true_damage_img.width + predicted_damage_img.width
            combined_height = max(true_damage_img.height, predicted_damage_img.height)
            combined_img = Image.new("RGBA", (combined_width, combined_height))
            combined_img.paste(true_damage_img, (0, 0))
            combined_img.paste(predicted_damage_img, (true_damage_img.width, 0))
            return combined_img
