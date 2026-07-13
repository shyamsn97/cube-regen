import random
from abc import ABC, abstractmethod
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from regen.device import preferred_device


class ReplayBuffer:
    """Replay buffer for intermediate NCA states and their targets."""

    def __init__(self, buffer_size=1000, sampling_prob=0.5):
        self.buffer = deque(maxlen=buffer_size)
        self.sampling_prob = sampling_prob

    def add(self, states, damage_directions, labels, conditions=None):
        for i in range(states.shape[0]):
            condition = None
            if conditions is not None:
                condition = conditions[i].detach().cpu().clone()
            self.buffer.append(
                (
                    states[i].detach().cpu().clone(),
                    damage_directions[i].detach().cpu().clone(),
                    labels[i].detach().cpu().clone(),
                    condition,
                )
            )

    def sample(self, batch_size, device):
        if len(self.buffer) < batch_size:
            return None

        indices = random.sample(range(len(self.buffer)), batch_size)
        states, damage_directions, labels, conditions = zip(
            *[self.buffer[i] for i in indices]
        )
        stacked_conditions = None
        if conditions[0] is not None:
            stacked_conditions = torch.stack(conditions).to(device)
        return (
            torch.stack(states).to(device),
            torch.stack(damage_directions).to(device),
            torch.stack(labels).to(device),
            stacked_conditions,
        )

    def __len__(self):
        return len(self.buffer)


class BaseNCA3DTrainer(ABC):
    """Shared training utilities for 3D NCA trainers."""

    def __init__(
        self,
        model,
        batch_size=8,
        lr=1e-4,
        iterations_per_epoch=100,
        min_steps_per_sample=96,
        max_steps_per_sample=128,
        buffer_size=1000,
        buffer_sampling_prob=0.5,
        device=None,
        grad_clip=1.0,
        gradient_checkpointing=False,
        wandb_project=None,
        wandb_run_name=None,
        wandb_config=None,
        wandb_watch=True,
        wandb_watch_log="all",
        wandb_watch_log_freq=100,
        wandb_log_gradient_sums=True,
        wandb_gradient_log_freq=1,
    ):
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.iterations_per_epoch = iterations_per_epoch
        if min_steps_per_sample <= 0 or max_steps_per_sample <= 0:
            raise ValueError("NCA rollout step bounds must be positive.")
        if min_steps_per_sample > max_steps_per_sample:
            raise ValueError("min_steps_per_sample cannot exceed max_steps_per_sample.")
        self.min_steps_per_sample = min_steps_per_sample
        self.max_steps_per_sample = max_steps_per_sample
        self.buffer_sampling_prob = buffer_sampling_prob
        self.grad_clip = grad_clip
        self.gradient_checkpointing = gradient_checkpointing
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        self.wandb_watch = wandb_watch
        self.wandb_watch_log = wandb_watch_log
        self.wandb_watch_log_freq = wandb_watch_log_freq
        self.wandb_log_gradient_sums = wandb_log_gradient_sums
        self.wandb_gradient_log_freq = wandb_gradient_log_freq
        self.wandb_initialized = False
        self.wandb = None
        self.global_step = 0
        self.device = preferred_device(device)

        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size, buffer_sampling_prob)

    def wandb_base_config(self):
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "iterations_per_epoch": self.iterations_per_epoch,
            "min_steps_per_sample": self.min_steps_per_sample,
            "max_steps_per_sample": self.max_steps_per_sample,
            "buffer_size": self.replay_buffer.buffer.maxlen,
            "buffer_sampling_prob": self.buffer_sampling_prob,
            "grad_clip": self.grad_clip,
            "gradient_checkpointing": self.gradient_checkpointing,
            "device": str(self.device),
            "wandb_watch": self.wandb_watch,
            "wandb_watch_log": self.wandb_watch_log,
            "wandb_watch_log_freq": self.wandb_watch_log_freq,
            "wandb_log_gradient_sums": self.wandb_log_gradient_sums,
            "wandb_gradient_log_freq": self.wandb_gradient_log_freq,
            **self.wandb_config,
        }

    def init_wandb(self, extra_config=None):
        if self.wandb_initialized or not self.wandb_project:
            return self.wandb

        import wandb

        config = self.wandb_base_config()
        if extra_config:
            config.update(extra_config)

        self.wandb = wandb
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_run_name,
            config=config,
        )
        if self.wandb_watch:
            wandb.watch(
                self.model,
                log=self.wandb_watch_log,
                log_freq=self.wandb_watch_log_freq,
            )
        self.wandb_initialized = True
        return wandb

    def log_wandb(self, metrics, step=None):
        wandb = self.init_wandb()
        if wandb is None:
            return
        wandb.log(metrics, step=step)

    def gradient_metrics(self):
        metrics = {}
        total_abs_sum = 0.0
        total_norm_sq = 0.0

        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue

            grad = parameter.grad.detach()
            safe_name = name.replace(".", "/")
            abs_sum = grad.abs().sum()
            norm = grad.norm(2)
            metrics[f"gradients/{safe_name}/sum"] = grad.sum().item()
            metrics[f"gradients/{safe_name}/abs_sum"] = abs_sum.item()
            metrics[f"gradients/{safe_name}/norm"] = norm.item()
            metrics[f"gradients/{safe_name}/max_abs"] = grad.abs().max().item()

            total_abs_sum += abs_sum.item()
            total_norm_sq += norm.item() ** 2

        metrics["gradients/total_abs_sum"] = total_abs_sum
        metrics["gradients/total_norm"] = total_norm_sq**0.5
        return metrics

    def maybe_log_gradients(self):
        if not self.wandb_log_gradient_sums:
            return
        if self.wandb_gradient_log_freq <= 0:
            return
        if self.global_step % self.wandb_gradient_log_freq != 0:
            return

        metrics = self.gradient_metrics()
        if metrics:
            metrics["global_step"] = self.global_step
            self.log_wandb(metrics, step=self.global_step)

    def set_train_dataset(self, dataset, num_workers=0):
        self.train_dataset = dataset
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.dataloader = self.train_loader
        self.train_iter = iter(self.train_loader)

    def make_eval_loader(self, dataset, num_workers=0):
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def next_train_batch(self):
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)

    def prepare_damage_batch(self, batch):
        structures, damage_directions, labels, original_shapes = batch
        return (
            structures.to(self.device),
            damage_directions.to(self.device),
            labels.to(self.device),
            original_shapes.to(self.device),
        )

    def should_use_replay_buffer(self):
        return (
            len(self.replay_buffer) >= self.batch_size
            and random.random() < self.buffer_sampling_prob
        )

    def sample_replay_buffer(self):
        return self.replay_buffer.sample(self.batch_size, self.device)

    def run_nca(self, states, condition=None):
        states_history = [states.detach()]
        steps_per_sample = random.randint(
            self.min_steps_per_sample,
            self.max_steps_per_sample,
        )
        for _ in range(steps_per_sample):
            if condition is None:
                if self.gradient_checkpointing and self.model.training:
                    states = checkpoint(self.model, states, use_reentrant=False)
                else:
                    states = self.model(states)
            else:
                if self.gradient_checkpointing and self.model.training:
                    states = checkpoint(
                        lambda current_states: self.model(current_states, condition),
                        states,
                        use_reentrant=False,
                    )
                else:
                    states = self.model(states, condition)
            states_history.append(states.detach())
        return states, states_history

    def add_replay_states(
        self,
        states_history,
        damage_directions,
        labels,
        conditions,
        num_samples=1,
    ):
        if not self.replay_buffer.buffer.maxlen or len(states_history) <= 1:
            return

        max_step = len(states_history) - 1
        min_step = min(max(1, max_step // 2), max_step)
        for _ in range(num_samples):
            step_idx = random.randint(min_step, max_step)
            self.replay_buffer.add(
                states_history[step_idx],
                damage_directions,
                labels,
                conditions,
            )

    def replay_samples_per_fresh_batch(self):
        return 1

    def rollout_condition(self, labels, original_shapes):
        if getattr(self.model, "use_shape_conditioning", False):
            return original_shapes
        if getattr(self.model, "use_class_embeddings", False):
            return labels
        return None

    def train_epoch(self, epoch):
        """Run one training epoch using the shared replay/rollout loop."""
        self.model.train()
        totals = {}
        batch_count = 0
        iterations = self.iterations_per_epoch or len(self.train_loader)

        from tqdm import tqdm

        pbar = tqdm(range(iterations), desc=f"Epoch {epoch}")
        for _ in pbar:
            use_buffer = self.should_use_replay_buffer()

            if use_buffer:
                sampled = self.sample_replay_buffer()
                if sampled is None:
                    use_buffer = False
                else:
                    states, damage_directions, labels, conditions = sampled

            if not use_buffer:
                (
                    structures,
                    damage_directions,
                    labels,
                    original_shapes,
                ) = self.prepare_damage_batch(self.next_train_batch())
                states = self.model.initialize(structures)
                conditions = self.rollout_condition(labels, original_shapes)

            final_state, states_history = self.run_nca(
                states,
                conditions,
            )

            if not use_buffer:
                self.add_replay_states(
                    states_history,
                    damage_directions,
                    labels,
                    conditions,
                    num_samples=self.replay_samples_per_fresh_batch(),
                )

            loss, metrics = self.loss_and_metrics(
                final_state,
                damage_directions,
                labels,
            )
            step = self.optimizer_step(loss)

            batch_count += 1
            metrics = self.metric_items(metrics)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value

            self.log_train_step_metrics(metrics, step)
            averages = {key: value / batch_count for key, value in totals.items()}
            pbar.set_postfix(self.progress_postfix(averages))

        return {key: value / batch_count for key, value in totals.items()}

    def metric_items(self, metrics):
        result = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                result[key] = value.detach().item()
            else:
                result[key] = float(value)
        return result

    def log_train_step_metrics(self, metrics, step):
        self.log_wandb(
            {f"train_step/{key}": value for key, value in metrics.items()},
            step=step,
        )

    def progress_postfix(self, averages):
        return {
            "loss": f"{averages.get('loss', 0.0):.4f}",
        }

    @abstractmethod
    def loss_and_metrics(self, final_state, damage_directions, labels):
        """Return `(loss, metrics)` for a completed NCA rollout."""

    def alive_mask(self, final_state):
        return (final_state[:, :, :, :, 0] > self.model.alpha_living_threshold).float()

    def masked_cross_entropy(self, loss_fn, predictions, targets, final_state):
        batch, _, _, _, n_class = predictions.shape
        pred_flat = (
            predictions.permute(0, 4, 1, 2, 3).contiguous().view(batch, n_class, -1)
        )
        targets_flat = targets.view(batch, -1)
        cell_loss = loss_fn(pred_flat, targets_flat)
        alive_mask_flat = self.alive_mask(final_state).view(batch, -1)
        return (cell_loss * alive_mask_flat).sum() / (alive_mask_flat.sum() + 1e-8)

    def masked_focal_loss(self, loss_fn, predictions, targets, final_state, gamma=2.0):
        batch, _, _, _, n_class = predictions.shape
        pred_flat = (
            predictions.permute(0, 4, 1, 2, 3).contiguous().view(batch, n_class, -1)
        )
        targets_flat = targets.view(batch, -1)
        cell_loss = loss_fn(pred_flat, targets_flat)
        log_probs = F.log_softmax(pred_flat, dim=1)
        target_log_probs = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt = target_log_probs.exp()
        focal_loss = ((1.0 - pt) ** gamma) * cell_loss
        alive_mask_flat = self.alive_mask(final_state).view(batch, -1)
        return (focal_loss * alive_mask_flat).sum() / (alive_mask_flat.sum() + 1e-8)

    def masked_damage_loss(
        self,
        loss_fn,
        predictions,
        targets,
        final_state,
        loss_type="cross_entropy",
        focal_gamma=2.0,
    ):
        if loss_type == "cross_entropy":
            return self.masked_cross_entropy(loss_fn, predictions, targets, final_state)
        if loss_type == "focal":
            return self.masked_focal_loss(
                loss_fn,
                predictions,
                targets,
                final_state,
                gamma=focal_gamma,
            )
        raise ValueError(f"Unsupported damage loss type: {loss_type}")

    def clipping_loss(self, final_state):
        batch = final_state.shape[0]
        with torch.no_grad():
            clipped_states = torch.clamp(
                final_state.detach(),
                -self.model.clip_range,
                self.model.clip_range,
            )
        clip_loss = (
            F.mse_loss(final_state, clipped_states, reduction="none")
            .sum(dim=-1)
            .view(batch, -1)
        )
        alive_mask_flat = self.alive_mask(final_state).view(batch, -1)
        return (clip_loss * alive_mask_flat).sum() / (alive_mask_flat.sum() + 1e-8)

    def damage_accuracy(self, predictions, targets, final_state):
        pred_classes = torch.argmax(predictions, dim=-1)
        correct = (pred_classes == targets).float()
        alive_mask = self.alive_mask(final_state)
        return (correct * alive_mask).sum() / (alive_mask.sum() + 1e-8)

    def damaged_accuracy(self, predictions, targets, final_state):
        pred_classes = torch.argmax(predictions, dim=-1)
        correct = (pred_classes == targets).float()
        alive_mask = self.alive_mask(final_state)
        damaged_mask = ((targets > 0).float() * alive_mask).float()
        return (correct * damaged_mask).sum() / (damaged_mask.sum() + 1e-8)

    def damage_diagnostics(self, predictions, targets, final_state):
        pred_classes = torch.argmax(predictions, dim=-1)
        alive_mask = self.alive_mask(final_state).bool()
        target_nonzero = (targets > 0) & alive_mask
        predicted_nonzero = (pred_classes > 0) & alive_mask
        true_positive = predicted_nonzero & target_nonzero
        alive_count = alive_mask.float().sum().clamp_min(1.0)
        predicted_count = predicted_nonzero.float().sum()
        target_count = target_nonzero.float().sum()
        precision = true_positive.float().sum() / predicted_count.clamp_min(1.0)
        recall = true_positive.float().sum() / target_count.clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        diagnostics = {
            "target_nonzero_fraction": target_count / alive_count,
            "predicted_nonzero_fraction": predicted_count / alive_count,
            "nonzero_precision": precision,
            "nonzero_recall": recall,
            "nonzero_f1": f1,
        }

        num_directions = getattr(self.model, "num_damage_directions", 7)
        for direction in range(1, num_directions):
            direction_mask = (targets == direction) & alive_mask
            direction_count = direction_mask.float().sum()
            direction_correct = (
                ((pred_classes == targets) & direction_mask).float().sum()
            )
            diagnostics[f"direction_{direction}_accuracy"] = (
                direction_correct / direction_count.clamp_min(1.0)
            )
            diagnostics[f"direction_{direction}_count"] = direction_count

        return diagnostics

    def optimizer_step(self, loss):
        step = self.global_step
        self.optimizer.zero_grad()
        loss.backward()
        self.maybe_log_gradients()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.global_step += 1
        return step

    def checkpoint_state(self, epoch, extra=None):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if extra:
            checkpoint.update(extra)
        return checkpoint

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]

    @abstractmethod
    def save_model(self, epoch, loss):
        """Save a model checkpoint."""

    @abstractmethod
    def train(self, epochs, save_frequency=5, visualization_frequency=10):
        """Run the full training loop."""
