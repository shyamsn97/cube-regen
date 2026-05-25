import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class ReplayBuffer:
    """Replay buffer for intermediate NCA states and their targets."""

    def __init__(self, buffer_size=1000, sampling_prob=0.5):
        self.buffer = deque(maxlen=buffer_size)
        self.sampling_prob = sampling_prob

    def add(self, states, damage_directions, labels):
        for i in range(states.shape[0]):
            self.buffer.append(
                (
                    states[i].detach().cpu().clone(),
                    damage_directions[i].detach().cpu().clone(),
                    labels[i].detach().cpu().clone(),
                )
            )

    def sample(self, batch_size, device):
        if len(self.buffer) < batch_size:
            return None

        indices = random.sample(range(len(self.buffer)), batch_size)
        states, damage_directions, labels = zip(*[self.buffer[i] for i in indices])
        return (
            torch.stack(states).to(device),
            torch.stack(damage_directions).to(device),
            torch.stack(labels).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class BaseNCA3DTrainer:
    """Shared training utilities for 3D NCA trainers."""

    def __init__(
        self,
        model,
        batch_size=8,
        lr=1e-4,
        iterations_per_epoch=100,
        steps_per_sample=96,
        buffer_size=1000,
        buffer_sampling_prob=0.5,
        device=None,
        grad_clip=1.0,
    ):
        self.model = model
        self.batch_size = batch_size
        self.iterations_per_epoch = iterations_per_epoch
        self.steps_per_sample = steps_per_sample
        self.buffer_sampling_prob = buffer_sampling_prob
        self.grad_clip = grad_clip
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size, buffer_sampling_prob)

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
        structures, damage_directions, labels, _ = batch
        return (
            structures.to(self.device),
            damage_directions.to(self.device),
            labels.to(self.device),
        )

    def should_use_replay_buffer(self):
        return (
            len(self.replay_buffer) >= self.batch_size
            and random.random() < self.buffer_sampling_prob
        )

    def sample_replay_buffer(self):
        return self.replay_buffer.sample(self.batch_size, self.device)

    def run_nca(self, states, labels=None):
        states_history = [states]
        for _ in range(self.steps_per_sample):
            if labels is None:
                states = self.model(states)
            else:
                states = self.model(states, labels)
            states_history.append(states)
        return states, states_history

    def add_replay_states(
        self,
        states_history,
        damage_directions,
        labels,
        num_samples=1,
    ):
        if not self.replay_buffer.buffer.maxlen or len(states_history) <= 1:
            return

        max_step = len(states_history) - 1
        min_step = min(max(1, self.steps_per_sample // 2), max_step)
        for _ in range(num_samples):
            step_idx = random.randint(min_step, max_step)
            self.replay_buffer.add(
                states_history[step_idx],
                damage_directions,
                labels,
            )

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

    def optimizer_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

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
