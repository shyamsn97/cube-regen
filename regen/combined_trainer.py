import json
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from regen.base_trainer import BaseNCA3DTrainer


class CombinedNCA3DTrainer(BaseNCA3DTrainer):
    """Trainer for joint damage direction detection and shape classification."""

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size: int = 8,
        lr: float = 2e-5,
        iterations_per_epoch: Optional[int] = None,
        steps_per_sample: int = 96,
        damage_loss_weight: float = 1.0,
        class_loss_weight: float = 1.0,
        buffer_size: int = 1000,
        buffer_sampling_prob: float = 0.5,
        grad_clip: float = 1.0,
        device=None,
        checkpoint_dir: str = "./combined_nca_models",
        num_workers: int = 0,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        super().__init__(
            model=model,
            batch_size=batch_size,
            lr=lr,
            iterations_per_epoch=iterations_per_epoch,
            steps_per_sample=steps_per_sample,
            buffer_size=buffer_size,
            buffer_sampling_prob=buffer_sampling_prob,
            device=device,
            grad_clip=grad_clip,
        )
        self.damage_loss_weight = damage_loss_weight
        self.class_loss_weight = class_loss_weight
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float("inf")

        damage_weights = torch.ones(self.model.num_damage_directions)
        self.damage_loss_fn = nn.CrossEntropyLoss(
            weight=damage_weights.to(self.device), reduction="none"
        )
        self.class_loss_fn = nn.CrossEntropyLoss()

        self.set_train_dataset(train_dataset, num_workers=num_workers)
        self.val_loader = self.make_eval_loader(val_dataset, num_workers=num_workers)

        os.makedirs(checkpoint_dir, exist_ok=True)

    def _losses_and_metrics(self, final_state, damage_directions, labels):
        damage_logits = self.model.damage_logits(final_state)
        class_logits = self.model.class_logits(final_state)

        damage_loss = self.masked_cross_entropy(
            self.damage_loss_fn,
            damage_logits,
            damage_directions,
            final_state,
        )

        class_loss = self.class_loss_fn(class_logits, labels)
        loss = (
            self.damage_loss_weight * damage_loss + self.class_loss_weight * class_loss
        )

        if not self.model.use_tanh:
            loss = loss + self.clipping_loss(final_state)

        with torch.no_grad():
            damage_predictions = torch.argmax(damage_logits, dim=-1)
            class_predictions = torch.argmax(class_logits, dim=-1)

            alive_mask = self.alive_mask(final_state)
            damage_correct = (damage_predictions == damage_directions).float()
            damage_accuracy = (damage_correct * alive_mask).sum() / (
                alive_mask.sum() + 1e-8
            )

            damaged_mask = ((damage_directions > 0).float() * alive_mask).float()
            damaged_accuracy = (damage_correct * damaged_mask).sum() / (
                damaged_mask.sum() + 1e-8
            )
            class_accuracy = (class_predictions == labels).float().mean()

        return loss, {
            "loss": loss.detach(),
            "damage_loss": damage_loss.detach(),
            "class_loss": class_loss.detach(),
            "damage_accuracy": damage_accuracy.detach(),
            "damaged_accuracy": damaged_accuracy.detach(),
            "class_accuracy": class_accuracy.detach(),
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals = {}
        batch_count = 0
        iterations = self.iterations_per_epoch or len(self.train_loader)

        pbar = tqdm(range(iterations), desc=f"Epoch {epoch}")
        for _ in pbar:
            use_buffer = self.should_use_replay_buffer()

            if use_buffer:
                sampled = self.sample_replay_buffer()
                if sampled is None:
                    use_buffer = False
                else:
                    states, damage_directions, labels = sampled

            if not use_buffer:
                structures, damage_directions, labels = self.prepare_damage_batch(
                    self.next_train_batch()
                )
                states = self.model.initialize(structures)

            final_state, states_history = self.run_nca(states)

            if not use_buffer:
                self.add_replay_states(
                    states_history,
                    damage_directions,
                    labels,
                )

            loss, metrics = self._losses_and_metrics(
                final_state,
                damage_directions,
                labels,
            )

            self.optimizer_step(loss)

            batch_count += 1
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value.item()

            averages = {key: value / batch_count for key, value in totals.items()}
            pbar.set_postfix(
                {
                    "loss": f"{averages['loss']:.4f}",
                    "dmg_acc": f"{averages['damage_accuracy']:.3f}",
                    "cls_acc": f"{averages['class_accuracy']:.3f}",
                }
            )

        return {key: value / batch_count for key, value in totals.items()}

    @torch.no_grad()
    def validate(self) -> Optional[Dict[str, float]]:
        if self.val_loader is None:
            return None

        self.model.eval()
        totals = {}
        batch_count = 0

        for batch in self.val_loader:
            structures, damage_directions, labels = self.prepare_damage_batch(batch)
            states = self.model.initialize(structures)
            final_state, _ = self.run_nca(states)

            _, metrics = self._losses_and_metrics(
                final_state,
                damage_directions,
                labels,
            )
            batch_count += 1
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value.item()

        return {key: value / batch_count for key, value in totals.items()}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best=False):
        checkpoint = self.checkpoint_state(
            epoch,
            {
                "metrics": metrics,
                "model_config": self.model.get_config(),
            },
        )
        latest_path = os.path.join(self.checkpoint_dir, "combined_latest.pt")
        torch.save(checkpoint, latest_path)

        epoch_path = os.path.join(self.checkpoint_dir, f"combined_epoch_{epoch}.pt")
        torch.save(checkpoint, epoch_path)

        config_path = os.path.join(self.checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.model.get_config(), f, indent=2)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "combined_best.pt")
            torch.save(checkpoint, best_path)

    def fit(
        self,
        epochs: int,
        save_frequency: int = 10,
        validate_frequency: int = 1,
    ):
        for epoch in range(epochs):
            start = time.time()
            train_metrics = self.train_epoch(epoch)
            metrics = {f"train_{key}": value for key, value in train_metrics.items()}

            val_metrics = None
            should_validate = (
                self.val_loader is not None
                and validate_frequency > 0
                and (epoch % validate_frequency == 0 or epoch == epochs - 1)
            )
            if should_validate:
                val_metrics = self.validate()
                metrics.update(
                    {f"val_{key}": value for key, value in val_metrics.items()}
                )

            monitor_loss = (
                val_metrics["loss"]
                if val_metrics is not None
                else train_metrics["loss"]
            )
            is_best = monitor_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = monitor_loss

            elapsed = time.time() - start
            log_parts = [
                f"Epoch {epoch}",
                f"train_loss={train_metrics['loss']:.4f}",
                f"train_damage_acc={train_metrics['damage_accuracy']:.3f}",
                f"train_class_acc={train_metrics['class_accuracy']:.3f}",
            ]
            if val_metrics is not None:
                log_parts.extend(
                    [
                        f"val_loss={val_metrics['loss']:.4f}",
                        f"val_damage_acc={val_metrics['damage_accuracy']:.3f}",
                        f"val_class_acc={val_metrics['class_accuracy']:.3f}",
                    ]
                )
            log_parts.append(f"time={elapsed:.1f}s")
            print(" | ".join(log_parts))

            should_save = save_frequency > 0 and (
                epoch % save_frequency == 0 or epoch == epochs - 1
            )
            if should_save or is_best:
                self.save_checkpoint(epoch, metrics, is_best=is_best)
