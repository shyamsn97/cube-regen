import json
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from huggingface_hub import HfApi

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
        damage_class_weight: float = 1.0,
        damage_loss_type: str = "cross_entropy",
        focal_gamma: float = 2.0,
        buffer_size: int = 1000,
        buffer_sampling_prob: float = 0.5,
        grad_clip: float = 1.0,
        gradient_checkpointing: bool = False,
        device=None,
        wandb_project: Optional[str] = "nca-3d-combined",
        wandb_run_name: Optional[str] = None,
        wandb_watch: bool = True,
        wandb_watch_log: str = "all",
        wandb_watch_log_freq: int = 100,
        wandb_log_gradient_sums: bool = True,
        wandb_gradient_log_freq: int = 1,
        checkpoint_dir: str = "./combined_nca_models",
        num_workers: int = 0,
        validate_frequency: int = 1,
        repo_id: Optional[str] = None,
        repo_type: str = "model",
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
            gradient_checkpointing=gradient_checkpointing,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_config={
                "trainer": "CombinedNCA3DTrainer",
                "optimizer": "Adam",
                "learning_rate": lr,
                "batch_size": batch_size,
                "iterations_per_epoch": iterations_per_epoch,
                "steps_per_sample": steps_per_sample,
                "buffer_size": buffer_size,
                "buffer_sampling_prob": buffer_sampling_prob,
                "grad_clip": grad_clip,
                "gradient_checkpointing": gradient_checkpointing,
                "wandb_watch": wandb_watch,
                "wandb_watch_log": wandb_watch_log,
                "wandb_watch_log_freq": wandb_watch_log_freq,
                "wandb_log_gradient_sums": wandb_log_gradient_sums,
                "wandb_gradient_log_freq": wandb_gradient_log_freq,
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
                "checkpoint_dir": checkpoint_dir,
                "repo_id": repo_id,
                "repo_type": repo_type,
            },
            wandb_watch=wandb_watch,
            wandb_watch_log=wandb_watch_log,
            wandb_watch_log_freq=wandb_watch_log_freq,
            wandb_log_gradient_sums=wandb_log_gradient_sums,
            wandb_gradient_log_freq=wandb_gradient_log_freq,
        )
        self.damage_loss_weight = damage_loss_weight
        self.class_loss_weight = class_loss_weight
        self.damage_class_weight = damage_class_weight
        self.damage_loss_type = damage_loss_type
        self.focal_gamma = focal_gamma
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float("inf")
        self.validate_frequency = validate_frequency
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.train_losses = []
        self.val_losses = []

        damage_weights = torch.ones(self.model.num_damage_directions)
        damage_weights[1:] = damage_class_weight
        self.damage_loss_fn = nn.CrossEntropyLoss(
            weight=damage_weights.to(self.device), reduction="none"
        )
        self.class_loss_fn = nn.CrossEntropyLoss()

        self.set_train_dataset(train_dataset, num_workers=num_workers)
        self.val_loader = self.make_eval_loader(val_dataset, num_workers=num_workers)

        os.makedirs(checkpoint_dir, exist_ok=True)

    def loss_and_metrics(self, final_state, damage_directions, labels):
        damage_logits = self.model.damage_logits(final_state)
        class_logits = self.model.class_logits(final_state)

        damage_loss = self.masked_damage_loss(
            self.damage_loss_fn,
            damage_logits,
            damage_directions,
            final_state,
            loss_type=self.damage_loss_type,
            focal_gamma=self.focal_gamma,
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

        metrics = {
            "loss": loss.detach(),
            "damage_loss": damage_loss.detach(),
            "class_loss": class_loss.detach(),
            "full_accuracy": damage_accuracy.detach(),
            "damage_accuracy": damage_accuracy.detach(),
            "damaged_accuracy": damaged_accuracy.detach(),
            "damage_boundary_accuracy": damaged_accuracy.detach(),
            "class_accuracy": class_accuracy.detach(),
        }
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

    def loss_function(self, final_state, damage_directions, labels):
        return self.loss_and_metrics(final_state, damage_directions, labels)

    def log_train_step_metrics(self, metrics, step):
        self.log_wandb(
            {f"train_step/{key}": value for key, value in metrics.items()},
            step=step,
        )

    def progress_postfix(self, averages):
        return {
            "loss": f"{averages.get('loss', 0.0):.4f}",
            "full_acc": f"{averages.get('damage_accuracy', 0.0):.3f}",
            "dmg_acc": f"{averages.get('damaged_accuracy', 0.0):.3f}",
            "cls_acc": f"{averages.get('class_accuracy', 0.0):.3f}",
        }

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

            _, metrics = self.loss_function(
                final_state,
                damage_directions,
                labels,
            )
            batch_count += 1
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value.item()

        return {key: value / batch_count for key, value in totals.items()}

    def save_model(
        self,
        epoch,
        loss,
        metrics: Optional[Dict[str, float]] = None,
        is_best=False,
    ):
        if metrics is None:
            metrics = {"loss": loss}
        print(
            f"Saving combined checkpoint for epoch {epoch} "
            f"(loss={loss:.4f}) to {self.checkpoint_dir}"
        )
        checkpoint = self.checkpoint_state(
            epoch,
            {
                "metrics": metrics,
                "model_config": self.model.get_config(),
            },
        )
        latest_path = os.path.join(self.checkpoint_dir, "combined_latest.pt")
        torch.save(checkpoint, latest_path)
        print(f"Saved local checkpoint: {latest_path}")

        epoch_path = os.path.join(self.checkpoint_dir, f"combined_epoch_{epoch}.pt")
        torch.save(checkpoint, epoch_path)
        print(f"Saved local checkpoint: {epoch_path}")

        config_path = os.path.join(self.checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.model.get_config(), f, indent=2)
        print(f"Saved model config: {config_path}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "combined_best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")

        if self.repo_id:
            print(
                "Uploading combined checkpoint artifacts to "
                f"Hugging Face {self.repo_type} repo '{self.repo_id}'"
            )
            self.upload_checkpoint(
                epoch=epoch,
                latest_path=latest_path,
                epoch_path=epoch_path,
                config_path=config_path,
                best_path=best_path if is_best else None,
            )
        else:
            print("No Hugging Face repo configured for combined checkpoint upload.")

    def upload_checkpoint(
        self,
        epoch,
        latest_path,
        epoch_path,
        config_path,
        best_path=None,
    ):
        api = HfApi()
        api.create_repo(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            exist_ok=True,
        )

        uploads = [
            (latest_path, "combined_latest.pt"),
            (epoch_path, f"combined_epoch_{epoch}.pt"),
            (config_path, "config.json"),
        ]
        if best_path:
            uploads.append((best_path, "combined_best.pt"))

        for local_path, path_in_repo in uploads:
            url = api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                commit_message=f"Save combined checkpoint for epoch {epoch}",
            )
            print(
                "Uploaded combined checkpoint artifact to "
                f"Hugging Face {self.repo_type} repo '{self.repo_id}': "
                f"{path_in_repo} ({url})"
            )

    def train(
        self,
        epochs,
        save_frequency=5,
        visualization_frequency=10,
    ):
        del visualization_frequency

        self.init_wandb()
        for epoch in range(epochs):
            start = time.time()
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics["loss"])
            metrics = {f"train_{key}": value for key, value in train_metrics.items()}

            val_metrics = None
            should_validate = (
                self.val_loader is not None
                and self.validate_frequency > 0
                and (epoch % self.validate_frequency == 0 or epoch == epochs - 1)
            )
            if should_validate:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics["loss"])
                metrics.update(
                    {f"val_{key}": value for key, value in val_metrics.items()}
                )

            monitor_loss = (
                val_metrics["loss"]
                if val_metrics is not None
                else train_metrics["loss"]
            )
            elapsed = time.time() - start
            log_parts = [
                f"Epoch {epoch}",
                f"train_loss={train_metrics['loss']:.4f}",
                f"train_full_acc={train_metrics['damage_accuracy']:.3f}",
                f"train_damage_acc={train_metrics['damaged_accuracy']:.3f}",
                f"train_class_acc={train_metrics['class_accuracy']:.3f}",
            ]
            if val_metrics is not None:
                log_parts.extend(
                    [
                        f"val_loss={val_metrics['loss']:.4f}",
                        f"val_full_acc={val_metrics['damage_accuracy']:.3f}",
                        f"val_damage_acc={val_metrics['damaged_accuracy']:.3f}",
                        f"val_class_acc={val_metrics['class_accuracy']:.3f}",
                    ]
                )
            log_parts.append(f"time={elapsed:.1f}s")
            print(" | ".join(log_parts))

            wandb_metrics = {
                "epoch": epoch,
                "epoch_time": elapsed,
                **metrics,
            }
            self.log_wandb(wandb_metrics, step=self.global_step)

            should_save = save_frequency > 0 and (
                epoch % save_frequency == 0 or epoch == epochs - 1
            )
            if should_save:
                is_best = monitor_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = monitor_loss
                self.save_model(epoch, monitor_loss, metrics=metrics, is_best=is_best)
