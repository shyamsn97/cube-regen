import os
import time

import numpy as np
import torch
import torch.nn as nn

# Combine images side by side
from PIL import Image
from torch.utils.data import DataLoader

from regen.base_trainer import BaseNCA3DTrainer, ReplayBuffer
from regen.model import save_weights_to_huggingface
from regen.utils import plot_voxels, save_weights

__all__ = ["ReplayBuffer", "NCA3DTrainer"]


class NCA3DTrainer(BaseNCA3DTrainer):
    def __init__(
        self,
        model,
        dataset,
        batch_size=8,
        lr=1e-4,
        iterations_per_epoch=100,
        steps_per_sample=96,
        buffer_size=1000,
        buffer_sampling_prob=0.5,
        grad_clip=1.0,
        gradient_checkpointing=False,
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
        repo_id="shyamsn97/cube",
        repo_type="model",
        model_repo_id="shyamsn97/cube-regen-damage-detection",
    ):
        """
        Train a 3D NCA model for damage detection.

        Args:
            model: The NCA3DDamageDetection model
            dataset: Dataset with 3D shapes and damage labels
            batch_size: Training batch size
            lr: Learning rate
            iterations_per_epoch: Number of iterations per epoch
            steps_per_sample: Number of NCA steps to run per sample
            buffer_size: Size of the replay buffer
            buffer_sampling_prob: Probability of sampling from buffer vs. from scratch
            device: Device to train on (cpu or cuda)
            save_dir: Directory to save models
            repo_id: Repository ID for weights
        """
        self.dataset = dataset
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
                "trainer": "NCA3DTrainer",
                "optimizer": "Adam",
                "learning_rate": lr,
                "batch_size": batch_size,
                "iterations_per_epoch": iterations_per_epoch,
                "steps_per_sample": steps_per_sample,
                "buffer_size": buffer_size,
                "buffer_sampling_prob": buffer_sampling_prob,
                "grad_clip": grad_clip,
                "gradient_checkpointing": gradient_checkpointing,
                "damage_class_weight": damage_class_weight,
                "damage_loss_type": damage_loss_type,
                "focal_gamma": focal_gamma,
                "model": model.get_config(),
                "dataset_size": len(dataset),
                "save_dir": save_dir,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "model_repo_id": model_repo_id,
            },
            wandb_watch=wandb_watch,
            wandb_watch_log=wandb_watch_log,
            wandb_watch_log_freq=wandb_watch_log_freq,
            wandb_log_gradient_sums=wandb_log_gradient_sums,
            wandb_gradient_log_freq=wandb_gradient_log_freq,
        )
        self.save_dir = save_dir
        self.damage_class_weight = damage_class_weight
        self.damage_loss_type = damage_loss_type
        self.focal_gamma = focal_gamma

        # Initialize loss function (ignore predictions for "dead" cells)
        # Use class weights to emphasize damage indices (1-6) more than no-damage (0)
        damage_weights = torch.ones(self.model.num_damage_directions)
        damage_weights[1:] = damage_class_weight
        self.loss_fn = nn.CrossEntropyLoss(
            weight=damage_weights.to(self.device), reduction="none"
        )

        # Create dataloader
        self.set_train_dataset(dataset)

        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.model_repo_id = model_repo_id
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def loss_function(self, predictions, targets, final_state):
        """
        Custom loss function that ignores predictions for "dead" cells.

        Args:
            predictions: Model predictions with shape [batch, depth, height, width, num_classes]
            targets: Ground truth damage labels [batch, depth, height, width]
            structure_mask: Binary mask indicating "alive" cells [batch, depth, height, width, 1]

        Returns:
            loss: Mean loss over alive cells
        """
        masked_loss = self.masked_damage_loss(
            self.loss_fn,
            predictions,
            targets,
            final_state,
            loss_type=self.damage_loss_type,
            focal_gamma=self.focal_gamma,
        )

        if not self.model.use_tanh:
            masked_loss += self.clipping_loss(final_state)

        return masked_loss

    def replay_samples_per_fresh_batch(self):
        return 2

    def loss_and_metrics(self, final_state, damage_directions, labels):
        del labels
        predictions = self.model.classify(final_state)
        loss = self.loss_function(
            predictions,
            damage_directions,
            final_state,
        )
        accuracy = self.damage_accuracy(predictions, damage_directions, final_state)
        damaged_accuracy = self.damaged_accuracy(
            predictions,
            damage_directions,
            final_state,
        )
        metrics = {
            "loss": loss.detach(),
            "full_accuracy": accuracy.detach(),
            "damage_accuracy": accuracy.detach(),
            "damaged_accuracy": damaged_accuracy.detach(),
            "damage_boundary_accuracy": damaged_accuracy.detach(),
        }
        metrics.update(
            {
                key: value.detach()
                for key, value in self.damage_diagnostics(
                    predictions,
                    damage_directions,
                    final_state,
                ).items()
            }
        )
        return loss, metrics

    def log_train_step_metrics(self, metrics, step):
        self.log_wandb(
            {f"train_step/{key}": value for key, value in metrics.items()},
            step=step,
        )

    def progress_postfix(self, averages):
        return {
            "loss": f"{averages.get('loss', 0.0):.4f}",
            "acc": f"{averages.get('full_accuracy', 0.0):.4f}",
            "dmg_acc": f"{averages.get('damaged_accuracy', 0.0):.4f}",
        }

    def save_model(self, epoch, loss):
        """Save the model checkpoint."""
        print(
            f"Saving damage checkpoint for epoch {epoch} "
            f"(loss={loss:.4f}) to {self.save_dir}"
        )
        checkpoint = self.checkpoint_state(epoch, {"loss": loss})
        checkpoint_path = f"{self.save_dir}/nca_epoch_{epoch}_loss_{loss:.4f}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved local checkpoint: {checkpoint_path}")
        print(
            "Uploading damage text/torch weights to "
            f"Hugging Face {self.repo_type} repo '{self.repo_id}'"
        )
        save_weights(self.model, epoch, repo_id=self.repo_id, repo_type=self.repo_type)
        print(
            "Uploading damage model weights to "
            f"Hugging Face model repo '{self.model_repo_id}'"
        )
        save_weights_to_huggingface(self.model, repo_id=self.model_repo_id)

    def train(self, epochs, save_frequency=5, visualization_frequency=10):
        """
        Run the full training loop.

        Args:
            epochs: Number of epochs to train
            val_dataset: Optional validation dataset
            save_frequency: How often to save model checkpoints
            visualization_frequency: How often to visualize results
        """
        from torch.utils.tensorboard import SummaryWriter

        self.init_wandb()
        self.writer = SummaryWriter(log_dir=f"{self.save_dir}/tensorboard")

        for epoch in range(epochs):
            start_time = time.time()

            # Training
            train_metrics = self.train_epoch(epoch)
            train_loss = train_metrics["loss"]
            self.train_losses.append(train_loss)

            # Calculate accuracy
            self.model.eval()
            correct = 0
            total = 0
            damaged_correct = 0
            damaged_total = 0
            with torch.no_grad():
                for batch in DataLoader(self.dataset, batch_size=self.batch_size):
                    damage_mask, damage_direction, label = self.prepare_damage_batch(
                        batch
                    )
                    # Initialize state and run NCA
                    state = self.model.initialize(damage_mask).to(self.device)
                    state, _ = self.run_nca(state, label)

                    # Get predictions
                    predictions = self.model.classify(state)
                    predicted_labels = torch.argmax(predictions, dim=-1)

                    # Calculate accuracy
                    total += damage_direction.numel()
                    correct += (predicted_labels == damage_direction).sum().item()
                    damaged_mask = damage_direction > 0
                    damaged_total += damaged_mask.sum().item()
                    damaged_correct += (
                        ((predicted_labels == damage_direction) & damaged_mask)
                        .sum()
                        .item()
                    )

            accuracy = 100 * correct / total
            damaged_accuracy = (
                100 * damaged_correct / damaged_total if damaged_total > 0 else 0.0
            )
            self.model.train()

            print(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f}, "
                f"Train Alive Acc: {train_metrics.get('damage_accuracy', 0.0):.4f}, "
                f"Train Damaged Acc: {train_metrics.get('damaged_accuracy', 0.0):.4f}, "
                f"Eval Accuracy: {accuracy:.2f}%, "
                f"Eval Damaged Accuracy: {damaged_accuracy:.2f}%, "
                f"Time: {time.time() - start_time:.2f}s"
            )
            # Log metrics to Weights & Biases
            self.log_wandb(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_alive_accuracy": train_metrics.get("damage_accuracy", 0.0),
                    "train_damaged_accuracy": train_metrics.get(
                        "damaged_accuracy", 0.0
                    ),
                    "accuracy": accuracy,
                    "damaged_accuracy": damaged_accuracy,
                    "epoch_time": time.time() - start_time,
                },
                step=self.global_step,
            )

            # Log metrics to tensorboard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar(
                "Accuracy/train_alive",
                train_metrics.get("damage_accuracy", 0.0),
                epoch,
            )
            self.writer.add_scalar(
                "Accuracy/train_damaged",
                train_metrics.get("damaged_accuracy", 0.0),
                epoch,
            )
            self.writer.add_scalar("Accuracy/train", accuracy, epoch)
            self.writer.add_scalar("Accuracy/damaged", damaged_accuracy, epoch)
            self.writer.add_scalar("Time/epoch", time.time() - start_time, epoch)

            # Save model
            if epoch % save_frequency == 0 or epoch == epochs - 1:
                self.save_model(epoch, train_loss)
                # # Log model to wandb
                # model_path = (
                #     f"{self.save_dir}/nca_epoch_{epoch}_loss_{train_loss:.4f}.pt"
                # )
                # wandb.save(model_path)

            # Visualize results
            if epoch % visualization_frequency == 0 or epoch == epochs - 1:
                img = self.visualize_results(epoch)
                if self.wandb_initialized:
                    self.log_wandb(
                        {"visualization": self.wandb.Image(img), "epoch": epoch},
                        step=self.global_step,
                    )
                self.writer.add_image(
                    "Visualization", np.array(img), epoch, dataformats="HWC"
                )

    def visualize_results(self, epoch):
        """Visualize current model predictions on a sample."""
        self.model.eval()

        with torch.no_grad():
            # Get a sample
            damage_mask_tensor, damage_direction_tensor, label, _ = self.dataset[0]
            damage_mask_tensor = damage_mask_tensor.unsqueeze(0)
            damage_direction_tensor = damage_direction_tensor.unsqueeze(0)
            label = label.unsqueeze(0)
            damage_mask_tensor = damage_mask_tensor.to(self.device)
            damage_direction_tensor = damage_direction_tensor.to(self.device)
            label = label.to(self.device)

            # Initialize state
            state = self.model.initialize(damage_mask_tensor).to(self.device)
            label = label.to(self.device)

            # Run NCA
            state, _ = self.run_nca(state, label)

            # Get final predictions
            predictions = self.model.classify(state)
            predictions = torch.argmax(predictions, dim=-1).detach().cpu().numpy()[0]

            # Convert tensors to numpy arrays and ensure they're properly shaped
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

            # Make sure predictions is properly shaped for plot_voxels
            predictions_np = predictions.astype(np.uint8)

            # Generate visualization images
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
