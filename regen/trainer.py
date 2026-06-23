import os
import time

import numpy as np
import torch
import torch.nn as nn

# Combine images side by side
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        device=None,
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
        )
        self.save_dir = save_dir

        # Initialize loss function (ignore predictions for "dead" cells)
        # Use class weights to emphasize damage indices (1-6) more than no-damage (0)
        damage_weights = torch.ones(self.model.num_damage_directions)
        damage_weights[1:] = 1.0  # Higher weight for damage indices (1-6)
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
        masked_loss = self.masked_cross_entropy(
            self.loss_fn,
            predictions,
            targets,
            final_state,
        )

        if not self.model.use_tanh:
            masked_loss += self.clipping_loss(final_state)

        return masked_loss

    def train_epoch(self, epoch):
        """Run a single training epoch."""
        self.model.train()
        epoch_loss = 0
        batch_count = 0

        pbar = tqdm(range(self.iterations_per_epoch), desc=f"Epoch {epoch}")
        for _ in pbar:
            # Decide whether to sample from buffer or use fresh samples
            use_buffer = self.should_use_replay_buffer()

            if use_buffer:
                # Sample from replay buffer
                sampled = self.sample_replay_buffer()
                if sampled is None:
                    use_buffer = False
                else:
                    states, damage_directions, labels = sampled

            if not use_buffer:
                # Get a fresh batch from the dataset
                structures, damage_directions, labels = self.prepare_damage_batch(
                    self.next_train_batch()
                )
                # Initialize states from structures
                states = self.model.initialize(structures)

            # Run NCA for several steps
            final_state, states_history = self.run_nca(states, labels)

            # Add some intermediate states to the replay buffer
            if not use_buffer:  # Only add if we're not already using buffer samples
                # Add some random intermediate states to buffer
                self.add_replay_states(
                    states_history,
                    damage_directions,
                    labels,
                    num_samples=2,
                )

            # Get predictions from final state
            predictions = self.model.classify(final_state)

            # Calculate loss using alive pixels
            # Apply higher weight to actual damage areas
            # damage_mask = (damage_directions > 0).float()
            # weights = 1.0 + damage_mask * 0.5  # 1.5x weight for actual damage areas

            loss = self.loss_function(
                predictions,
                damage_directions,
                final_state,
            )

            # Backpropagation
            self.optimizer_step(loss)

            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1

            # Calculate accuracy
            accuracy = self.damage_accuracy(predictions, damage_directions, final_state)

            # Update progress bar with loss and accuracy
            batch_epoch_loss = epoch_loss / batch_count
            pbar.set_postfix(
                {
                    "loss": f"{batch_epoch_loss:.4f}",
                    "acc": f"{accuracy.item():.4f}",
                }
            )
        return batch_epoch_loss

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
        if not hasattr(self, "wandb_initialized"):
            import wandb
            from torch.utils.tensorboard import SummaryWriter

            wandb.init(
                project="nca-3d-damage-detection",
                config={
                    "batch_size": self.batch_size,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "steps_per_sample": self.steps_per_sample,
                    "buffer_size": self.replay_buffer.buffer.maxlen,
                    "buffer_sampling_prob": self.buffer_sampling_prob,
                },
            )
            self.wandb_initialized = True
            self.writer = SummaryWriter(log_dir=f"{self.save_dir}/tensorboard")

        for epoch in range(epochs):
            start_time = time.time()

            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Calculate accuracy
            self.model.eval()
            correct = 0
            total = 0
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

            accuracy = 100 * correct / total
            self.model.train()

            print(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {time.time() - start_time:.2f}s"
            )
            # Log metrics to wandb
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "accuracy": accuracy,
                    "epoch_time": time.time() - start_time,
                }
            )

            # Log metrics to tensorboard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/train", accuracy, epoch)
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
                wandb.log({"visualization": wandb.Image(img), "epoch": epoch})
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
            print(predictions.shape)
            print(damage_mask_tensor.detach().cpu().numpy().shape)
            print(damage_direction_tensor.detach().cpu().numpy().shape)

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
