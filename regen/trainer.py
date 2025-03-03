import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Combine images side by side
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from regen.utils import plot_voxels, save_weights


class ReplayBuffer:
    """Replay buffer to store and sample intermediate NCA states."""

    def __init__(self, buffer_size=1000, sampling_prob=0.5):
        """
        Args:
            buffer_size: Maximum size of the buffer
            sampling_prob: Probability of sampling from the buffer vs. sampling initial states
        """
        self.buffer = deque(maxlen=buffer_size)
        self.sampling_prob = sampling_prob

    def add(self, states, damage_directions, labels):
        """Add batch of states and corresponding labels to the buffer."""
        for i in range(states.shape[0]):
            self.buffer.append(
                (
                    states[i].detach().clone(),
                    damage_directions[i].detach().clone(),
                    labels[i].detach().clone(),
                )
            )

    def sample(self, batch_size, device):
        """Sample states from the buffer."""
        if len(self.buffer) < batch_size:
            return None, None  # Buffer not filled enough yet

        indices = random.sample(range(len(self.buffer)), batch_size)
        states, damage_directions, labels = zip(*[self.buffer[i] for i in indices])

        return (
            torch.stack(states).to(device),
            torch.stack(damage_directions).to(device),
            torch.stack(labels).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class NCA3DTrainer:
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
        device=None,
        save_dir="./nca_models",
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
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.iterations_per_epoch = iterations_per_epoch
        self.steps_per_sample = steps_per_sample
        self.buffer_sampling_prob = buffer_sampling_prob
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.save_dir = save_dir

        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Initialize loss function (ignore predictions for "dead" cells)
        # Use class weights to emphasize damage indices (1-6) more than no-damage (0)
        damage_weights = torch.ones(self.model.num_damage_directions)
        damage_weights[1:] = 2.0  # Higher weight for damage indices (1-6)
        self.loss_fn = nn.CrossEntropyLoss(
            weight=damage_weights.to(self.device), reduction="none"
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, buffer_sampling_prob)

        # Create dataloader
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []

        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def loss_function(self, predictions, targets, structure_mask):
        """
        Custom loss function that ignores predictions for "dead" cells.

        Args:
            predictions: Model predictions with shape [batch, depth, height, width, num_classes]
            targets: Ground truth damage labels [batch, depth, height, width]
            structure_mask: Binary mask indicating "alive" cells [batch, depth, height, width, 1]

        Returns:
            loss: Mean loss over alive cells
        """
        # Reshape predictions for CrossEntropyLoss
        batch, d, h, w, n_class = predictions.shape
        pred_flat = (
            predictions.permute(0, 4, 1, 2, 3).contiguous().view(batch, n_class, -1)
        )
        targets_flat = targets.view(batch, -1)

        # Calculate per-cell loss
        cell_loss = self.loss_fn(pred_flat, targets_flat)

        # Mask for alive cells
        alive_mask = (
            structure_mask.view(batch, -1) > self.model.alpha_living_threshold
        ).float()

        # Apply mask and average
        masked_loss = (cell_loss * alive_mask).sum() / (alive_mask.sum() + 1e-8)

        return masked_loss

    def train_epoch(self, epoch):
        """Run a single training epoch."""
        self.model.train()
        epoch_loss = 0
        batch_count = 0

        pbar = tqdm(range(self.iterations_per_epoch), desc=f"Epoch {epoch}")
        for _ in pbar:
            # Decide whether to sample from buffer or use fresh samples
            use_buffer = (
                len(self.replay_buffer) > self.batch_size
                and random.random() < self.buffer_sampling_prob
            )

            if use_buffer:
                # Sample from replay buffer
                states, damage_directions, label = self.replay_buffer.sample(
                    self.batch_size, self.device
                )
                labels = label.to(self.device)
            else:
                # Get a fresh batch from the dataset
                try:
                    damage_mask_tensor, damage_direction_tensor, label, _ = next(
                        iter_loader
                    )
                except (StopIteration, NameError):
                    iter_loader = iter(self.dataloader)
                    damage_mask_tensor, damage_direction_tensor, label, _ = next(
                        iter_loader
                    )

                structures = damage_mask_tensor.to(self.device)
                damage_directions = damage_direction_tensor.to(self.device)
                labels = label.to(self.device)

                # Initialize states from structures
                states = self.model.initialize(structures)

            # Run NCA for several steps
            states_history = [states]
            for _ in range(self.steps_per_sample):
                states = self.model(states, labels)
                states_history.append(states)

            # Add some intermediate states to the replay buffer
            if not use_buffer:  # Only add if we're not already using buffer samples
                # Add some random intermediate states to buffer
                for i in range(2):
                    step_idx = random.randint(
                        max(1, self.steps_per_sample // 2), self.steps_per_sample - 1
                    )
                    self.replay_buffer.add(
                        states_history[step_idx], damage_directions, labels
                    )

            # Get predictions from final state
            final_state = states_history[-1]
            predictions = self.model.classify(final_state)

            # Calculate loss using alive pixels
            loss = self.loss_function(
                predictions, damage_directions, final_state[:, :, :, :, :1]
            )

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1

            # Calculate accuracy
            pred_classes = torch.argmax(predictions, dim=-1)
            target_classes = damage_directions
            correct = (pred_classes == target_classes).float()

            # Apply mask for alive cells only
            alive_mask = (
                final_state[:, :, :, :, 0] > self.model.alpha_living_threshold
            ).float()
            accuracy = (correct * alive_mask).sum() / (alive_mask.sum() + 1e-8)

            # Update progress bar with loss and accuracy
            pbar.set_postfix(
                {
                    "loss": f"{epoch_loss/batch_count:.4f}",
                    "acc": f"{accuracy.item():.4f}",
                }
            )
        return epoch_loss / batch_count

    def save_model(self, epoch, loss):
        """Save the model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, f"{self.save_dir}/nca_epoch_{epoch}_loss_{loss:.4f}.pt")
        save_weights(self.model, epoch)

    def load_model(self, checkpoint_path):
        """Load a model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]

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
                for damage_mask, damage_direction, label, _ in DataLoader(
                    self.dataset, batch_size=self.batch_size
                ):
                    damage_mask = damage_mask.to(self.device)
                    label = label.to(self.device)

                    # Initialize state and run NCA
                    state = self.model.initialize(damage_mask).to(self.device)
                    for _ in range(self.steps_per_sample):
                        state = self.model(state, label)

                    # Get predictions
                    predictions = self.model.classify(state)
                    predicted_labels = torch.argmax(predictions, dim=-1).detach().cpu()

                    # Calculate accuracy
                    total += damage_direction.size(0)
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
                # Log model to wandb
                model_path = (
                    f"{self.save_dir}/nca_epoch_{epoch}_loss_{train_loss:.4f}.pt"
                )
                wandb.save(model_path)

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
            damage_mask_tensor, damage_direction_tensor, label, _ = next(
                iter(DataLoader(self.dataset, batch_size=1, shuffle=True))
            )
            damage_mask_tensor = damage_mask_tensor.to(self.device)
            damage_direction_tensor = damage_direction_tensor.to(self.device)
            label = label.to(self.device)

            # Initialize state
            state = self.model.initialize(damage_mask_tensor).to(self.device)
            label = label.to(self.device)

            # Run NCA and collect states
            states = [state.detach().cpu().numpy()]
            for step in range(self.steps_per_sample):
                state = self.model(state, label)
                states.append(state.detach().cpu().numpy())

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
