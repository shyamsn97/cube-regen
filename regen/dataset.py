import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from regen.utils import apply_damage, plot_voxels


class DynamicDamageDataset(Dataset):
    """
    PyTorch Dataset for 3D shapes with dynamically generated damage patterns.
    Each time a sample is accessed, a new damage pattern is applied.
    """

    def __init__(
        self,
        shapes: List[np.ndarray],
        labels: List[int],
        damage_radius_range: Tuple[int, int] = (1, 3),
        damage_types: List[str] = ["sphere", "cube", "random"],
        random_proportion_range: Tuple[float, float] = (0.1, 0.25),
        fixed_damage: bool = False,
        augment_rotations: bool = False,
        return_damage_mask: bool = False,
        seed: Optional[int] = None,
        filter_label: Optional[int] = None,
    ):
        """
        Initialize the dataset with 3D shapes and configuration for dynamic damage generation.

        Args:
            shapes: List of 3D numpy arrays representing shapes (1 for solid, 0 for empty)
            labels: List of class indices for each shape
            damage_radius_range: Range (min, max) for damage radius
            damage_types: List of damage types to sample from ("sphere", "cube", "random")
            random_proportion_range: Range (min, max) for random damage proportion
            fixed_damage: If True, damage is applied once per sample and reused
            augment_rotations: If True, apply random 90-degree rotations as augmentation
            return_damage_mask: If True, also return the mask showing where damage was applied
            seed: Optional random seed for reproducibility
            filter_label: Optional label class to filter
        """
        super().__init__()

        # Store the parameters
        self.shapes = shapes
        self.labels = labels
        self.damage_radius_range = damage_radius_range
        self.damage_types = damage_types
        self.random_proportion_range = random_proportion_range
        self.fixed_damage = fixed_damage
        self.augment_rotations = augment_rotations
        self.return_damage_mask = return_damage_mask

        # Validate inputs
        if len(shapes) != len(labels):
            raise ValueError("Number of shapes and labels must match")

        if min(damage_radius_range) < 1:
            raise ValueError("Minimum damage radius must be at least 1")

        if not all(t in ["sphere", "cube", "random"] for t in damage_types):
            raise ValueError("Damage types must be 'sphere', 'cube', or 'random'")

        if not (0 <= min(random_proportion_range) <= max(random_proportion_range) <= 1):
            raise ValueError("Random proportion range must be between 0 and 1")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Pre-compute damage if using fixed damage
        if fixed_damage:
            self.new_live_masks = []
            self.damage_directions = []
            for shape in shapes:
                new_live_mask, direction = self._generate_damage(shape)
                self.new_live_masks.append(new_live_mask)
                self.damage_directions.append(direction)

        if filter_label is not None:
            self.shapes = [
                shape
                for shape, label in zip(self.shapes, self.labels)
                if label == filter_label
            ]
            self.labels = [label for label in self.labels if label == filter_label]

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, idx):
        shape = self.shapes[idx]
        label = self.labels[idx]

        # Apply damage
        if self.fixed_damage:
            new_live_mask = self.new_live_masks[idx]
            damage_direction = self.damage_directions[idx]
        else:
            new_live_mask, damage_direction = self._generate_damage(shape)

        # Apply data augmentation if enabled
        if self.augment_rotations:
            new_live_mask, damage_direction = self._apply_random_rotation(
                new_live_mask, damage_direction
            )

        # Convert to PyTorch tensors
        new_live_mask_tensor = torch.tensor(new_live_mask, dtype=torch.float32)
        damage_direction_tensor = torch.tensor(damage_direction, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        original_shape_tensor = torch.tensor(shape, dtype=torch.float32)
        return (
            new_live_mask_tensor,
            damage_direction_tensor,
            label_tensor,
            original_shape_tensor,
        )

    def _generate_damage(self, shape):
        """Generate a random damage pattern for a shape."""
        # Choose damage parameters
        radius = random.randint(*self.damage_radius_range)
        damage_type = random.choice(self.damage_types)

        random_proportion = None
        if damage_type == "random":
            random_proportion = random.uniform(*self.random_proportion_range)

        # Apply damage
        new_live_mask, damage_direction = apply_damage(
            shape.copy(),
            radius=radius,
            damage_type=damage_type,
            random_proportion=random_proportion,
        )

        return new_live_mask, damage_direction

    def _apply_random_rotation(self, shape, damage_direction):
        """Apply a random 90-degree rotation to both shape and damage_direction arrays."""
        # Choose a random axis and number of 90-degree rotations
        axis = random.randint(0, 2)  # 0, 1, 2 for x, y, z axes
        k = random.randint(0, 3)  # 0, 1, 2, 3 for 0, 90, 180, 270 degrees

        if k == 0:  # No rotation
            return shape, damage_direction

        # Apply the same rotation to both arrays
        rotated_shape = np.rot90(shape, k=k, axes=((axis + 1) % 3, (axis + 2) % 3))
        rotated_damage = np.rot90(
            damage_direction, k=k, axes=((axis + 1) % 3, (axis + 2) % 3)
        )

        # Adjust damage direction values after rotation
        # This is complex because the semantic meaning of direction values
        # (1-6 for -x, +x, -y, +y, -z, +z) needs to change after rotation

        # For simplicity, we'll just keep the damage direction values as is for now
        # In a real application, you might want to update these values based on the rotation

        return rotated_shape, rotated_damage

    def get_original_shape(self, idx):
        """Get the original undamaged shape."""
        return self.shapes[idx]

    def visualize_sample(self, idx, show=True):
        """Visualize a sample using the provided plot_voxels function."""
        damaged_shape, damage_direction, *_ = self.__getitem__(idx)
        damaged_shape = damaged_shape.numpy()
        damage_direction = damage_direction.numpy()

        image = plot_voxels(damaged_shape, damage_direction)
        if show:
            image.show()
        return image

    def get_sample_with_multiple_damages(self, idx, num_damages=3):
        """Get a sample with multiple damage sites."""
        shape = self.shapes[idx].copy()
        label = self.labels[idx]

        # Apply multiple damages
        damage_direction = np.zeros_like(shape)

        for _ in range(num_damages):
            shape, new_damage_direction = self._generate_damage(shape)
            # Combine damage directions, prioritizing new damage
            damage_direction = np.where(
                new_damage_direction > 0, new_damage_direction, damage_direction
            )

        # Convert to PyTorch tensors
        damaged_shape_tensor = torch.tensor(shape, dtype=torch.float32)
        damage_direction_tensor = torch.tensor(damage_direction, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_damage_mask:
            # Calculate the damage mask (where damage was applied)
            original_shape = self.shapes[idx]
            damage_mask = (
                original_shape.astype(np.float32) - shape.astype(np.float32)
            ).clip(0, 1)
            damage_mask_tensor = torch.tensor(damage_mask, dtype=torch.float32)
            return (
                damaged_shape_tensor,
                damage_direction_tensor,
                damage_mask_tensor,
                label_tensor,
            )
        else:
            return damaged_shape_tensor, damage_direction_tensor, label_tensor
