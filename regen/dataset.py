import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from regen.utils import (
    apply_damage,
    exposed_surface_indices,
    has_connected_live_cells,
    live_neighbors,
    plot_voxels,
)


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
        recovery_seed_proportion_range: Tuple[float, float] = (0.1, 0.9),
        center_seed_augment_damage_types: List[str] = ["sphere", "cube", "random"],
        center_seed_augment_sites_range: Tuple[int, int] = (0, 2),
        center_seed_augment_proportion_range: Tuple[float, float] = (0.02, 0.12),
        center_seed_augment_max_attempts: int = 8,
        num_damage_sites_range: Tuple[int, int] = (1, 1),
        min_damage_target_cells: int = 0,
        max_damage_attempts: int = 1,
        fixed_damage: bool = False,
        augment_rotations: bool = False,
        return_damage_mask: bool = False,
        seed: Optional[int] = None,
        filter_label: Optional[int] = None,
        filter_indices: Optional[List[int]] = None,
    ):
        """
        Initialize the dataset with 3D shapes and configuration for dynamic damage generation.

        Args:
            shapes: List of 3D numpy arrays representing shapes (1 for solid, 0 for empty)
            labels: List of class indices for each shape
            damage_radius_range: Range (min, max) for damage radius
            damage_types: List of damage types to sample from
                ("sphere", "cube", "random", "center_seed")
            random_proportion_range: Range (min, max) for random damage proportion
            recovery_seed_proportion_range: Range of original live cells to keep for
                center_seed recovery samples
            center_seed_augment_damage_types: Small damage types applied on top of
                center_seed samples
            center_seed_augment_sites_range: Range of small augment sites for
                center_seed samples
            center_seed_augment_proportion_range: Fraction of the current seed that a
                single augment site may remove
            center_seed_augment_max_attempts: Retry count for bounded augment damage
            num_damage_sites_range: Range (min, max) for damage sites per sample
            min_damage_target_cells: Retry until at least this many nonzero target cells
            max_damage_attempts: Maximum attempts when enforcing min_damage_target_cells
            fixed_damage: If True, damage is applied once per sample and reused
            augment_rotations: If True, apply random 90-degree rotations as augmentation
            return_damage_mask: If True, also return the mask showing where damage was applied
            seed: Optional random seed for reproducibility
            filter_label: Optional label class to filter
            filter_indices: Optional list of indices to filter
        """
        super().__init__()

        # Store the parameters
        self.shapes = shapes
        self.labels = labels
        self.damage_radius_range = damage_radius_range
        self.damage_types = damage_types
        self.random_proportion_range = random_proportion_range
        self.recovery_seed_proportion_range = recovery_seed_proportion_range
        self.center_seed_augment_damage_types = center_seed_augment_damage_types
        self.center_seed_augment_sites_range = center_seed_augment_sites_range
        self.center_seed_augment_proportion_range = center_seed_augment_proportion_range
        self.center_seed_augment_max_attempts = center_seed_augment_max_attempts
        self.num_damage_sites_range = num_damage_sites_range
        self.min_damage_target_cells = min_damage_target_cells
        self.max_damage_attempts = max_damage_attempts
        self.fixed_damage = fixed_damage
        self.augment_rotations = augment_rotations
        self.return_damage_mask = return_damage_mask
        self.filter_indices = filter_indices

        # Validate inputs
        if len(shapes) != len(labels):
            raise ValueError("Number of shapes and labels must match")
        non_empty_indices = [
            index
            for index, shape in enumerate(shapes)
            if np.count_nonzero(shape == 1) > 0
        ]
        if len(non_empty_indices) != len(shapes):
            skipped = len(shapes) - len(non_empty_indices)
            print(f"Skipping {skipped} empty shape sample(s) with no live cells.")
            shapes = [shapes[index] for index in non_empty_indices]
            labels = [labels[index] for index in non_empty_indices]

        if min(damage_radius_range) < 1:
            raise ValueError("Minimum damage radius must be at least 1")

        valid_damage_types = ["sphere", "cube", "random", "center_seed"]
        if not all(t in valid_damage_types for t in damage_types):
            raise ValueError(
                "Damage types must be 'sphere', 'cube', 'random', or 'center_seed'"
            )

        if not (0 <= min(random_proportion_range) <= max(random_proportion_range) <= 1):
            raise ValueError("Random proportion range must be between 0 and 1")

        if not (
            0
            < min(recovery_seed_proportion_range)
            <= max(recovery_seed_proportion_range)
            <= 1
        ):
            raise ValueError("Recovery seed proportion range must be in (0, 1]")
        if not all(
            t in valid_damage_types[:3] for t in center_seed_augment_damage_types
        ):
            raise ValueError(
                "Center seed augment damage types must be 'sphere', 'cube', or 'random'"
            )
        if not (
            0
            < min(center_seed_augment_proportion_range)
            <= max(center_seed_augment_proportion_range)
            <= 1
        ):
            raise ValueError("Center seed augment proportion range must be in (0, 1]")
        if min(center_seed_augment_sites_range) < 0:
            raise ValueError("Center seed augment sites range must be non-negative")
        if center_seed_augment_max_attempts < 1:
            raise ValueError("Center seed augment max attempts must be at least 1")

        if min(num_damage_sites_range) < 1:
            raise ValueError("Minimum number of damage sites must be at least 1")

        if min_damage_target_cells < 0:
            raise ValueError("min_damage_target_cells must be non-negative")

        if max_damage_attempts < 1:
            raise ValueError("max_damage_attempts must be at least 1")

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
        if filter_indices is not None:
            self.shapes = [self.shapes[i] for i in filter_indices]
            self.labels = [self.labels[i] for i in filter_indices]

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

    def _generate_damage(self, shape, damage_type=None):
        """Generate a random damage pattern for a shape."""
        best_live_mask = None
        best_damage_direction = None
        best_target_count = -1

        for _ in range(self.max_damage_attempts):
            new_live_mask, damage_direction = self._generate_damage_once(
                shape,
                damage_type=damage_type,
            )
            if np.count_nonzero(new_live_mask == 1) == 0:
                continue

            target_count = int(np.count_nonzero(damage_direction))

            if target_count > best_target_count:
                best_live_mask = new_live_mask
                best_damage_direction = damage_direction
                best_target_count = target_count

            if target_count >= self.min_damage_target_cells:
                return new_live_mask, damage_direction

        if best_live_mask is None:
            print(
                f"Failed to generate damage for shape... returning empty damage direction"
            )
            return shape.copy(), np.zeros_like(shape)

        return best_live_mask, best_damage_direction

    def _generate_damage_once(self, shape, damage_type=None):
        """Generate one damage sample, possibly with multiple damage sites."""
        current_live_mask = shape.copy()
        combined_damage_direction = np.zeros_like(shape)
        num_sites = random.randint(*self.num_damage_sites_range)

        for _ in range(num_sites):
            site_damage_type = damage_type or random.choice(self.damage_types)
            if site_damage_type == "center_seed":
                return self._generate_center_seed_damage(shape)

            random_proportion = None
            if site_damage_type == "random":
                random_proportion = random.uniform(*self.random_proportion_range)

            current_live_mask, site_damage_direction = apply_damage(
                current_live_mask.copy(),
                radius=random.randint(*self.damage_radius_range),
                damage_type=site_damage_type,
                random_proportion=random_proportion,
            )
            if np.count_nonzero(current_live_mask == 1) == 0:
                return current_live_mask, np.zeros_like(shape)

            combined_damage_direction = np.where(
                site_damage_direction > 0,
                site_damage_direction,
                combined_damage_direction,
            )

        combined_damage_direction = np.where(
            current_live_mask == 1,
            combined_damage_direction,
            0,
        )
        return current_live_mask, combined_damage_direction

    def _generate_center_seed_damage(self, shape):
        """Sample a center seed, lightly damage it, and label toward the full shape."""
        seed_mask = centered_subset_mask(
            shape,
            random.uniform(*self.recovery_seed_proportion_range),
        )
        current_mask = self._augment_center_seed_damage(seed_mask)
        return current_mask, frontier_direction_labels(current_mask, shape)

    def _augment_center_seed_damage(self, seed_mask):
        """Apply bounded small damage so tiny center seeds are not wiped out."""
        current_mask = seed_mask.copy()
        num_sites = random.randint(*self.center_seed_augment_sites_range)
        for _ in range(num_sites):
            live_count = int(np.count_nonzero(current_mask == 1))
            if live_count <= 1:
                break

            max_fraction = random.uniform(*self.center_seed_augment_proportion_range)
            max_removed = max(1, int(round(live_count * max_fraction)))
            current_mask = self._apply_bounded_center_seed_augment(
                current_mask,
                max_removed,
            )
        return current_mask

    def _apply_bounded_center_seed_augment(self, live_mask, max_removed):
        before_count = int(np.count_nonzero(live_mask == 1))
        for _ in range(self.center_seed_augment_max_attempts):
            damage_type = random.choice(self.center_seed_augment_damage_types)
            random_proportion = None
            if damage_type == "random":
                random_proportion = max_removed / max(before_count, 1)

            damaged_mask, _ = apply_damage(
                live_mask.copy(),
                radius=random.randint(*self.damage_radius_range),
                damage_type=damage_type,
                random_proportion=random_proportion,
            )
            after_count = int(np.count_nonzero(damaged_mask == 1))
            removed_count = before_count - after_count
            if (
                0 < removed_count <= max_removed
                and after_count > 0
                and has_connected_live_cells(damaged_mask)
            ):
                return damaged_mask

        return remove_random_live_cells(live_mask, max_removed)

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


def frontier_direction_labels(current_live_mask, original_shape):
    """Label live cells with one adjacent direction toward missing original cells."""
    directions = np.zeros_like(original_shape)
    shape = original_shape.shape
    for x, y, z in np.argwhere(current_live_mask == 1):
        if (
            x > 0
            and original_shape[x - 1, y, z] == 1
            and current_live_mask[x - 1, y, z] == 0
        ):
            directions[x, y, z] = 1
        elif (
            x < shape[0] - 1
            and original_shape[x + 1, y, z] == 1
            and current_live_mask[x + 1, y, z] == 0
        ):
            directions[x, y, z] = 2
        elif (
            y > 0
            and original_shape[x, y - 1, z] == 1
            and current_live_mask[x, y - 1, z] == 0
        ):
            directions[x, y, z] = 3
        elif (
            y < shape[1] - 1
            and original_shape[x, y + 1, z] == 1
            and current_live_mask[x, y + 1, z] == 0
        ):
            directions[x, y, z] = 4
        elif (
            z > 0
            and original_shape[x, y, z - 1] == 1
            and current_live_mask[x, y, z - 1] == 0
        ):
            directions[x, y, z] = 5
        elif (
            z < shape[2] - 1
            and original_shape[x, y, z + 1] == 1
            and current_live_mask[x, y, z + 1] == 0
        ):
            directions[x, y, z] = 6
    return directions


def centered_subset_mask(shape, keep_proportion):
    """Grow a connected live-cell subset from the shape centroid."""
    live_indices = np.argwhere(shape == 1)
    if len(live_indices) == 0:
        return shape.copy()

    keep_count = max(1, int(round(len(live_indices) * keep_proportion)))
    center = live_indices.mean(axis=0)
    start = tuple(
        live_indices[np.argmin(np.linalg.norm(live_indices - center, axis=1))]
    )

    seed_mask = np.zeros_like(shape)
    seed_mask[start] = 1
    frontier = {
        neighbor
        for neighbor in live_neighbors(start, shape.shape)
        if shape[neighbor] == 1
    }

    while int(seed_mask.sum()) < keep_count and frontier:
        next_cell = min(
            frontier, key=lambda coord: np.linalg.norm(np.array(coord) - center)
        )
        frontier.remove(next_cell)
        seed_mask[next_cell] = 1
        for neighbor in live_neighbors(next_cell, shape.shape):
            if shape[neighbor] == 1 and seed_mask[neighbor] == 0:
                frontier.add(neighbor)

    return seed_mask


def remove_random_live_cells(live_mask, max_removed):
    damaged_mask = live_mask.copy()
    remove_count = int(max_removed)
    while remove_count > 0:
        live_indices = np.argwhere(damaged_mask == 1)
        if len(live_indices) <= 1:
            break

        surface_indices = exposed_surface_indices(damaged_mask)
        candidate_order = list(range(len(surface_indices)))
        random.shuffle(candidate_order)
        accepted = False
        for index in candidate_order:
            candidate_mask = damaged_mask.copy()
            x, y, z = surface_indices[index]
            candidate_mask[x, y, z] = 0
            if has_connected_live_cells(candidate_mask):
                damaged_mask = candidate_mask
                remove_count -= 1
                accepted = True
                break
        if not accepted:
            break
    return damaged_mask
