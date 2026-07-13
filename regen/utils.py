import random
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def apply_damage(
    live_mask,
    radius=2,
    damage_type="sphere",
    random_proportion=None,
    damage_center=None,
):
    """
    Apply damage to a 3D shape based on various damage modes.

    Args:
        live_mask: 3D numpy array where 1 represents live cells and 0 represents dead cells
        radius: Radius of the damage area (for sphere/cube mode)
        damage_type: 'sphere', 'cube', or 'random'
        random_proportion: If damage_type is 'random', proportion of live cells to damage (0.0-1.0)
        damage_center: Optional specific coordinate to use as damage center (x,y,z)

    Returns:
        new_live_mask: Updated live mask after damage
        damage_direction: 3D array indicating direction of nearby damage (0-6)
    """
    # Create copies to avoid modifying the original
    new_live_mask = np.copy(live_mask)
    shape = new_live_mask.shape

    # Find all live cells
    live_indices = np.argwhere(live_mask == 1)

    if len(live_indices) == 0:
        print("No live cells to damage!")
        return new_live_mask, np.zeros_like(new_live_mask)

    # Random proportion damage mode
    if damage_type == "random" and random_proportion is not None:
        if random_proportion <= 0 or random_proportion > 1:
            raise ValueError("Random proportion must be between 0 and 1")

        # Determine how many cells to damage
        num_live_cells = len(live_indices)
        num_cells_to_damage = int(num_live_cells * random_proportion)

        # Randomly select cells to damage
        cells_to_damage = random.sample(range(num_live_cells), num_cells_to_damage)

        # Apply damage to selected cells
        for idx in cells_to_damage:
            x, y, z = live_indices[idx]
            new_live_mask[x, y, z] = 0

    # Sphere or Cube damage mode
    else:
        # Pick a random live cell as the damage center if not provided
        if damage_center is None:
            damage_center = random.choice(live_indices)

        x0, y0, z0 = damage_center

        # Apply damage based on type
        for x in range(max(0, x0 - radius), min(shape[0], x0 + radius + 1)):
            for y in range(max(0, y0 - radius), min(shape[1], y0 + radius + 1)):
                for z in range(max(0, z0 - radius), min(shape[2], z0 + radius + 1)):
                    if damage_type == "sphere":
                        # Calculate Euclidean distance for spherical damage
                        distance = np.sqrt(
                            (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
                        )
                        if distance <= radius:
                            new_live_mask[x, y, z] = 0
                    elif damage_type == "cube":
                        new_live_mask[x, y, z] = 0

    # Create damage direction array
    damage_direction = np.zeros_like(new_live_mask)

    # Check each live cell to see if it's adjacent to damaged cells
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if new_live_mask[x, y, z] == 1:  # Only check live cells
                    # Check the 6 adjacent directions
                    if (
                        x > 0
                        and live_mask[x - 1, y, z] == 1
                        and new_live_mask[x - 1, y, z] == 0
                    ):
                        damage_direction[x, y, z] = 1  # Damage in -x direction
                    elif (
                        x < shape[0] - 1
                        and live_mask[x + 1, y, z] == 1
                        and new_live_mask[x + 1, y, z] == 0
                    ):
                        damage_direction[x, y, z] = 2  # Damage in +x direction
                    elif (
                        y > 0
                        and live_mask[x, y - 1, z] == 1
                        and new_live_mask[x, y - 1, z] == 0
                    ):
                        damage_direction[x, y, z] = 3  # Damage in -y direction
                    elif (
                        y < shape[1] - 1
                        and live_mask[x, y + 1, z] == 1
                        and new_live_mask[x, y + 1, z] == 0
                    ):
                        damage_direction[x, y, z] = 4  # Damage in +y direction
                    elif (
                        z > 0
                        and live_mask[x, y, z - 1] == 1
                        and new_live_mask[x, y, z - 1] == 0
                    ):
                        damage_direction[x, y, z] = 5  # Damage in -z direction
                    elif (
                        z < shape[2] - 1
                        and live_mask[x, y, z + 1] == 1
                        and new_live_mask[x, y, z + 1] == 0
                    ):
                        damage_direction[x, y, z] = 6  # Damage in +z direction

    return new_live_mask, damage_direction


def plot_voxels(
    live_mask, damage_direction, add_legend=False, remove_background=True, size=(10, 10)
):
    """
    Plot a 3D visualization of the live mask with damage directions.

    Args:
        live_mask: 3D numpy array where 1 represents live cells and 0 represents dead cells
        damage_direction: 3D array indicating direction of nearby damage (0-6)
        add_legend: Whether to add color legend
        remove_background: Whether to remove all background elements (grid, axes, etc.)
        size: Tuple specifying figure size (width, height)
    """
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")

    from matplotlib.colors import to_rgba

    color_names = [
        "blue",
        "gray",
        "orange",
        "purple",
        "pink",
        "yellow",
        "black",
        "red",
        "green",
        "skyblue",
        "lightgreen",
        "lightyellow",
        "lightpink",
    ]
    color_rgba = np.array([to_rgba(c) for c in color_names])

    dd = damage_direction.astype(int)
    colors_array = color_rgba[dd]
    colors_array[live_mask != 1] = 0

    ax.voxels(
        live_mask, facecolors=colors_array, edgecolor="k", linewidth=0.1, alpha=0.7
    )

    # Keep the camera scale fixed to the full voxel grid so damaged shapes
    # don't get auto-zoomed larger when fewer voxels are present.
    ax.set_xlim(0, live_mask.shape[0])
    ax.set_ylim(0, live_mask.shape[1])
    ax.set_zlim(0, live_mask.shape[2])

    # Add a legend for color meanings
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="blue", label="No Damage Direction"),
        plt.Rectangle((0, 0), 1, 1, color="gray", label="Damage in -X Direction"),
        plt.Rectangle((0, 0), 1, 1, color="orange", label="Damage in +X Direction"),
        plt.Rectangle((0, 0), 1, 1, color="purple", label="Damage in -Y Direction"),
        plt.Rectangle((0, 0), 1, 1, color="pink", label="Damage in +Y Direction"),
        plt.Rectangle((0, 0), 1, 1, color="yellow", label="Damage in -Z Direction"),
        plt.Rectangle((0, 0), 1, 1, color="black", label="Damage in +Z Direction"),
    ]
    if add_legend:
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.3, 1))

    # Remove grid lines, axis labels, and title
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    # Additional background removal options
    if remove_background:
        # Remove axes completely
        ax.set_axis_off()
        # Make figure background transparent
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        # Remove the pane backgrounds
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Make pane edges invisible
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)

    # Match the aspect ratio to the underlying voxel grid dimensions.
    ax.set_box_aspect(live_mask.shape)

    plt.tight_layout()
    # Instead of saving the figure, directly convert it to a PIL image
    buf = BytesIO()
    plt.savefig(buf, format="png", transparent=remove_background)
    plt.close(fig)
    buf.seek(0)
    pil_image = Image.open(buf)
    # pil_image.save(name)
    return pil_image
