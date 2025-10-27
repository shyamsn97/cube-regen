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

    # Define color map for damage directions
    color_map = {
        0: "blue",  # No damage direction
        1: "gray",  # Damage in -x direction
        2: "orange",  # Damage in +x direction
        3: "purple",  # Damage in -y direction
        4: "pink",  # Damage in +y direction
        5: "yellow",  # Damage in -z direction
        6: "black",  # Damage in +z direction
    }

    # Create a 3D array of colors
    colors_array = np.empty(live_mask.shape, dtype=object)

    # Fill the colors array based on live cells and damage direction
    for x in range(live_mask.shape[0]):
        for y in range(live_mask.shape[1]):
            for z in range(live_mask.shape[2]):
                if live_mask[x, y, z] == 1:  # Cell is alive
                    direction = int(damage_direction[x, y, z])
                    colors_array[x, y, z] = color_map[direction]
                else:
                    colors_array[
                        x, y, z
                    ] = None  # Dead or damaged cells are transparent

    # Plot the voxels
    ax.voxels(live_mask, facecolors=colors_array, edgecolor="k", alpha=0.7)

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

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    # Instead of saving the figure, directly convert it to a PIL image
    buf = BytesIO()
    plt.savefig(buf, format="png", transparent=remove_background)
    plt.close(fig)
    buf.seek(0)
    pil_image = Image.open(buf)
    # pil_image.save(name)
    return pil_image


def save_weights(model, it, repo_id="shyamsn97/cube", binary=False):
    import numpy as np

    # Extract weights from the perceive layer
    perceive_weights = model.perceive[0].weight.data.cpu().numpy()

    # Extract biases (or create zero bias if not present)
    if model.perceive[0].bias is not None:
        perceive_bias = model.perceive[0].bias.data.cpu().numpy()
    else:
        perceive_bias = np.zeros(model.perception_channels)

    # Extract weights from dmodel layers
    dmodel_layer1 = model.dmodel[1]  # First conv layer after ReLU
    dmodel_kernel_1 = dmodel_layer1.weight.data.cpu().numpy()
    dmodel_bias_1 = dmodel_layer1.bias.data.cpu().numpy()

    dmodel_layer2 = model.dmodel[3]  # Last conv layer
    dmodel_kernel_2 = dmodel_layer2.weight.data.cpu().numpy()
    if dmodel_layer2.bias is not None:
        dmodel_bias_2 = dmodel_layer2.bias.data.cpu().numpy()
    else:
        dmodel_bias_2 = np.zeros(model.channel_n - 1)

    # Extract class embeddings if available
    if model.use_class_embeddings:
        class_embeddings = model.class_embeddings.weight.data.cpu().numpy()

    # Extract directional kernels
    # Reshape perceive_weights to match the TensorFlow extraction
    perceive_kernel = perceive_weights.reshape(
        model.perception_channels, model.channel_n, 3, 3, 3
    )

    # Get kernels for each direction
    percieve_self = perceive_kernel[:, :, 1, 1, 1]
    percieve_top = perceive_kernel[:, :, 0, 1, 1]
    percieve_bottom = perceive_kernel[:, :, 2, 1, 1]
    percieve_right = perceive_kernel[:, :, 1, 2, 1]
    percieve_left = perceive_kernel[:, :, 1, 0, 1]
    percieve_front = perceive_kernel[:, :, 1, 1, 0]
    percieve_back = perceive_kernel[:, :, 1, 1, 2]

    # Transpose the arrays to match TensorFlow's format
    percieve_self = percieve_self.T
    percieve_top = percieve_top.T
    percieve_bottom = percieve_bottom.T
    percieve_right = percieve_right.T
    percieve_left = percieve_left.T
    percieve_front = percieve_front.T
    percieve_back = percieve_back.T

    # Calculate dimensions
    in_channels = model.channel_n
    out_channels = model.perception_channels
    dmodel_out = model.channel_n - 1

    # Write to text file
    text_file = open(f"neural_network_output_damage_detection_{it}.txt", "w+")

    text_file.write(
        "#ifndef NEURAL_NET_H \n"
        "#define NEURAL_NET_H \n"
        "#include <Arduino.h> \n"
        f"PROGMEM const float percieve_kernel_back[{in_channels}][{out_channels}]  = {{"
    )

    for p in range(in_channels):
        text_file.write("{")
        for s in range(out_channels):
            temp = str(percieve_back[p][s])
            text_file.write(temp + ",")
        text_file.write("},\n")
    text_file.write("};\n")

    text_file.write(
        f"PROGMEM const float percieve_kernel_front[{in_channels}][{out_channels}]  = {{"
    )

    for p in range(in_channels):
        text_file.write("{")
        for s in range(out_channels):
            temp = str(percieve_front[p][s])
            text_file.write(temp + ",")
        text_file.write("},\n")
    text_file.write("};\n")

    text_file.write(
        f"PROGMEM const float percieve_kernel_north[{in_channels}][{out_channels}]  = {{"
    )

    for p in range(in_channels):
        text_file.write("{")
        for s in range(out_channels):
            temp = str(percieve_top[p][s])
            text_file.write(temp + ",")
        text_file.write("},\n")
    text_file.write("};\n")

    text_file.write(
        f"PROGMEM const float percieve_kernel_south[{in_channels}][{out_channels}]  = {{"
    )

    for p in range(in_channels):
        text_file.write("{")
        for s in range(out_channels):
            temp = str(percieve_bottom[p][s])
            text_file.write(temp + ",")
        text_file.write("},\n")
    text_file.write("};\n")

    text_file.write(
        f"PROGMEM const float percieve_kernel_east[{in_channels}][{out_channels}]  = {{"
    )

    for p in range(in_channels):
        text_file.write("{")
        for s in range(out_channels):
            temp = str(percieve_right[p][s])
            text_file.write(temp + ",")
        text_file.write("},\n")
    text_file.write("};\n")

    text_file.write(
        f"PROGMEM const float percieve_kernel_west[{in_channels}][{out_channels}]  = {{"
    )

    for p in range(in_channels):
        text_file.write("{")
        for s in range(out_channels):
            temp = str(percieve_left[p][s])
            text_file.write(temp + ",")
        text_file.write("},\n")
    text_file.write("};\n")

    text_file.write(
        f"PROGMEM const float percieve_kernel_self[{in_channels}][{out_channels}]  = {{"
    )

    for p in range(in_channels):
        text_file.write("{")
        for s in range(out_channels):
            temp = str(percieve_self[p][s])
            text_file.write(temp + ",")
        text_file.write("},\n")
    text_file.write("};\n")

    text_file.write(f"PROGMEM const float percieve_bias[{out_channels}]  = {{")
    for s in range(out_channels):
        temp = str(perceive_bias[s])
        text_file.write(temp + ",")
    text_file.write("};\n")

    text_file.write(f"PROGMEM const float dmodel_bias1[{out_channels}]  = {{")
    for s in range(out_channels):
        temp = str(dmodel_bias_1[s])
        text_file.write(temp + ",")
    text_file.write("};\n")

    text_file.write(f"PROGMEM const float dmodel_bias2[{dmodel_out}]  = {{")
    for s in range(dmodel_out):
        temp = str(dmodel_bias_2[s])
        text_file.write(temp + ",")
    text_file.write("};\n")

    # Reshape kernels to match expected format
    dmodel_kernel_1_flat = dmodel_kernel_1.reshape(out_channels, out_channels)
    dmodel_kernel_2_flat = dmodel_kernel_2.reshape(dmodel_out, out_channels).T

    text_file.write(
        f"PROGMEM const float dmodel_kernel_1[{out_channels}][{out_channels}]  = {{"
    )
    for p in range(out_channels):
        text_file.write("{")
        for s in range(out_channels):
            temp = str(dmodel_kernel_1_flat[p][s])
            text_file.write(temp + ",")
        text_file.write("},\n")
    text_file.write("};\n")

    text_file.write(
        f"PROGMEM const float dmodel_kernel_2[{out_channels}][{dmodel_out}]  = {{"
    )
    for p in range(out_channels):
        text_file.write("{")
        for s in range(dmodel_out):
            temp = str(dmodel_kernel_2_flat[p][s])
            text_file.write(temp + ",")
        text_file.write("},\n")
    text_file.write("};\n")

    if model.use_class_embeddings:
        text_file.write(
            f"PROGMEM const float class_embeddings[{model.num_classes}][{dmodel_out}]  = {{"
        )
        for p in range(model.num_classes):
            text_file.write("{")
            for s in range(dmodel_out):
                temp = str(class_embeddings[p, s])
                text_file.write(temp + ",")
            text_file.write("},\n")
        text_file.write("};\n")

    text_file.write("#endif")
    text_file.close()

    # Read and print the content (optional)
    text_file = open(f"neural_network_output_damage_detection_{it}.txt", "r")
    s = text_file.read()
    print(
        "=========================================================================== WEIGHTS ===========================================================================\n\n"
    )
    print(s)

    # Optional upload to Hugging Face
    try:
        import torch
        from huggingface_hub import HfApi

        api = HfApi()

        # Save and upload text weights
        weights_name = "weights_binary.txt" if binary else "weights.txt"
        api.upload_file(
            path_or_fileobj=f"neural_network_output_damage_detection_{it}.txt",
            path_in_repo=weights_name,
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Save and upload torch weights
        torch_weights_path = f"model_weights_{it}.pt"
        torch.save(model.state_dict(), torch_weights_path)
        api.upload_file(
            path_or_fileobj=torch_weights_path,
            path_in_repo=f"model_weights_{it}.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        print(f"Uploaded weights to Hugging Face repository 'shyamsn97/cube'")
    except ImportError:
        print(
            "Warning: huggingface_hub not installed. Skipping upload to Hugging Face."
        )
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
