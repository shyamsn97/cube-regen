import re

import numpy as np
import torch
import torch.nn as nn


class NCA3D(nn.Module):
    def __init__(
        self,
        num_hidden_channels: int = 20,
        num_classes: int = 7,
        alpha_living_threshold: float = 0.1,
        cell_fire_rate: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_hidden_channels = num_hidden_channels
        self.alpha_living_threshold = alpha_living_threshold
        self.cell_fire_rate = cell_fire_rate

        self.channel_n = num_hidden_channels + num_classes + 1
        self.perception_channels = self.channel_n * 3

        # Define the 3D kernel mask
        self.kernel_mask = torch.tensor(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3, 3)

        # Perception network
        self.perceive = nn.Sequential(
            nn.Conv3d(
                self.channel_n, self.perception_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

        # Processing network
        self.dmodel = nn.Sequential(
            nn.Conv3d(
                self.perception_channels, self.perception_channels, kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv3d(
                self.perception_channels, self.channel_n - 1, kernel_size=1, bias=True
            ),
        )

        # Initialize the last layer to zeros
        self._init_weights()
        self.reset_diag_kernel()

    def _init_weights(self):
        # Initialize the last layer to zeros (equivalent to tf.zeros_initializer)
        last_layer = self.dmodel[-1]
        nn.init.zeros_(last_layer.weight)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)

    def reset_diag_kernel(self):
        # Apply the kernel mask to the perception layer's weights
        conv_layer = self.perceive[0]
        kernel_mask = self.kernel_mask.to(conv_layer.weight.device)
        kernel_mask = kernel_mask.repeat(
            conv_layer.out_channels, conv_layer.in_channels, 1, 1, 1
        )
        conv_layer.weight.data *= kernel_mask

    def forward(self, x):
        # Split input into gray and state channels
        gray, state = torch.split(x, [1, self.channel_n - 1], dim=-1)

        # Permute for Conv3d (batch, channels, depth, height, width)
        x_permuted = x.permute(0, 4, 1, 2, 3)

        # Process through perception and dmodel
        perceived = self.perceive(x_permuted)
        update = self.dmodel(perceived)

        # Permute back to (batch, depth, height, width, channels)
        update = update.permute(0, 2, 3, 4, 1)

        # Create update mask based on cell fire rate
        update_mask = torch.rand_like(x[:, :, :, :, :1]) <= self.cell_fire_rate

        # Create living mask based on threshold
        living_mask = gray > self.alpha_living_threshold

        # Combine masks
        residual_mask = (update_mask & living_mask).float()

        # Apply update with masking and tanh activation
        state = state + residual_mask * torch.tanh(update)

        return torch.cat([gray, state], dim=-1)

    def classify(self, x):
        # The last num_classes channels are the classification predictions
        return x[:, :, :, :, -self.num_classes :]

    def initialize(self, structure):
        shape = structure.shape
        state = torch.zeros(
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            self.channel_n - 1,
            device=structure.device,
        )
        structure = structure.view(shape[0], shape[1], shape[2], shape[3], 1)
        return torch.cat([structure, state], dim=-1)


def load_weights_from_classification_file(model, filepath):
    """
    Load weights from classification.txt file (generated from Keras model) into PyTorch NCA3D model.

    Args:
        model: The PyTorch NCA3D model to load weights into
        filepath: Path to the classification.txt file
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Dictionary to store parsed arrays
    weight_arrays = {}

    # Regular expression to match PROGMEM const float declarations with nested braces
    # This pattern handles arrays that start with {{ and end with }}
    array_pattern = (
        r"PROGMEM const float (\w+)\[([^\]]+)\]\s*=\s*\{\{([^}]+(?:\}[^}]*)*)\}\}"
    )

    matches = re.findall(array_pattern, content, re.DOTALL)

    # If the nested braces pattern doesn't work, try the simple pattern
    if not matches:
        print("Trying simpler pattern...")
        array_pattern = r"PROGMEM const float (\w+)\[([^\]]+)\]\s*=\s*\{([^;]+)\};"
        matches = re.findall(array_pattern, content, re.DOTALL)

    print(f"Found {len(matches)} arrays")

    for name, shape_str, data_str in matches:
        print(f"Processing array: {name}")

        # Parse the data - handle nested braces by removing them
        data_str = data_str.replace("{", "").replace("}", "")

        data_values = []
        for line in data_str.split("\n"):
            line = line.strip()
            if line and not line.startswith("//"):
                # Remove trailing comma and split by comma
                line = line.rstrip(",")
                values = [float(x.strip()) for x in line.split(",") if x.strip()]
                data_values.extend(values)

        # Parse shape dimensions
        shape_parts = shape_str.split("][")
        if len(shape_parts) == 1:
            # 1D array
            shape = (int(shape_parts[0]),)
        else:
            # 2D array
            shape = (int(shape_parts[0]), int(shape_parts[1]))

        # Create numpy array and reshape
        array = np.array(data_values).reshape(shape)
        weight_arrays[name] = array
        print(f"Loaded {name} with shape {array.shape}")

    # Print all available keys for debugging
    print("Available weight arrays:")
    for key in sorted(weight_arrays.keys()):
        print(f"  {key}: {weight_arrays[key].shape}")

    # Now map the weights to the PyTorch model
    device = next(model.parameters()).device

    try:
        # Load perception kernels (need to be combined and transposed for PyTorch Conv3d)
        # In Keras: filters are [depth, height, width, input_channels, output_channels]
        # In PyTorch: filters are [output_channels, input_channels, depth, height, width]

        perception_kernels = np.stack(
            [
                weight_arrays["percieve_kernel_self"],  # center
                weight_arrays["percieve_kernel_back"],  # back
                weight_arrays["percieve_kernel_front"],  # front
                weight_arrays["percieve_kernel_north"],  # north
                weight_arrays["percieve_kernel_south"],  # south
                weight_arrays["percieve_kernel_east"],  # east
                weight_arrays["percieve_kernel_west"],  # west
            ],
            axis=0,
        )  # Shape: [7, 28, 84]

        # The perception kernels need to be reshaped to [28, 84, 3, 3, 3]
        # where the 3x3x3 kernel has the pattern defined in the original model
        perception_weights = np.zeros((28, 84, 3, 3, 3))

        # Map the 7 directional kernels to the 3x3x3 positions
        kernel_positions = [
            (1, 1, 1),  # self (center)
            (0, 1, 1),  # back
            (2, 1, 1),  # front
            (1, 0, 1),  # north
            (1, 2, 1),  # south
            (1, 1, 0),  # east
            (1, 1, 2),  # west
        ]

        for i, (d, h, w) in enumerate(kernel_positions):
            perception_weights[:, :, d, h, w] = perception_kernels[
                i
            ].T  # Transpose for PyTorch

        model.perceive[0].weight.data = (
            torch.from_numpy(perception_weights).float().to(device)
        )
        model.perceive[0].bias.data = (
            torch.from_numpy(weight_arrays["percieve_bias"]).float().to(device)
        )

        # Load dense model weights
        # For PyTorch Linear layers: weight shape is [out_features, in_features]
        # Keras Dense layers: weight shape is [in_features, out_features]

        dmodel_weight1 = weight_arrays["dmodel_kernel_1"].T  # Transpose for PyTorch
        model.dmodel[0].weight.data = (
            torch.from_numpy(dmodel_weight1).float().to(device)
        )
        model.dmodel[0].bias.data = (
            torch.from_numpy(weight_arrays["dmodel_bias1"]).float().to(device)
        )

        dmodel_weight2 = weight_arrays["dmodel_kernel_2"].T  # Transpose for PyTorch
        model.dmodel[2].weight.data = (
            torch.from_numpy(dmodel_weight2).float().to(device)
        )
        model.dmodel[2].bias.data = (
            torch.from_numpy(weight_arrays["dmodel_bias2"]).float().to(device)
        )

        print("Successfully loaded all weights from classification.txt")
        print(f"Perception layer: {model.perceive[0].weight.shape}")
        print(f"Dense layer 1: {model.dmodel[0].weight.shape}")
        print(f"Dense layer 2: {model.dmodel[2].weight.shape}")

    except KeyError as e:
        print(f"Missing weight array: {e}")
        raise
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise
