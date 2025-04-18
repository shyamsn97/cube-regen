import torch
import torch.nn as nn
import re
import numpy as np


class NCA3DDamageDetection(nn.Module):
    def __init__(
        self,
        use_class_embeddings: bool = True,
        num_hidden_channels: int = 20,
        num_classes: int = 7,
        num_damage_directions: int = 7,
        alpha_living_threshold: float = 0.1,
        cell_fire_rate: float = 0.5,
        clip_range: float = 64.0,
        use_tanh: bool = True,
    ):
        super().__init__()
        self.use_class_embeddings = use_class_embeddings
        self.num_classes = num_classes
        self.num_damage_directions = num_damage_directions
        self.num_hidden_channels = num_hidden_channels
        self.alpha_living_threshold = alpha_living_threshold
        self.cell_fire_rate = cell_fire_rate
        self.clip_range = clip_range
        self.use_tanh = use_tanh

        self.channel_n = num_hidden_channels + 1 + num_damage_directions
        self.perception_channels = self.channel_n * 3

        if self.use_class_embeddings:
            self.class_embeddings = nn.Embedding(num_classes, self.channel_n - 1)

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
                self.channel_n,
                self.perception_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        )

        # Processing network
        self.dmodel = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(
                self.perception_channels, self.perception_channels, kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv3d(
                self.perception_channels, self.channel_n - 1, kernel_size=1, bias=False
            ),
        )

        # Initialize weights
        self._init_weights()
        self.reset_diag_kernel()

    def _init_weights(self):
        # Initialize the last layer to zeros
        last_layer = self.dmodel[-1]
        nn.init.zeros_(last_layer.weight)

    def reset_diag_kernel(self):
        # Apply the kernel mask to the perception layer's weights
        conv_layer = self.perceive[0]
        conv_layer.weight.requires_grad = False  # Make the Conv3D layer not trainable
        kernel_mask = self.kernel_mask.to(conv_layer.weight.device)
        kernel_mask = kernel_mask.repeat(
            conv_layer.out_channels, conv_layer.in_channels, 1, 1, 1
        )
        conv_layer.weight.data *= kernel_mask

    def forward(self, x, y=None):
        gray, state = torch.split(x, [1, self.channel_n - 1], dim=-1)

        if y is not None and self.use_class_embeddings:
            y_embedding = self.class_embeddings(y)
            y_embedding = y_embedding.view(
                y_embedding.shape[0], 1, 1, 1, y_embedding.shape[-1]
            )
            update = (
                self.dmodel(self.perceive(x.permute(0, 4, 1, 2, 3))).permute(
                    0, 2, 3, 4, 1
                )
                + y_embedding
            )
        else:
            update = self.dmodel(self.perceive(x.permute(0, 4, 1, 2, 3))).permute(
                0, 2, 3, 4, 1
            )

        # Create update mask based on cell fire rate
        update_mask = torch.rand_like(x[:, :, :, :, :1]) <= self.cell_fire_rate

        # Create living mask based on threshold
        living_mask = gray > self.alpha_living_threshold

        # Combine masks
        residual_mask = (update_mask & living_mask).float()

        # Apply update with masking
        if self.use_tanh:
            state = state + residual_mask * torch.tanh(update)
        else:
            state = state + residual_mask * update

        # Optional clipping
        # state = torch.clamp(state, -self.clip_range, self.clip_range)

        return torch.cat([gray, state], dim=-1)

    def classify(self, x):
        # Extract the classification predictions
        return x[:, :, :, :, -self.num_damage_directions :]

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


def load_weights_from_file(model, weights_file_path):
    """
    Load weights from a text file in the format saved by the save_weights function.
    
    Args:
        model: The NCA3DDamageDetection model to load weights into
        weights_file_path: Path to the weights file
    
    Returns:
        The model with loaded weights
    """
    # Read the weights file
    with open(weights_file_path, 'r') as f:
        content = f.read()
    
    # Extract dimensions from the file
    in_channels = model.channel_n
    out_channels = model.perception_channels
    dmodel_out = model.channel_n - 1
    
    # Helper function to extract array data
    def extract_array(pattern, content, shape):
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find pattern: {pattern}")
        
        # Extract the array content
        array_str = match.group(1)
        
        # Parse the array content
        if len(shape) == 1:
            # 1D array
            values = re.findall(r'(-?\d+\.?\d*e?[-+]?\d*)', array_str)
            array = np.array([float(v) for v in values])
        elif len(shape) == 2:
            # 2D array
            rows = re.findall(r'{([^}]*)},?', array_str)
            array = np.zeros(shape)
            for i, row in enumerate(rows):
                if i >= shape[0]:
                    break
                values = re.findall(r'(-?\d+\.?\d*e?[-+]?\d*)', row)
                for j, val in enumerate(values):
                    if j >= shape[1]:
                        break
                    array[i, j] = float(val)
        
        return array
    
    # Extract all the kernel weights
    percieve_back = extract_array(
        r'percieve_kernel_back\[\d+\]\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (in_channels, out_channels)
    ).T
    
    percieve_front = extract_array(
        r'percieve_kernel_front\[\d+\]\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (in_channels, out_channels)
    ).T
    
    percieve_top = extract_array(
        r'percieve_kernel_north\[\d+\]\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (in_channels, out_channels)
    ).T
    
    percieve_bottom = extract_array(
        r'percieve_kernel_south\[\d+\]\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (in_channels, out_channels)
    ).T
    
    percieve_right = extract_array(
        r'percieve_kernel_east\[\d+\]\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (in_channels, out_channels)
    ).T
    
    percieve_left = extract_array(
        r'percieve_kernel_west\[\d+\]\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (in_channels, out_channels)
    ).T
    
    percieve_self = extract_array(
        r'percieve_kernel_self\[\d+\]\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (in_channels, out_channels)
    ).T
    
    # Extract biases
    perceive_bias = extract_array(
        r'percieve_bias\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (out_channels,)
    )
    
    dmodel_bias_1 = extract_array(
        r'dmodel_bias1\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (out_channels,)
    )
    
    dmodel_bias_2 = extract_array(
        r'dmodel_bias2\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (dmodel_out,)
    )
    
    # Extract kernels
    dmodel_kernel_1 = extract_array(
        r'dmodel_kernel_1\[\d+\]\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (out_channels, out_channels)
    )
    
    dmodel_kernel_2 = extract_array(
        r'dmodel_kernel_2\[\d+\]\[\d+\]\s*=\s*{(.*?)};', 
        content, 
        (out_channels, dmodel_out)
    ).T
    
    # Extract class embeddings if available
    if model.use_class_embeddings:
        try:
            class_embeddings = extract_array(
                r'class_embeddings\[\d+\]\[\d+\]\s*=\s*{(.*?)};', 
                content, 
                (model.num_classes, dmodel_out)
            )
            model.class_embeddings.weight.data = torch.tensor(
                class_embeddings, 
                dtype=torch.float32, 
                device=model.class_embeddings.weight.device
            )
        except Exception as e:
            print(f"Warning: Could not load class embeddings: {e}")
    
    # Reconstruct the perceive kernel
    perceive_kernel = np.zeros((model.perception_channels, model.channel_n, 3, 3, 3))
    perceive_kernel[:, :, 1, 1, 1] = percieve_self
    perceive_kernel[:, :, 0, 1, 1] = percieve_top
    perceive_kernel[:, :, 2, 1, 1] = percieve_bottom
    perceive_kernel[:, :, 1, 2, 1] = percieve_right
    perceive_kernel[:, :, 1, 0, 1] = percieve_left
    perceive_kernel[:, :, 1, 1, 0] = percieve_front
    perceive_kernel[:, :, 1, 1, 2] = percieve_back
    
    # Reshape to match PyTorch's expected format
    perceive_weights = perceive_kernel.reshape(
        model.perception_channels, model.channel_n, 3, 3, 3
    )
    
    # Load weights into the model
    model.perceive[0].weight.data = torch.tensor(
        perceive_weights, 
        dtype=torch.float32, 
        device=model.perceive[0].weight.device
    )
    
    if model.perceive[0].bias is not None:
        model.perceive[0].bias.data = torch.tensor(
            perceive_bias, 
            dtype=torch.float32, 
            device=model.perceive[0].bias.device
        )
    
    # Load dmodel weights
    dmodel_layer1 = model.dmodel[1]  # First conv layer after ReLU
    dmodel_layer1.weight.data = torch.tensor(
        dmodel_kernel_1.reshape(dmodel_layer1.weight.shape), 
        dtype=torch.float32, 
        device=dmodel_layer1.weight.device
    )
    dmodel_layer1.bias.data = torch.tensor(
        dmodel_bias_1, 
        dtype=torch.float32, 
        device=dmodel_layer1.bias.device
    )
    
    dmodel_layer2 = model.dmodel[3]  # Last conv layer
    dmodel_layer2.weight.data = torch.tensor(
        dmodel_kernel_2.reshape(dmodel_layer2.weight.shape), 
        dtype=torch.float32, 
        device=dmodel_layer2.weight.device
    )
    
    if dmodel_layer2.bias is not None:
        dmodel_layer2.bias.data = torch.tensor(
            dmodel_bias_2, 
            dtype=torch.float32, 
            device=dmodel_layer2.bias.device
        )
    
    # Apply the kernel mask to ensure the correct structure
    model.reset_diag_kernel()
    
    return model

# Example usage
model = NCA3DDamageDetection()
load_weights_from_file(model, 'weights.txt')
