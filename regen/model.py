import torch
import torch.nn as nn


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
