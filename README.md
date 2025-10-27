# Smart Cellular Bricks for Decentralized Shape Classification and Damage Recovery

## Training


We train models separately for each class, in this one we specify tables at index 3 in the dataset.

```python
from regen.dataset import DynamicDamageDataset
from regen.model import NCA3DDamageDetection
from regen.trainer import NCA3DTrainer
import numpy as np

def train(shapes, labels, num_hidden_channels=20, model_repo_id="shyamsn97/table-cube-regen-damage-detection"):
    model_repo_id = f"shyamsn97/table-cube-regen-damage-detection-{num_hidden_channels}"
    repo_id = f"shyamsn97/table-cube-regen-damage-detection-txt-{num_hidden_channels}"
    model = NCA3DDamageDetection(use_tanh=False, clip_range=64.0, num_hidden_channels=num_hidden_channels)
    dataset = DynamicDamageDataset(shapes, labels, damage_radius_range=(1, 3), damage_types=["sphere", "cube", "random"], random_proportion_range=(0.1, 0.2), fixed_damage=False, augment_rotations=False, return_damage_mask=True, seed=None, filter_label=3)
    trainer = NCA3DTrainer(model, dataset, batch_size=8, lr=2e-5, iterations_per_epoch=100, steps_per_sample=96, buffer_size=1000, buffer_sampling_prob=0.5, repo_id=repo_id, model_repo_id=model_repo_id)
    trainer.train(epochs=500, save_frequency=10, visualization_frequency=10)

if __name__ == "__main__":
    labels = np.load('../data/ydata_7class.npy')
    shapes = np.load('../data/xdata_7class.npy')
    train(shapes, labels, num_hidden_channels=20, model_repo_id="shyamsn97/table-cube-regen-damage-detection")
```

## Inference

To load the model:

```python

from regen.model import load_weights_from_huggingface
import numpy as np
from regen.dataset import DynamicDamageDataset
import torch
from PIL import Image, ImageDraw, ImageFont
from regen.utils import plot_voxels

loaded_model, config = load_weights_from_huggingface(
        model=None,  # Will create model from config
        repo_id=model_repo_id,
        filename="pytorch_model.pt",
        load_config=True,
        config_filename="config.json"
    )
```

inferencing should looped

```python
class_label = torch.tensor([0])
state = loaded_model.initialize(initial_mask.unsqueeze(0))
with torch.no_grad():
    for i in range(96):
        state = loaded_model(state, class_label.unsqueeze(0))
        predictions = loaded_model.classify(state)
        predictions = torch.argmax(predictions, dim=-1).detach().cpu().numpy()[0]
```

Full example can be seen in [inference](./scripts/inference.py)