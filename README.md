# Smart Cellular Bricks for Decentralized Shape Classification and Damage Recovery

## Install
```bash
pip install -e .
```

## Pretrained Models

- Normal combined model: [shyamsn97/cube-regen-combined](https://huggingface.co/shyamsn97/cube-regen-combined)
- ShapeNet combined model: [shyamsn97/shapenet-cube-regen-combined-hdim-48](https://huggingface.co/shyamsn97/shapenet-cube-regen-combined-hdim-48)

## Todo

- In progress: clean pretrained model loading from Hugging Face
- In progress: recovery script
- In progress: live UI


## Training
Training is driven by YAML configs and a single entry point for both local and Modal runs.

```bash
python scripts/train.py --config configs/train_combined.yaml
```

Common training runs also have Make targets:

```bash
make train-damage
make train-damage-modal
make train-combined
make train-combined-modal
```

The default config trains the combined model, which predicts both:

- voxel-wise damage direction labels
- a shape-level class label

To train the same config on Modal:

```bash
python scripts/train.py --config configs/train_combined_modal.yaml
```

You can also override the configured mode:

```bash
python scripts/train.py --config configs/train_combined.yaml --mode modal
python scripts/train.py --config configs/train_combined_modal.yaml --mode local
```

### Configs

Available configs:

- `configs/train_combined.yaml`: local combined training from `data/xdata_7class.npy` and `data/ydata_7class.npy`
- `configs/train_combined_modal.yaml`: Modal combined training with the same dataset
- `configs/train_shapenet.yaml`: combined training from pre-voxelized ShapeNet folders
- `configs/train_damage.yaml`: local damage-only training
- `configs/train_damage_modal.yaml`: Modal damage-only training
- `configs/train_damage_shapenet.yaml`: local ShapeNet damage-only training
- `configs/train_damage_shapenet_modal.yaml`: Modal ShapeNet damage-only training

Important sections:

```yaml
run:
  mode: local # or modal
dataset:
  source: npy # or shapenet
model:
  type: combined # or damage
training:
  epochs: 500
output:
  checkpoint_dir: combined_nca_models
```

For the older damage-only model, set:

```yaml
model:
  type: damage
dataset:
  filter_label: 3 # optional, for one-class damage-only training
```

Damage configs use chunk-style `sphere`/`cube` damage, weighted cross entropy for sparse nonzero boundary labels, and lower replay sampling so fresh damage examples drive early learning.

### ShapeNet

`configs/train_shapenet.yaml` expects pre-voxelized ShapeNet data arranged by category:

```text
data/shapenet_voxels/
  chair/**/*.npy
  table/**/*.npz
  03001627/**/*.binvox
```

Supported voxel formats are `.npy`, `.npz`, and `.binvox`. Edit these fields for your dataset:

```yaml
dataset:
  source: shapenet
  root: data/shapenet_voxels
  categories: [chair, table]
  max_shapes_per_class: 500
  target_size: 32
```

For Modal ShapeNet runs, the Make targets download only the sampled files, upload that subset to the Modal volume, and then start a detached Modal training job:

```bash
make train-combined-shapenet-modal
make train-damage-shapenet-modal
```

By default, the subset downloader caches the public ShapeNet voxel archive in `~/.cache/cube-regen/shapenet`, samples from it, writes the selected files to `data/shapenet_voxels`, uploads that directory to the Modal volume, and starts training. The config fields `sample_categories`, `max_shapes_per_class`, `category_seed`, and `shape_seed` control which files are selected.

## Inference
To load the model:

```python
from regen.model import load_weights_from_huggingface
import torch

loaded_model, config = load_weights_from_huggingface(
    model=None,  # Creates the right model from config.json
    repo_id=model_repo_id,
    filename="pytorch_model.pt",
    load_config=True,
    config_filename="config.json",
)
```

Run NCA updates for several steps, then read the outputs:

```python
state = loaded_model.initialize(initial_mask.unsqueeze(0))
with torch.no_grad():
    for _ in range(96):
        state = loaded_model(state)

    damage_logits = loaded_model.classify(state)
    damage_predictions = torch.argmax(damage_logits, dim=-1)

    if hasattr(loaded_model, "classify_shape"):
        class_logits = loaded_model.classify_shape(state)
        class_prediction = torch.argmax(class_logits, dim=-1)
```

For a damage-only model with class embeddings, pass the class label during rollout:

```python
class_label = torch.tensor([0])
state = loaded_model.initialize(initial_mask.unsqueeze(0))
with torch.no_grad():
    for _ in range(96):
        state = loaded_model(state, class_label)
```

An older damage-only example can be seen in [inference](./scripts/inference.py).

### Sakana Example

The Sakana example trains on a generated 3D `SAKANA AI` voxel asset and includes Modal training, prediction montages, and an iterative 3D recovery GIF:

```bash
make train-sakana-modal
make visualize-sakana-damage
make visualize-sakana-recovery-gif
```

The visualization targets load the Hugging Face model repo by default. The recovery GIF applies multiple deterministic damage spots, repeatedly predicts damage directions, and adds repaired voxels back into the shape. See [`examples/sakana/README.md`](./examples/sakana/README.md) for the debug overfit command and more details.
