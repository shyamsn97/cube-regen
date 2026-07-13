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
- `configs/train_damage_shape_conditioned.yaml`: local shape-conditioned damage-only training
- `configs/train_damage_shape_conditioned_modal.yaml`: Modal shape-conditioned damage-only training
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
  save_dir: combined_nca_models
```

For damage-only training, set:

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
from regen.model import CellRecoveryModel

model = CellRecoveryModel.from_pretrained("shyamsn97/shapenet-cube-regen-combined-hdim-48")
prediction = model.predict(damaged_voxels, steps=96)
```

The prediction object includes voxel-wise damage labels and, for combined models, a class label:

```python
damage_predictions = prediction.damage_labels
class_prediction = prediction.class_label
```

To save or upload a trained model:

```python
model.save_pretrained("outputs/model")
model.save_pretrained("username/repo-name", push_to_hub=True)
```

### Sakana Example

The Sakana example trains on a generated 3D `SAKANA AI` voxel asset and includes Modal training, prediction montages, and an iterative 3D recovery GIF:

```bash
make train-sakana-modal
make visualize-sakana-damage
make visualize-sakana-damage-recovery-gif
make visualize-sakana-seed-recovery-gif
```

The visualization targets load the Hugging Face model repo by default. Recovery GIFs can start from deterministic damage spots or from a small seed of live cells. See [`examples/sakana/README.md`](./examples/sakana/README.md) for the debug overfit command and more details.

### Recovery GIFs

Render recovery on a ShapeNet sample:

```bash
make visualize-shapenet-recovery-gif
make visualize-shapenet-seed-recovery-gif
```

Render recovery on an NPY sample with the combined model:

```bash
make visualize-combined-recovery-gif
make visualize-combined-seed-recovery-gif
```

For ShapeNet, set `SHAPENET_RECOVERY_CATEGORY` or `SHAPENET_RECOVERY_SAMPLE_INDEX` to choose a specific sample. For NPY combined recovery, set `COMBINED_RECOVERY_CLASS_LABEL`, `COMBINED_RECOVERY_CLASS_NAME`, or `COMBINED_RECOVERY_SAMPLE_INDEX`. Seed recovery starts from `RECOVERY_SEED_CELLS` live cells.
