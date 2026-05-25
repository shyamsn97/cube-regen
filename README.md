# Smart Cellular Bricks for Decentralized Shape Classification and Damage Recovery

## Install
```bash
pip install -e .
```


## Training
Training is driven by YAML configs and a single entry point for both local and Modal runs.

```bash
python scripts/train.py --config configs/train_local.yaml
```

The default config trains the combined model, which predicts both:

- voxel-wise damage direction labels
- a shape-level class label

To train the same config on Modal:

```bash
python scripts/train.py --config configs/train_modal.yaml
```

You can also override the configured mode:

```bash
python scripts/train.py --config configs/train_local.yaml --mode modal
python scripts/train.py --config configs/train_modal.yaml --mode local
```

### Configs

Available configs:

- `configs/train_local.yaml`: local combined training from `data/xdata_7class.npy` and `data/ydata_7class.npy`
- `configs/train_modal.yaml`: Modal combined training with the same dataset
- `configs/train_shapenet.yaml`: combined training from pre-voxelized ShapeNet folders

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

## OpenAI Gym Analogy
If you come from an OpenAI Gym or reinforcement learning background, you can think of each voxel shape as the environment state, and each NCA update step as the environment transition. The model observes the current damaged voxel grid, predicts local repair and classification signals, and then rolls the state forward for another step.

It is not a standard Gym setup because there is no explicit policy taking discrete actions and no scalar reward being optimized online. Instead, training teaches the cellular automaton to behave like a learned environment dynamics plus repair rule, where repeated updates gradually recover structure and improve classification.



```latex

\begin{table}[t]
\centering
\caption{\textbf{Recovery accuracy by voxel resolution}}
\label{table:resolution}

{\small
\begin{tabular}{lcccc}
\toprule
\textbf{Resolution / \#Hidden}
& \textbf{20} & \textbf{48} & \textbf{96} & \textbf{128} \\
\midrule
$15 \times 15 \times 15$ & 0.71 & 0.83 & 0.98 & 0.98 \\
$32 \times 32 \times 32$ & 0.40 & 0.54 & 0.87 & 0.96 \\
$64 \times 64 \times 64$ & 0.38 & 0.53 & 0.87 & 0.96 \\
\bottomrule
\end{tabular}
}
\end{table}

```