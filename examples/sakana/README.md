# Sakana AI Voxel Damage Example

This example builds an upright 3D voxel text asset that literally spells `SAKANA AI` with a connected underline, saves the voxel and a preview image, then trains a lightweight damage-direction NCA on dynamically generated damage.

Generate the voxel asset:

```bash
python examples/sakana/generate_sakana_voxel.py
```

This writes:

- `examples/sakana/sakana_ai_voxel.npy`
- `examples/sakana/sakana_ai_voxel.png`

Run the local training example:

```bash
python examples/sakana/train_sakana_damage.py
```

The trainer uses the generated voxel as the only base shape, applies random sphere/cube damage each batch, and learns the 7-way damage direction labels through the repo's standard `NCA3DTrainer`. Sakana defaults to damage radius `3..6` and a `25x` class weight for nonzero damage-direction labels, because the target boundary cells are sparse. Watch `damaged_acc`, not just `alive_acc`. Outputs are written to `examples/sakana/outputs/`.

For a faster smoke test:

```bash
python examples/sakana/train_sakana_damage.py --epochs 1 --num-samples 4 --batch-size 2 --steps 4 --device cpu
```

Preview the sampled training damage without training:

```bash
python examples/sakana/train_sakana_damage.py --preview-damage
```

This writes `sakana_damage_previews.png` plus individual `sakana_damage_sample_*.png` files. In those images, blue is the surviving voxel text, red is removed damage, and yellow is the damage-direction target boundary.

Run inference on deterministic sphere/cube damage cases:

```bash
python examples/sakana/infer_sakana_damage.py
```

This loads `examples/sakana/outputs/sakana_damage_model.pt` if it exists. To load the Hugging Face upload instead:

```bash
python examples/sakana/infer_sakana_damage.py --repo-id shyamsn97/sakana-cube-regen-damage-detection
```

or:

```bash
make visualize-sakana-damage
```

The inference script intentionally supports only `sphere` and `cube` damage, not random damage. It writes a montage to `examples/sakana/outputs/inference/sakana_damage_inference.png`.

Run on Modal:

```bash
python examples/sakana/train_sakana_damage.py --mode modal --repo-id shyamsn97/sakana-cube-regen-damage-detection
```

or:

```bash
make train-sakana-modal
```

`make train-sakana-modal` uploads `pytorch_model.pt` and `config.json` to the Hugging Face model repo configured by `SAKANA_REPO`.
