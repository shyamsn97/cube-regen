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

The trainer uses the generated voxel as the only base shape, applies random sphere/cube damage each batch, and learns the 7-way damage direction labels through the repo's standard `NCA3DTrainer`. Watch `damaged_acc`, not just `alive_acc`, because most live voxels have the no-damage label. Outputs are written to `examples/sakana/outputs/`.

For a faster smoke test:

```bash
python examples/sakana/train_sakana_damage.py --epochs 1 --num-samples 4 --batch-size 2 --steps 4 --device cpu
```

Preview the sampled training damage without training:

```bash
python examples/sakana/train_sakana_damage.py --preview-damage
```

This writes `sakana_damage_previews.png` plus individual `sakana_damage_sample_*.png` files. In those images, blue is the surviving voxel text, red is removed damage, and yellow is the damage-direction target boundary.

Debug whether the model/loss can overfit one fixed damage sample:

```bash
python examples/sakana/train_sakana_damage.py \
  --overfit-fixed-sample \
  --epochs 1000 \
  --iterations-per-epoch 8 \
  --damage-radius-min 3 \
  --damage-radius-max 3 \
  --num-damage-sites-min 1 \
  --num-damage-sites-max 1 \
  --output-dir examples/sakana/outputs/overfit_fixed_sample
```

This forces `fixed_damage=True`, `num_samples=1`, `batch_size=1`, and disables replay.

Run inference on deterministic sphere/cube damage cases:

```bash
python examples/sakana/infer_sakana_damage.py
```

This loads the Hugging Face model repo by default. To use a local checkpoint instead, pass `--checkpoint`:

```bash
python examples/sakana/infer_sakana_damage.py --checkpoint examples/sakana/outputs/sakana_damage_model.pt
```

or:

```bash
make visualize-sakana-damage
```

The inference script intentionally supports only `sphere` and `cube` damage, not random damage. It writes a montage to `examples/sakana/outputs/inference/sakana_damage_inference.png`.

Render an iterative 3D recovery GIF:

```bash
make visualize-sakana-recovery-gif
```

The recovery GIF applies multiple deterministic damage spots, predicts damage directions, adds neighboring voxels in the predicted directions, and repeats. Existing voxels are blue, predicted repair-boundary voxels are yellow, newly recovered voxels are green, and damaged voxels are empty holes.

Run on Modal:

```bash
python examples/sakana/train_sakana_damage.py --mode modal --repo-id shyamsn97/sakana-cube-regen-damage-detection
```

or:

```bash
make train-sakana-modal
```

`make train-sakana-modal` uploads `pytorch_model.pt` and `config.json` to the Hugging Face model repo configured by `SAKANA_REPO`.
