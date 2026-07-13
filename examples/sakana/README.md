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
python examples/sakana/train_sakana_damage.py --epochs 1 --num-samples 4 --batch-size 2 --min-steps 4 --max-steps 6 --device cpu
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
python examples/sakana/infer_sakana_damage.py --checkpoint examples/sakana/outputs
```

or:

```bash
make visualize-sakana-damage
```

The inference script intentionally supports only `sphere` and `cube` damage, not random damage. It writes a montage to `examples/sakana/outputs/inference/sakana_damage_inference.png`.

Render an iterative 3D recovery GIF:

```bash
make visualize-sakana-damage-recovery-gif
make visualize-sakana-seed-recovery-gif
```

The damage recovery target starts from multiple deterministic damage spots. The seed recovery target starts from a small live-cell seed, defaulting to 64 starting cells and 96 recovery iterations so you can watch whether the model regrows the shape. The 3D GIF renders every fourth recovery step by default to keep generation quick; set `SAKANA_RECOVERY_FRAME_STRIDE=1` to render every step. Existing voxels are blue, predicted repair-boundary voxels are yellow, newly recovered voxels are green, and damaged voxels are empty holes. Recovery uses repeated prediction voting by default, so a repair target must be predicted in 6 of the last 12 recovery passes before it is added; tune this with `SAKANA_RECOVERY_SEED_CELLS`, `SAKANA_RECOVERY_FRAME_STRIDE`, `SAKANA_RECOVERY_CONFIDENCE_WINDOW`, and `SAKANA_RECOVERY_CONFIDENCE_REQUIRED`.

Run on Modal:

```bash
python examples/sakana/train_sakana_damage.py --mode modal --repo-id shyamsn97/sakana-cube-regen-damage-detection
```

or:

```bash
make train-sakana-modal
```

`make train-sakana-modal` uploads `pytorch_model.pt` and `config.json` to the Hugging Face model repo configured by `SAKANA_REPO`.
