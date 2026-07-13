#!/usr/bin/env python3
"""Small local damage-direction training run for the Sakana AI voxel."""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_sakana_voxel import DEFAULT_OUTPUT_DIR, save_sakana_assets
from regen.dataset import DynamicDamageDataset
from regen.model import CellRecoveryModel
from regen.trainer import NCA3DTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tiny damage-direction NCA on the Sakana AI voxel."
    )
    parser.add_argument("--mode", choices=["local", "modal"], default="local")
    parser.add_argument(
        "--voxel-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "sakana_ai_voxel.npy",
        help="Path to the generated Sakana voxel .npy file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "outputs",
        help="Directory for checkpoints and prediction preview images.",
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument(
        "--min-steps",
        type=int,
        default=96,
        help="Minimum NCA rollout steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=128,
        help="Maximum NCA rollout steps.",
    )
    parser.add_argument(
        "--damage-radius-min",
        type=int,
        default=3,
        help="Minimum sphere/cube damage radius sampled during training.",
    )
    parser.add_argument(
        "--damage-radius-max",
        type=int,
        default=6,
        help="Maximum sphere/cube damage radius sampled during training.",
    )
    parser.add_argument(
        "--damage-class-weight",
        type=float,
        default=5.0,
        help="Cross-entropy class weight for nonzero damage-direction labels.",
    )
    parser.add_argument(
        "--num-damage-sites-min",
        type=int,
        default=1,
        help="Minimum number of damage sites per sample.",
    )
    parser.add_argument(
        "--num-damage-sites-max",
        type=int,
        default=3,
        help="Maximum number of damage sites per sample.",
    )
    parser.add_argument(
        "--min-damage-target-cells",
        type=int,
        default=32,
        help="Retry damage generation until this many nonzero target cells are present.",
    )
    parser.add_argument(
        "--max-damage-attempts",
        type=int,
        default=10,
        help="Maximum attempts to satisfy --min-damage-target-cells.",
    )
    parser.add_argument(
        "--damage-loss-type",
        choices=["cross_entropy", "focal"],
        default="focal",
        help="Loss type for damage direction training.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal-loss gamma when --damage-loss-type=focal.",
    )
    parser.add_argument(
        "--iterations-per-epoch",
        type=int,
        default=None,
        help="Trainer iterations per epoch. Defaults to one pass over num-samples.",
    )
    parser.add_argument("--hidden-channels", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--buffer-sampling-prob", type=float, default=0.5)
    parser.add_argument(
        "--overfit-fixed-sample",
        action="store_true",
        help=(
            "Debug mode: train on one fixed damage sample with replay disabled "
            "to check whether the model/loss can overfit."
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-frequency", type=int, default=50)
    parser.add_argument("--visualization-frequency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--preview-damage",
        action="store_true",
        help="Save damage preview images and exit without training.",
    )
    parser.add_argument(
        "--damage-preview-count",
        type=int,
        default=6,
        help="Number of damage samples to render with --preview-damage.",
    )
    parser.add_argument(
        "--damage-preview-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for damage preview PNGs.",
    )
    parser.add_argument("--modal-app-name", default="sakana-damage-trainer")
    parser.add_argument("--modal-gpu", default="A10G")
    parser.add_argument("--modal-timeout", type=int, default=60 * 60 * 4)
    parser.add_argument("--detach", dest="detach", action="store_true", default=True)
    parser.add_argument("--no-detach", dest="detach", action="store_false")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional Hugging Face model repo for uploading trained Sakana NCA weights.",
    )
    parser.add_argument(
        "--hf-filename",
        default="pytorch_model.pt",
        help="Filename for the uploaded Hugging Face model weights.",
    )
    parser.add_argument(
        "--hf-config-filename",
        default="config.json",
        help="Filename for the uploaded Hugging Face model config.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Optional Weights & Biases project for Sakana training metrics.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional Weights & Biases run name for Sakana training.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, for example 'cpu', 'cuda', or 'mps'.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_or_generate_voxel(voxel_path):
    if not voxel_path.exists():
        save_sakana_assets(voxel_path.parent)
    voxel = np.load(voxel_path).astype(np.float32)
    if voxel.ndim != 3:
        raise ValueError(f"Expected a 3D voxel grid, got shape {voxel.shape}")
    return voxel


def make_dataset(
    voxel,
    num_samples,
    seed,
    damage_radius_range=(3, 6),
    num_damage_sites_range=(1, 3),
    min_damage_target_cells=32,
    max_damage_attempts=10,
    fixed_damage=False,
):
    shapes = np.repeat(voxel[None, ...], num_samples, axis=0)
    labels = [0] * num_samples
    return DynamicDamageDataset(
        shapes=shapes,
        labels=labels,
        damage_radius_range=damage_radius_range,
        damage_types=["sphere", "cube"],
        num_damage_sites_range=num_damage_sites_range,
        min_damage_target_cells=min_damage_target_cells,
        max_damage_attempts=max_damage_attempts,
        fixed_damage=fixed_damage,
        augment_rotations=False,
        return_damage_mask=True,
        seed=seed,
    )


def render_damage_view(original, damaged, damage_direction, view="front", size=(8, 3)):
    visible = original > 0
    removed = visible & (damaged == 0)
    live = visible & (damaged > 0)
    boundary = live & (damage_direction > 0)

    facecolors = np.zeros(visible.shape + (4,), dtype=np.float32)
    facecolors[live] = [0.0, 0.05, 0.85, 0.68]
    facecolors[boundary] = [1.0, 0.72, 0.0, 1.0]
    facecolors[removed] = [1.0, 0.05, 0.0, 1.0]

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(visible, facecolors=facecolors, edgecolor="k", linewidth=0.06)
    ax.set_xlim(0, original.shape[0])
    ax.set_ylim(0, original.shape[1])
    ax.set_zlim(0, original.shape[2])
    ax.set_box_aspect(original.shape)
    if view == "front":
        ax.view_init(elev=8, azim=-88)
    else:
        ax.view_init(elev=24, azim=-55)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("white")
    plt.tight_layout(pad=0)

    from io import BytesIO

    buf = BytesIO()
    plt.savefig(
        buf,
        format="png",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_damage_projection(original, damaged, damage_direction, scale=8):
    live_projection = (damaged > 0).max(axis=1)
    removed_projection = ((original > 0) & (damaged == 0)).max(axis=1)
    target_projection = (damage_direction > 0).max(axis=1)

    width = original.shape[0] * scale
    height = original.shape[2] * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for x in range(original.shape[0]):
        for z in range(original.shape[2]):
            color = None
            if live_projection[x, z]:
                color = (0, 18, 210)
            if target_projection[x, z]:
                color = (255, 190, 0)
            if removed_projection[x, z]:
                color = (240, 20, 20)

            if color is None:
                continue

            y0 = (original.shape[2] - z - 1) * scale
            draw.rectangle(
                [x * scale, y0, (x + 1) * scale - 1, y0 + scale - 1],
                fill=color,
            )

    return image


def add_label(image, text):
    label_height = 34
    labeled = Image.new("RGB", (image.width, image.height + label_height), "white")
    labeled.paste(image, (0, label_height))
    draw = ImageDraw.Draw(labeled)
    draw.text((8, 10), text, fill="black")
    return labeled


def concat_horiz(images):
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    combined = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        combined.paste(image, (x, 0))
        x += image.width
    return combined


def concat_vert(images):
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    combined = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        combined.paste(image, (0, y))
        y += image.height
    return combined


def save_damage_previews(
    voxel,
    output_dir,
    count,
    seed,
    damage_radius_range=(3, 6),
    num_damage_sites_range=(1, 3),
    min_damage_target_cells=32,
    max_damage_attempts=10,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(
        voxel,
        max(1, count),
        seed,
        damage_radius_range=damage_radius_range,
        num_damage_sites_range=num_damage_sites_range,
        min_damage_target_cells=min_damage_target_cells,
        max_damage_attempts=max_damage_attempts,
    )
    rows = []
    for idx in range(count):
        damaged_tensor, damage_direction_tensor, _, original_tensor = dataset[idx]
        original = original_tensor.numpy().astype(np.uint8)
        damaged = damaged_tensor.numpy().astype(np.uint8)
        damage_direction = damage_direction_tensor.numpy().astype(np.uint8)
        removed_count = int(((original == 1) & (damaged == 0)).sum())
        boundary_count = int((damage_direction > 0).sum())

        projection = render_damage_projection(original, damaged, damage_direction)
        angle = render_damage_view(original, damaged, damage_direction, view="angle")
        row = concat_horiz(
            [
                add_label(
                    projection,
                    f"sample {idx:02d} projection | red removed={removed_count}, yellow target={boundary_count}",
                ),
                add_label(angle, f"sample {idx:02d} angled 3D"),
            ]
        )
        sample_path = output_dir / f"sakana_damage_sample_{idx:02d}.png"
        row.save(sample_path)
        rows.append(row)
        print(f"Saved damage preview: {sample_path}")

    montage_path = output_dir / "sakana_damage_previews.png"
    concat_vert(rows).save(montage_path)
    print(f"Saved damage preview montage: {montage_path}")
    return montage_path


def args_to_config(args):
    config = vars(args).copy()
    config["voxel_path"] = str(args.voxel_path)
    config["output_dir"] = str(args.output_dir)
    return config


def train_sakana_damage(voxel, config):
    config = config.copy()
    if config.get("overfit_fixed_sample"):
        config["num_samples"] = 1
        config["batch_size"] = 1
        config["buffer_size"] = 0
        config["buffer_sampling_prob"] = 0.0
        if config.get("iterations_per_epoch") is None:
            config["iterations_per_epoch"] = 1

    set_seed(config["seed"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.get("device"):
        device = torch.device(config["device"])
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    damage_radius_range = (
        config["damage_radius_min"],
        config["damage_radius_max"],
    )
    dataset = make_dataset(
        voxel,
        config["num_samples"],
        config["seed"],
        damage_radius_range=damage_radius_range,
        num_damage_sites_range=(
            config["num_damage_sites_min"],
            config["num_damage_sites_max"],
        ),
        min_damage_target_cells=config["min_damage_target_cells"],
        max_damage_attempts=config["max_damage_attempts"],
        fixed_damage=config.get("overfit_fixed_sample", False),
    )
    iterations_per_epoch = config.get("iterations_per_epoch")
    if iterations_per_epoch is None:
        iterations_per_epoch = math.ceil(config["num_samples"] / config["batch_size"])

    model = CellRecoveryModel(
        use_class_embeddings=False,
        num_hidden_channels=config["hidden_channels"],
        num_classes=1,
        class_channels=0,
        num_damage_directions=7,
        use_tanh=False,
    )

    trainer = NCA3DTrainer(
        model=model,
        dataset=dataset,
        batch_size=config["batch_size"],
        lr=config["lr"],
        iterations_per_epoch=iterations_per_epoch,
        min_steps_per_sample=config["min_steps"],
        max_steps_per_sample=config["max_steps"],
        buffer_size=config["buffer_size"],
        buffer_sampling_prob=config["buffer_sampling_prob"],
        grad_clip=config["grad_clip"],
        damage_class_weight=config["damage_class_weight"],
        damage_loss_type=config["damage_loss_type"],
        focal_gamma=config["focal_gamma"],
        device=device,
        wandb_project=config.get("wandb_project"),
        wandb_run_name=config.get("wandb_run_name"),
        save_dir=str(output_dir),
        repo_id=None,
    )

    print(
        "Training Sakana damage NCA with NCA3DTrainer "
        f"on {device} with {int(voxel.sum())} live voxels"
    )
    if config.get("overfit_fixed_sample"):
        print(
            "Overfit debug mode enabled: fixed_damage=True, num_samples=1, "
            "batch_size=1, replay disabled.",
            flush=True,
        )
    trainer.train(
        epochs=config["epochs"],
        save_frequency=config["save_frequency"],
        visualization_frequency=config["visualization_frequency"],
    )

    pretrained_paths = trainer.model.save_pretrained(
        output_dir,
        filename=config.get("hf_filename", "pytorch_model.pt"),
        config_filename=config.get("hf_config_filename", "config.json"),
    )
    preview_path = output_dir / "sakana_damage_target_vs_prediction.png"
    trainer.visualize_results(config["epochs"] - 1).save(preview_path)
    print(f"Saved Sakana pretrained model: {pretrained_paths}", flush=True)
    print(f"Saved Sakana prediction preview: {preview_path}", flush=True)

    uploaded_files = None
    if config.get("repo_id"):
        print(f"Uploading Sakana model to Hugging Face: {config['repo_id']}", flush=True)
        uploaded_files = trainer.model.save_pretrained(
            config["repo_id"],
            push_to_hub=True,
            token=os.environ.get("HF_TOKEN"),
            filename=config.get("hf_filename", "pytorch_model.pt"),
            config_filename=config.get("hf_config_filename", "config.json"),
            commit_message="Save Sakana damage NCA model",
        )

    result = {
        "model": pretrained_paths,
        "preview_path": str(preview_path),
    }
    if uploaded_files:
        result["huggingface"] = uploaded_files

    return result


def run_modal(args):
    try:
        import modal
    except ImportError as exc:
        raise ImportError(
            "Modal Sakana training requires the `modal` package. "
            "Install it or run with `--mode local`."
        ) from exc

    if not args.repo_id:
        raise ValueError(
            "Modal Sakana training requires --repo-id for Hugging Face upload."
        )

    voxel = load_or_generate_voxel(args.voxel_path)
    config = args_to_config(args)
    config["output_dir"] = "/tmp/sakana_outputs"
    config["device"] = None

    app = modal.App(args.modal_app_name)
    env_variables = {
        key: os.environ.get(key)
        for key in ["HF_TOKEN", "WANDB_API_KEY"]
        if os.environ.get(key) is not None
    }
    secrets = [modal.Secret.from_dict(env_variables)] if env_variables else []
    image = (
        modal.Image.debian_slim()
        .pip_install(
            [
                "numpy",
                "torch",
                "matplotlib",
                "Pillow",
                "tqdm",
                "huggingface_hub",
                "wandb",
            ]
        )
        .add_local_python_source("regen")
    )

    @app.function(
        image=image,
        gpu=args.modal_gpu,
        timeout=args.modal_timeout,
        secrets=secrets,
        serialized=True,
    )
    def train_remote(remote_voxel, remote_config):
        return train_sakana_damage(remote_voxel, remote_config)

    with app.run(detach=args.detach):
        if args.detach:
            call = train_remote.spawn(voxel, config)
            call_id = getattr(call, "object_id", None) or getattr(
                call, "function_call_id", None
            )
            print(f"Spawned Modal Sakana training call: {call_id or call}")
            print(f"Trained weights will upload to Hugging Face: {args.repo_id}")
        else:
            result = train_remote.remote(voxel, config)
            print(result)


def main():
    args = parse_args()
    if args.preview_damage:
        voxel = load_or_generate_voxel(args.voxel_path)
        save_damage_previews(
            voxel,
            args.damage_preview_dir,
            args.damage_preview_count,
            args.seed,
            damage_radius_range=(args.damage_radius_min, args.damage_radius_max),
            num_damage_sites_range=(
                args.num_damage_sites_min,
                args.num_damage_sites_max,
            ),
            min_damage_target_cells=args.min_damage_target_cells,
            max_damage_attempts=args.max_damage_attempts,
        )
        return

    if args.mode == "modal":
        run_modal(args)
        return

    voxel = load_or_generate_voxel(args.voxel_path)
    train_sakana_damage(voxel, args_to_config(args))


if __name__ == "__main__":
    main()
