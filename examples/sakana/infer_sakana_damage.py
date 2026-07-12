#!/usr/bin/env python3
"""Run Sakana damage-direction inference on deterministic sphere/cube damage."""

import argparse
import os
import sys
from io import BytesIO
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

from generate_sakana_voxel import DEFAULT_OUTPUT_DIR
from regen.model import NCA3DDamageDetection, load_weights_from_huggingface
from regen.utils import apply_damage, plot_voxels
from train_sakana_damage import (
    add_label,
    concat_horiz,
    concat_vert,
    load_or_generate_voxel,
    render_damage_projection,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render Sakana damage inference for non-random sphere/cube damage."
    )
    parser.add_argument(
        "--voxel-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "sakana_ai_voxel.npy",
        help="Path to the generated Sakana voxel .npy file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional local Sakana checkpoint. If omitted, inference loads the "
            "Hugging Face repo so Modal-trained weights are used by default."
        ),
    )
    parser.add_argument(
        "--repo-id",
        default="shyamsn97/sakana-cube-regen-damage-detection",
        help="Hugging Face model repo loaded by default unless --checkpoint is passed.",
    )
    parser.add_argument("--hf-filename", default="pytorch_model.pt")
    parser.add_argument("--hf-config-filename", default="config.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "outputs" / "inference",
    )
    parser.add_argument("--output", default="sakana_damage_inference.png")
    parser.add_argument(
        "--damage-types",
        nargs="+",
        choices=["sphere", "cube"],
        default=["sphere", "cube"],
        help="Damage types to render. Random damage is intentionally unsupported.",
    )
    parser.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=[3, 5],
        help="Damage radii to render for each damage type.",
    )
    parser.add_argument(
        "--center-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="X-axis fractions used to choose deterministic live-cell damage centers.",
    )
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--hidden-channels", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=5)
    parser.add_argument(
        "--recovery-gif",
        action="store_true",
        help="Also save an iterative predicted-damage recovery GIF.",
    )
    parser.add_argument("--recovery-output", default="sakana_damage_recovery.gif")
    parser.add_argument("--recovery-iterations", type=int, default=24)
    parser.add_argument("--recovery-frame-duration", type=int, default=250)
    parser.add_argument(
        "--recovery-damage-type",
        choices=["sphere", "cube"],
        default="sphere",
    )
    parser.add_argument("--recovery-radius", type=int, default=3)
    parser.add_argument(
        "--recovery-center-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="X-axis fractions for deterministic recovery damage spots.",
    )
    parser.add_argument(
        "--recovery-confidence-threshold",
        type=float,
        default=0.0,
        help="Only add predicted repairs with at least this softmax confidence.",
    )
    parser.add_argument(
        "--recovery-max-additions-per-step",
        type=int,
        default=None,
        help="Optional cap on repaired voxels per GIF frame.",
    )
    parser.add_argument(
        "--unconstrained-recovery",
        action="store_true",
        help="Allow predicted recovery outside the known original Sakana shape.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    voxel = load_or_generate_voxel(args.voxel_path)
    model = load_sakana_model(args, device)
    model.eval()

    rows = []
    case_idx = 0
    for damage_type in args.damage_types:
        for radius in args.radii:
            center_fraction = args.center_fractions[
                case_idx % len(args.center_fractions)
            ]
            rows.append(
                render_inference_case(
                    model=model,
                    voxel=voxel,
                    damage_type=damage_type,
                    radius=radius,
                    center_fraction=center_fraction,
                    steps=args.steps,
                    device=device,
                    image_size=args.image_size,
                )
            )
            case_idx += 1

    montage = concat_vert(rows)
    montage_path = args.output_dir / args.output
    montage.save(montage_path)
    print(f"Saved Sakana damage inference montage: {montage_path}")

    if args.recovery_gif:
        gif_path = args.output_dir / args.recovery_output
        save_recovery_gif(
            model=model,
            voxel=voxel,
            output_path=gif_path,
            damage_type=args.recovery_damage_type,
            radius=args.recovery_radius,
            center_fractions=args.recovery_center_fractions,
            steps=args.steps,
            iterations=args.recovery_iterations,
            frame_duration=args.recovery_frame_duration,
            confidence_threshold=args.recovery_confidence_threshold,
            max_additions_per_step=args.recovery_max_additions_per_step,
            constrain_to_original=not args.unconstrained_recovery,
            device=device,
        )
        print(f"Saved Sakana recovery GIF: {gif_path}")


def load_sakana_model(args, device):
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Missing local Sakana checkpoint: {args.checkpoint}")

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = checkpoint.get("model_config", {})
        model = NCA3DDamageDetection(
            use_class_embeddings=config.get("use_class_embeddings", False),
            num_hidden_channels=config.get(
                "num_hidden_channels",
                args.hidden_channels,
            ),
            num_classes=config.get("num_classes", 1),
            num_damage_directions=config.get("num_damage_directions", 7),
            alpha_living_threshold=config.get("alpha_living_threshold", 0.1),
            cell_fire_rate=config.get("cell_fire_rate", 0.5),
            clip_range=config.get("clip_range", 64.0),
            use_tanh=config.get("use_tanh", False),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded local Sakana checkpoint: {args.checkpoint}")
        return model.to(device)

    if args.repo_id:
        model, _ = load_weights_from_huggingface(
            model=None,
            repo_id=args.repo_id,
            token=os.environ.get("HF_TOKEN"),
            filename=args.hf_filename,
            load_config=True,
            config_filename=args.hf_config_filename,
        )
        print(f"Loaded Sakana model from Hugging Face: {args.repo_id}")
        return model.to(device)

    raise FileNotFoundError(
        "No model source configured. Pass --repo-id or --checkpoint."
    )


def render_inference_case(
    model,
    voxel,
    damage_type,
    radius,
    center_fraction,
    steps,
    device,
    image_size,
):
    center = choose_live_center(voxel, center_fraction)
    damaged, true_damage = apply_damage(
        voxel.copy(),
        radius=radius,
        damage_type=damage_type,
        damage_center=center,
    )
    predicted_damage = predict_damage(model, damaged, steps, device)
    full_acc, boundary_acc = damage_accuracies(predicted_damage, true_damage, damaged)
    removed_count = int(((voxel == 1) & (damaged == 0)).sum())
    boundary_count = int((true_damage > 0).sum())

    zeros = np.zeros_like(true_damage, dtype=np.uint8)
    panels = [
        add_label(
            render_damage_projection(voxel, damaged, true_damage),
            (
                f"{damage_type} r={radius} | removed={removed_count}, "
                f"target={boundary_count}"
            ),
        ),
        add_label(
            plot_voxels(
                damaged.astype(np.uint8),
                zeros,
                size=(image_size, image_size),
            ).convert("RGB"),
            "damaged input",
        ),
        add_label(
            plot_voxels(
                damaged.astype(np.uint8),
                true_damage.astype(np.uint8),
                size=(image_size, image_size),
            ).convert("RGB"),
            "true damage labels",
        ),
        add_label(
            plot_voxels(
                damaged.astype(np.uint8),
                predicted_damage.astype(np.uint8),
                size=(image_size, image_size),
            ).convert("RGB"),
            f"prediction | full={full_acc:.3f}, boundary={boundary_acc:.3f}",
        ),
    ]
    return concat_horiz(panels)


def choose_live_center(voxel, x_fraction):
    live_indices = np.argwhere(voxel == 1)
    if len(live_indices) == 0:
        raise ValueError("Cannot damage an empty voxel grid.")

    target = np.array(
        [
            x_fraction * (voxel.shape[0] - 1),
            (voxel.shape[1] - 1) / 2,
            (voxel.shape[2] - 1) / 2,
        ]
    )
    distances = np.linalg.norm(live_indices - target, axis=1)
    return tuple(live_indices[int(np.argmin(distances))])


def predict_damage(model, damaged, steps, device):
    damaged_tensor = torch.tensor(damaged, dtype=torch.float32, device=device)
    state = model.initialize(damaged_tensor.unsqueeze(0))
    with torch.no_grad():
        for _ in range(steps):
            state = model(state)
        logits = model.classify(state)
        prediction = torch.argmax(logits, dim=-1).squeeze(0)
    return prediction.detach().cpu().numpy().astype(np.uint8)


def predict_damage_with_confidence(model, damaged, steps, device):
    damaged_tensor = torch.tensor(damaged, dtype=torch.float32, device=device)
    state = model.initialize(damaged_tensor.unsqueeze(0))
    with torch.no_grad():
        for _ in range(steps):
            state = model(state)
        logits = model.classify(state)
        probabilities = torch.softmax(logits, dim=-1)
        confidence, prediction = torch.max(probabilities, dim=-1)
    return (
        prediction.squeeze(0).detach().cpu().numpy().astype(np.uint8),
        confidence.squeeze(0).detach().cpu().numpy(),
    )


def save_recovery_gif(
    model,
    voxel,
    output_path,
    damage_type,
    radius,
    center_fractions,
    steps,
    iterations,
    frame_duration,
    confidence_threshold,
    max_additions_per_step,
    constrain_to_original,
    device,
):
    current = apply_multi_spot_damage(
        voxel,
        damage_type=damage_type,
        radius=radius,
        center_fractions=center_fractions,
    )
    current = current.astype(np.uint8)
    original = voxel.astype(np.uint8)

    frames = []
    previous_added = None
    previous_added_count = 0
    for step_idx in range(iterations + 1):
        predicted_damage, confidence = predict_damage_with_confidence(
            model,
            current,
            steps,
            device,
        )
        missing_count = int(((original == 1) & (current == 0)).sum())
        predicted_count = int(((predicted_damage > 0) & (current == 1)).sum())
        frame = render_recovery_3d(
            original,
            current,
            predicted_damage,
            added_mask=previous_added,
        )
        frames.append(
            add_label(
                frame,
                (
                    f"step {step_idx:02d} | missing={missing_count} | "
                    f"predicted={predicted_count} | added_last={previous_added_count}"
                ),
            )
        )

        if step_idx == iterations or missing_count == 0:
            break

        current, previous_added, previous_added_count = recover_from_prediction(
            current,
            predicted_damage,
            confidence,
            original,
            confidence_threshold=confidence_threshold,
            max_additions=max_additions_per_step,
            constrain_to_original=constrain_to_original,
        )
        if previous_added_count == 0:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
    )


def apply_multi_spot_damage(voxel, damage_type, radius, center_fractions):
    current = voxel.copy()
    original = voxel.copy()
    for center_fraction in center_fractions:
        center = choose_live_center(original, center_fraction)
        current, _ = apply_damage(
            current,
            radius=radius,
            damage_type=damage_type,
            damage_center=center,
        )
    return current


def recover_from_prediction(
    current,
    predicted_damage,
    confidence,
    original,
    confidence_threshold=0.0,
    max_additions=None,
    constrain_to_original=True,
):
    candidates = []
    for x, y, z in np.argwhere((current == 1) & (predicted_damage > 0)):
        if confidence[x, y, z] < confidence_threshold:
            continue

        target = repair_target((x, y, z), int(predicted_damage[x, y, z]), current.shape)
        if target is None:
            continue

        tx, ty, tz = target
        if current[tx, ty, tz] == 1:
            continue
        if constrain_to_original and original[tx, ty, tz] != 1:
            continue

        candidates.append((float(confidence[x, y, z]), target))

    candidates.sort(reverse=True)
    if max_additions is not None:
        candidates = candidates[:max(0, max_additions)]

    next_current = current.copy()
    added_mask = np.zeros_like(current, dtype=np.uint8)
    for _, target in candidates:
        tx, ty, tz = target
        next_current[tx, ty, tz] = 1
        added_mask[tx, ty, tz] = 1

    return next_current, added_mask, int(added_mask.sum())


def repair_target(source, direction, shape):
    offsets = {
        1: (-1, 0, 0),
        2: (1, 0, 0),
        3: (0, -1, 0),
        4: (0, 1, 0),
        5: (0, 0, -1),
        6: (0, 0, 1),
    }
    if direction not in offsets:
        return None

    target = tuple(source[axis] + offsets[direction][axis] for axis in range(3))
    if any(target[axis] < 0 or target[axis] >= shape[axis] for axis in range(3)):
        return None
    return target


def render_recovery_3d(
    original,
    current,
    predicted_damage,
    added_mask=None,
    size=(12, 4),
):
    predicted_mask = (predicted_damage > 0) & (current > 0)
    if added_mask is None:
        added_mask = np.zeros_like(current, dtype=bool)
    else:
        added_mask = added_mask > 0

    visible = (current > 0) | added_mask
    facecolors = np.zeros(visible.shape + (4,), dtype=np.float32)
    facecolors[current > 0] = [0.0, 0.05, 0.85, 1.0]
    facecolors[predicted_mask] = [1.0, 0.72, 0.0, 1.0]
    facecolors[added_mask] = [0.0, 0.72, 0.25, 1.0]

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(visible, facecolors=facecolors, edgecolor="k", linewidth=0.04)
    ax.set_xlim(0, original.shape[0])
    ax.set_ylim(0, original.shape[1])
    ax.set_zlim(0, original.shape[2])
    ax.set_box_aspect(original.shape)
    ax.view_init(elev=24, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("white")
    plt.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(
        buf,
        format="png",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_recovery_projection(
    original,
    current,
    predicted_damage,
    added_mask=None,
    scale=8,
):
    live_projection = (current > 0).max(axis=1)
    missing_projection = ((original > 0) & (current == 0)).max(axis=1)
    predicted_projection = ((predicted_damage > 0) & (current > 0)).max(axis=1)
    if added_mask is None:
        added_projection = np.zeros_like(live_projection)
    else:
        added_projection = (added_mask > 0).max(axis=1)

    width = original.shape[0] * scale
    height = original.shape[2] * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for x in range(original.shape[0]):
        for z in range(original.shape[2]):
            color = None
            if live_projection[x, z]:
                color = (0, 18, 210)
            if missing_projection[x, z]:
                color = (240, 20, 20)
            if predicted_projection[x, z]:
                color = (255, 190, 0)
            if added_projection[x, z]:
                color = (0, 170, 70)

            if color is None:
                continue

            y0 = (original.shape[2] - z - 1) * scale
            draw.rectangle(
                [x * scale, y0, (x + 1) * scale - 1, y0 + scale - 1],
                fill=color,
            )

    return image


def damage_accuracies(predicted_damage, true_damage, damaged):
    alive_mask = damaged > 0
    boundary_mask = (true_damage > 0) & alive_mask
    if alive_mask.sum() == 0:
        full_acc = 0.0
    else:
        full_acc = float(((predicted_damage == true_damage) & alive_mask).sum()) / float(
            alive_mask.sum()
        )

    if boundary_mask.sum() == 0:
        boundary_acc = 0.0
    else:
        boundary_acc = float(
            ((predicted_damage == true_damage) & boundary_mask).sum()
        ) / float(boundary_mask.sum())

    return full_acc, boundary_acc


if __name__ == "__main__":
    main()
