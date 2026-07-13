#!/usr/bin/env python3
"""Run Sakana damage-direction inference on deterministic sphere/cube damage."""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_sakana_voxel import DEFAULT_OUTPUT_DIR
from regen.device import preferred_device
from regen.model import CellRecoveryModel
from regen.utils import apply_damage, plot_voxels
from regen.visualization import sample_recovery_steps, save_recovery_gif
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
    parser.add_argument(
        "--device",
        default=None,
        help="Device override. Defaults to cuda, then mps, then cpu.",
    )
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
        "--recovery-frame-stride",
        type=int,
        default=4,
        help=(
            "Render every Nth recovery step in the 3D GIF, while always keeping "
            "the final frame."
        ),
    )
    parser.add_argument(
        "--recovery-start-mode",
        choices=["damage", "seed"],
        default="damage",
        help="Start recovery from normal damage or from a tiny live-cell seed.",
    )
    parser.add_argument(
        "--recovery-seed-cells",
        type=int,
        default=64,
        help="Number of original live cells kept when --recovery-start-mode=seed.",
    )
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
        "--recovery-confidence-window",
        type=int,
        default=12,
        help="Number of recent prediction passes used for recovery voting.",
    )
    parser.add_argument(
        "--recovery-confidence-required",
        type=int,
        default=6,
        help="Votes required within --recovery-confidence-window before adding a voxel.",
    )
    parser.add_argument(
        "--recovery-consensus-min-votes",
        type=int,
        default=2,
        help="Independent target votes required before spatial consensus accepts a repair.",
    )
    parser.add_argument(
        "--recovery-single-vote-confidence",
        type=float,
        default=0.99,
        help="Confidence required to accept a repair with only one spatial vote.",
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

    device = preferred_device(args.device)
    voxel = load_or_generate_voxel(args.voxel_path)
    model = load_sakana_model(args, device)

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
        damaged = make_recovery_start(voxel, args)
        print(
            "Starting recovery from "
            f"{int(damaged.sum())}/{int(voxel.sum())} live voxels "
            f"for {args.recovery_iterations} iterations"
        )
        trajectory = model.recover(
            damaged,
            original_mask=voxel.astype(np.uint8),
            steps_per_prediction=args.steps,
            recovery_steps=args.recovery_iterations,
            confidence_threshold=args.recovery_confidence_threshold,
            confidence_window=args.recovery_confidence_window,
            confidence_required=args.recovery_confidence_required,
            max_additions_per_step=args.recovery_max_additions_per_step,
            constrain_to_original=not args.unconstrained_recovery,
            show_progress=True,
            consensus_min_votes=args.recovery_consensus_min_votes,
            single_vote_confidence_threshold=args.recovery_single_vote_confidence,
        )
        final_step = trajectory.steps[-1]
        print(
            "Recovery stopped after "
            f"{len(trajectory.steps) - 1} steps with "
            f"{int(final_step.voxels.sum())} live voxels and "
            f"missing={final_step.missing_count}, "
            f"extra={final_step.extra_count}, "
            f"added={final_step.total_added_count}"
        )
        frame_count = len(
            sample_recovery_steps(
                trajectory.steps,
                args.recovery_frame_stride,
            )
        )
        skipped_frame_count = len(trajectory.steps) - frame_count
        print(
            f"Rendering {frame_count}/{len(trajectory.steps)} recovery GIF frames "
            f"with stride={args.recovery_frame_stride}; "
            f"skipped {skipped_frame_count} warm-up/sampled frames..."
        )
        save_recovery_gif(
            trajectory,
            gif_path,
            frame_duration=args.recovery_frame_duration,
            show_progress=True,
            frame_stride=args.recovery_frame_stride,
        )
        print(f"Saved Sakana recovery GIF: {gif_path}")


def load_sakana_model(args, device):
    if args.checkpoint is not None:
        model = CellRecoveryModel.from_pretrained(
            args.checkpoint,
            device=device,
            filename=args.hf_filename,
            config_filename=args.hf_config_filename,
        )
        print(f"Loaded local Sakana checkpoint: {args.checkpoint}")
        return model

    if args.repo_id:
        model = CellRecoveryModel.from_pretrained(
            pretrained_model_name_or_path=args.repo_id,
            device=device,
            token=os.environ.get("HF_TOKEN"),
            filename=args.hf_filename,
            config_filename=args.hf_config_filename,
        )
        print(f"Loaded Sakana model from Hugging Face: {args.repo_id}")
        return model

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
    image_size,
):
    center = choose_live_center(voxel, center_fraction)
    damaged, true_damage = apply_damage(
        voxel.copy(),
        radius=radius,
        damage_type=damage_type,
        damage_center=center,
    )
    prediction = model.predict(damaged, steps=steps)
    predicted_damage = (
        prediction.damage_labels.squeeze(0).detach().cpu().numpy().astype(np.uint8)
    )
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


def make_recovery_start(voxel, args):
    if args.recovery_start_mode == "damage":
        return apply_multi_spot_damage(
            voxel=voxel,
            damage_type=args.recovery_damage_type,
            radius=args.recovery_radius,
            center_fractions=args.recovery_center_fractions,
        )
    return seed_live_cells(voxel, args.recovery_seed_cells)


def seed_live_cells(voxel, count):
    live_indices = np.argwhere(voxel == 1)
    if len(live_indices) == 0:
        raise ValueError("Cannot seed recovery from an empty voxel grid.")
    count = max(1, min(int(count), len(live_indices)))

    center = live_indices.mean(axis=0)
    distances = np.linalg.norm(live_indices - center, axis=1)
    selected = live_indices[np.argsort(distances)[:count]]
    seed = np.zeros_like(voxel, dtype=np.uint8)
    seed[tuple(selected.T)] = 1
    return seed


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
