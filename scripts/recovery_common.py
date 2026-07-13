import os
from pathlib import Path

import numpy as np

from regen.dataset import DynamicDamageDataset
from regen.device import preferred_device
from regen.model import CellRecoveryModel
from regen.train_config import load_config
from regen.utils import apply_damage
from regen.visualization import sample_recovery_steps, save_recovery_gif


def add_common_recovery_args(parser):
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--weights-filename", default="pytorch_model.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--device",
        default=None,
        help="Device override. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument("--recovery-iterations", type=int, default=128)
    parser.add_argument("--recovery-frame-duration", type=int, default=250)
    parser.add_argument("--recovery-frame-stride", type=int, default=4)
    parser.add_argument("--recovery-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--recovery-confidence-window", type=int, default=12)
    parser.add_argument("--recovery-confidence-required", type=int, default=6)
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
    parser.add_argument("--recovery-max-additions-per-step", type=int, default=None)
    parser.add_argument(
        "--recovery-no-progress-patience",
        type=int,
        default=12,
        help="Continue this many prediction passes after a no-addition step.",
    )
    parser.add_argument(
        "--recovery-extra-steps-after-complete",
        type=int,
        default=8,
        help="Continue this many passes after missing reaches 0.",
    )
    parser.add_argument(
        "--unconstrained-recovery",
        action="store_true",
        help="Allow recovery outside the known original shape mask.",
    )


def load_recovery_model(args):
    device = preferred_device(args.device)
    model = CellRecoveryModel.from_pretrained(
        args.repo_id,
        device=device,
        filename=args.weights_filename,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Loaded pretrained model: {args.repo_id} on {device}")
    return model


def run_recovery(model, original, damaged, args, description):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output
    original = original.astype(np.uint8)
    damaged = damaged.astype(np.uint8)

    print(
        f"Recovering {description}: "
        f"{int(damaged.sum())}/{int(original.sum())} live voxels, "
        f"{args.recovery_iterations} max iterations"
    )
    trajectory = model.recover(
        damaged,
        original_mask=original,
        steps_per_prediction=args.steps,
        recovery_steps=args.recovery_iterations,
        confidence_threshold=args.recovery_confidence_threshold,
        confidence_window=args.recovery_confidence_window,
        confidence_required=args.recovery_confidence_required,
        max_additions_per_step=args.recovery_max_additions_per_step,
        constrain_to_original=not args.unconstrained_recovery,
        show_progress=True,
        no_progress_patience=args.recovery_no_progress_patience,
        extra_steps_after_complete=args.recovery_extra_steps_after_complete,
        consensus_min_votes=args.recovery_consensus_min_votes,
        single_vote_confidence_threshold=args.recovery_single_vote_confidence,
    )

    final_step = trajectory.steps[-1]
    sampled_frames = sample_recovery_steps(
        trajectory.steps,
        args.recovery_frame_stride,
    )
    print(
        "Recovery stopped after "
        f"{len(trajectory.steps) - 1} steps with "
        f"{int(final_step.voxels.sum())} live voxels and "
        f"missing={final_step.missing_count}, "
        f"extra={final_step.extra_count}, "
        f"added={final_step.total_added_count}"
    )
    print(
        f"Rendering {len(sampled_frames)}/{len(trajectory.steps)} recovery GIF frames "
        f"with stride={args.recovery_frame_stride}..."
    )
    save_recovery_gif(
        trajectory,
        output_path,
        frame_duration=args.recovery_frame_duration,
        show_progress=True,
        frame_stride=args.recovery_frame_stride,
    )
    print(f"Saved recovery GIF: {output_path}")


def load_recovery_config(path, data_root=None):
    config = load_config(path)
    dataset_config = config.setdefault("dataset", {})
    if data_root:
        dataset_config["root"] = data_root
        return config

    configured_root = Path(str(dataset_config.get("root", ""))).expanduser()
    if configured_root.exists():
        return config

    local_root = Path(__file__).resolve().parents[1] / "data" / "shapenet_voxels"
    if dataset_config.get("source") == "shapenet" and local_root.exists():
        dataset_config["root"] = str(local_root)
    return config


def make_recovery_dataset(config, shapes, labels, seed):
    dataset_config = config.get("dataset", {})
    return DynamicDamageDataset(
        shapes=shapes,
        labels=labels.tolist(),
        damage_radius_range=tuple(dataset_config.get("damage_radius_range", [1, 3])),
        damage_types=dataset_config.get("damage_types", ["sphere", "cube"]),
        random_proportion_range=tuple(
            dataset_config.get("random_proportion_range", [0.1, 0.2])
        ),
        num_damage_sites_range=tuple(
            dataset_config.get("num_damage_sites_range", [1, 1])
        ),
        min_damage_target_cells=dataset_config.get("min_damage_target_cells", 0),
        max_damage_attempts=dataset_config.get("max_damage_attempts", 1),
        fixed_damage=True,
        augment_rotations=dataset_config.get("augment_rotations", False),
        return_damage_mask=True,
        seed=seed,
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
