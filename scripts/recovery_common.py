import os
from pathlib import Path

import numpy as np

from regen.dataset import DynamicDamageDataset
from regen.device import preferred_device
from regen.model import CellRecoveryModel
from regen.train_config import load_config
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
    parser.add_argument("--use-adaptive-pooling", type=parse_bool, default=None)
    parser.add_argument(
        "--force-download",
        action="store_true",
        default=True,
        help="Redownload Hugging Face config/weights even if cached (default).",
    )
    parser.add_argument(
        "--no-force-download",
        dest="force_download",
        action="store_false",
        help="Allow Hugging Face downloads to reuse cached config/weights.",
    )
    parser.add_argument("--recovery-iterations", type=int, default=128)
    parser.add_argument("--recovery-frame-duration", type=int, default=250)
    parser.add_argument("--recovery-frame-stride", type=int, default=4)
    parser.add_argument(
        "--damage-types",
        nargs="+",
        default=None,
        help="Override dataset damage types, e.g. sphere cube.",
    )
    parser.add_argument(
        "--damage-radius-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("MIN", "MAX"),
        help="Override dataset damage radius range.",
    )
    parser.add_argument(
        "--random-proportion-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Override random damage live-cell proportion range.",
    )
    parser.add_argument(
        "--num-damage-sites-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("MIN", "MAX"),
        help="Override number of damage spots.",
    )
    parser.add_argument(
        "--min-damage-target-cells",
        type=int,
        default=None,
        help="Override minimum removed cells for generated damage.",
    )
    parser.add_argument(
        "--max-damage-attempts",
        type=int,
        default=None,
        help="Override damage sampling attempts.",
    )
    parser.add_argument(
        "--recovery-start-mode",
        choices=["damage", "seed"],
        default="damage",
        help="Start from damaged shape or a small seed of live cells.",
    )
    parser.add_argument(
        "--recovery-seed-cells",
        type=int,
        default=64,
        help="Number of original live cells kept when --recovery-start-mode=seed.",
    )
    parser.add_argument(
        "--recovery-seed-proportion",
        type=float,
        default=None,
        help=(
            "Seed a fraction of each shape's live cells instead of a fixed count. "
            "Robust across shapes of very different sizes; overrides "
            "--recovery-seed-cells when set."
        ),
    )
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
    parser.add_argument(
        "--recovery-direction-probability-threshold",
        type=float,
        default=0.6,
        help=(
            "Minimum per-direction probability for fan-out repair votes. "
            "Only used when recovery consensus requires multiple spatial votes."
        ),
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
        dest="unconstrained_recovery",
        action="store_true",
        default=True,
        help="Allow recovery outside the known original shape mask (default).",
    )
    parser.add_argument(
        "--constrained-recovery",
        dest="unconstrained_recovery",
        action="store_false",
        help="Restrict recovery to the known original shape mask.",
    )


def load_recovery_model(args):
    device = preferred_device(args.device)
    print(f"Downloading latest Hugging Face model files: {args.repo_id}")
    config_overrides = {}
    if args.use_adaptive_pooling is not None:
        config_overrides["use_adaptive_pooling"] = args.use_adaptive_pooling
    model = CellRecoveryModel.from_pretrained(
        args.repo_id,
        device=device,
        filename=args.weights_filename,
        token=os.environ.get("HF_TOKEN"),
        force_download=args.force_download,
        config_overrides=config_overrides or None,
    )
    print(f"Loaded pretrained model: {args.repo_id} on {device}")
    return model


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def run_recovery(model, original, damaged, args, description):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output
    original = original.astype(np.uint8)
    start = recovery_start(original, damaged.astype(np.uint8), args)

    print(
        f"Recovering {description}: "
        f"{int(start.sum())}/{int(original.sum())} live voxels, "
        f"start={args.recovery_start_mode}, "
        f"{args.recovery_iterations} max iterations"
    )
    trajectory = model.recover(
        start,
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
        direction_probability_threshold=args.recovery_direction_probability_threshold,
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


def recovery_start(original, damaged, args):
    if args.recovery_start_mode == "damage":
        return damaged
    return seed_live_cells(original, seed_cell_count(original, args))


def seed_cell_count(original, args):
    num_live = int((original == 1).sum())
    if num_live == 0:
        raise ValueError("Cannot seed recovery from an empty voxel grid.")
    proportion = getattr(args, "recovery_seed_proportion", None)
    if proportion is not None:
        count = int(round(proportion * num_live))
    else:
        count = int(args.recovery_seed_cells)
    # Always leave at least one missing cell so recovery has something to do.
    return max(1, min(count, num_live - 1))


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


def apply_recovery_damage_overrides(config, args):
    dataset_config = config.setdefault("dataset", {})
    if args.damage_types is not None:
        dataset_config["damage_types"] = args.damage_types
    if args.damage_radius_range is not None:
        dataset_config["damage_radius_range"] = list(args.damage_radius_range)
    if args.random_proportion_range is not None:
        dataset_config["random_proportion_range"] = list(args.random_proportion_range)
    if args.num_damage_sites_range is not None:
        dataset_config["num_damage_sites_range"] = list(args.num_damage_sites_range)
    if args.min_damage_target_cells is not None:
        dataset_config["min_damage_target_cells"] = args.min_damage_target_cells
    if args.max_damage_attempts is not None:
        dataset_config["max_damage_attempts"] = args.max_damage_attempts


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
        recovery_seed_proportion_range=tuple(
            dataset_config.get("recovery_seed_proportion_range", [0.1, 0.9])
        ),
        center_seed_augment_damage_types=dataset_config.get(
            "center_seed_augment_damage_types",
            ["sphere", "cube", "random"],
        ),
        center_seed_augment_sites_range=tuple(
            dataset_config.get("center_seed_augment_sites_range", [0, 2])
        ),
        center_seed_augment_proportion_range=tuple(
            dataset_config.get("center_seed_augment_proportion_range", [0.02, 0.12])
        ),
        center_seed_augment_max_attempts=dataset_config.get(
            "center_seed_augment_max_attempts",
            8,
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
