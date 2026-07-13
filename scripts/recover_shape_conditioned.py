#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recovery_common import (  # noqa: E402
    add_common_recovery_args,
    apply_recovery_damage_overrides,
    load_recovery_config,
    load_recovery_model,
    make_recovery_dataset,
    run_recovery,
)
from regen.train_config import load_training_data  # noqa: E402

DEFAULT_CLASS_NAMES = {
    0: "plane",
    1: "chair",
    2: "car",
    3: "table",
    4: "cabinet",
    5: "lamp",
    6: "bench",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render recovery from an NPY sample using a shape-conditioned model."
    )
    add_common_recovery_args(parser)
    parser.set_defaults(
        repo_id="shyamsn97/cube-regen-shape-conditioned-damage-hdim-full-64",
        output_dir=Path("examples/recovery/shape_conditioned"),
        output="shape_conditioned_recovery.gif",
    )
    parser.add_argument(
        "--config",
        default="configs/train_damage_shape_conditioned_modal.yaml",
        help="Shape-conditioned training YAML used for data loading and damage settings.",
    )
    parser.add_argument(
        "--class-label",
        default=None,
        help="Optional numeric class label to recover from.",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help="Optional class name alias, e.g. table/chair/plane.",
    )
    parser.add_argument(
        "--shape-seed",
        default=0,
        help="Override dataset.shape_seed for selecting the object per class.",
    )
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_recovery_config(args.config)
    apply_shape_seed_override(config, args.shape_seed)
    apply_recovery_damage_overrides(config, args)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    np.random.seed(seed)

    shapes, labels, _ = load_training_data(config)
    class_label = resolve_class_label(args.class_label, args.class_name)
    shapes, labels = filter_class_label(shapes, labels, class_label)
    dataset = make_recovery_dataset(config, shapes, labels, seed)
    sample_index = validate_sample_index(args.sample_index, len(dataset))
    damaged, _, label, original = dataset[sample_index]
    label_id = int(label.item())
    class_name = DEFAULT_CLASS_NAMES.get(label_id, str(label_id))
    shape_seed = config.get("dataset", {}).get("shape_seed", config.get("seed", 0))
    print(
        f"Selected shape-conditioned sample: class={label_id} ({class_name}), "
        f"sample_index={sample_index}, shape_seed={shape_seed}, "
        f"live_voxels={int(original.sum().item())}"
    )

    model = load_recovery_model(args)
    run_recovery(
        model=model,
        original=original.numpy(),
        damaged=damaged.numpy(),
        args=args,
        description=f"shape-conditioned {class_name} sample {sample_index}",
    )


def apply_shape_seed_override(config, shape_seed):
    if shape_seed in (None, ""):
        return
    config.setdefault("dataset", {})["shape_seed"] = int(shape_seed)


def resolve_class_label(class_label, class_name):
    if class_label not in (None, ""):
        return int(class_label)
    if class_name in (None, ""):
        return None

    normalized = str(class_name).lower()
    for label, name in DEFAULT_CLASS_NAMES.items():
        if normalized == name:
            return label
    available = ", ".join(DEFAULT_CLASS_NAMES.values())
    raise ValueError(f"Unknown class name '{class_name}'. Available: {available}")


def filter_class_label(shapes, labels, class_label):
    if class_label in (None, ""):
        return shapes, labels
    class_label = int(class_label)

    indices = np.where(labels == class_label)[0]
    if len(indices) == 0:
        available = ", ".join(str(label) for label in sorted(np.unique(labels)))
        raise ValueError(
            f"No samples found for class label {class_label}. Available: {available}"
        )
    return shapes[indices], labels[indices]


def validate_sample_index(sample_index, dataset_size):
    if dataset_size <= 0:
        raise ValueError("No samples available for recovery.")
    if sample_index < 0 or sample_index >= dataset_size:
        raise ValueError(
            f"--sample-index must be in [0, {dataset_size - 1}], got {sample_index}."
        )
    return sample_index


if __name__ == "__main__":
    main()
