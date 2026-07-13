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
    load_recovery_config,
    load_recovery_model,
    make_recovery_dataset,
    run_recovery,
)
from regen.train_config import load_training_data  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a recovery GIF from a damaged ShapeNet voxel sample."
    )
    add_common_recovery_args(parser)
    parser.set_defaults(
        repo_id="shyamsn97/shapenet-cube-regen-combined-hdim-48",
        output_dir=Path("examples/recovery/shapenet"),
        output="shapenet_recovery.gif",
    )
    parser.add_argument(
        "--config",
        default="configs/train_shapenet_modal.yaml",
        help="ShapeNet training YAML used for data loading and damage settings.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--category",
        default=None,
        help="Optional ShapeNet category folder/name to sample from.",
    )
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_recovery_config(args.config, args.data_root)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    np.random.seed(seed)

    shapes, labels, class_to_idx = load_training_data(config)
    shapes, labels, category_label = filter_category(
        shapes,
        labels,
        class_to_idx,
        args.category,
    )
    dataset = make_recovery_dataset(config, shapes, labels, seed)
    sample_index = validate_sample_index(args.sample_index, len(dataset))
    damaged, _, label, original = dataset[sample_index]

    idx_to_class = {idx: name for name, idx in (class_to_idx or {}).items()}
    label_id = int(label.item())
    category_name = idx_to_class.get(label_id, str(label_id))
    if category_label is not None:
        category_name = args.category

    model = load_recovery_model(args)
    run_recovery(
        model=model,
        original=original.numpy(),
        damaged=damaged.numpy(),
        args=args,
        description=f"ShapeNet {category_name} sample {sample_index}",
    )


def filter_category(shapes, labels, class_to_idx, category):
    if not category:
        return shapes, labels, None
    if not class_to_idx:
        raise ValueError("--category requires a dataset with class names.")
    if category not in class_to_idx:
        available = ", ".join(sorted(class_to_idx))
        raise ValueError(f"Unknown category '{category}'. Available: {available}")

    label = class_to_idx[category]
    indices = np.where(labels == label)[0]
    if len(indices) == 0:
        raise ValueError(f"No ShapeNet samples found for category '{category}'.")
    return shapes[indices], labels[indices], label


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
