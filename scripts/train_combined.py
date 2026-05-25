import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regen.combined_trainer import CombinedNCA3DTrainer
from regen.dataset import DynamicDamageDataset
from regen.model import NCA3DCombinedDamageClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a combined NCA for damage directions and shape classes."
    )
    parser.add_argument("--shapes-path", default="data/xdata_7class.npy")
    parser.add_argument("--labels-path", default="data/ydata_7class.npy")
    parser.add_argument("--checkpoint-dir", default="combined_nca_models")
    parser.add_argument("--num-hidden-channels", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--num-damage-directions", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--iterations-per-epoch", type=int, default=100)
    parser.add_argument("--steps-per-sample", type=int, default=96)
    parser.add_argument("--damage-loss-weight", type=float, default=1.0)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--damage-radius-min", type=int, default=1)
    parser.add_argument("--damage-radius-max", type=int, default=3)
    parser.add_argument("--damage-types", default="sphere,cube,random")
    parser.add_argument("--random-proportion-min", type=float, default=0.1)
    parser.add_argument("--random-proportion-max", type=float, default=0.2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--buffer-sampling-prob", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-frequency", type=int, default=10)
    parser.add_argument("--validate-frequency", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-tanh", action="store_true")
    parser.add_argument("--train-perception", action="store_true")
    parser.add_argument("--augment-rotations", action="store_true")
    return parser.parse_args()


def stratified_split(labels, val_split, seed):
    if val_split <= 0:
        return np.arange(len(labels)), np.array([], dtype=np.int64)
    if val_split >= 1:
        raise ValueError("--val-split must be less than 1.0")

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_indices = []
    val_indices = []

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)

        if len(class_indices) <= 1:
            train_indices.extend(class_indices.tolist())
            continue

        num_val = max(1, int(round(len(class_indices) * val_split)))
        num_val = min(num_val, len(class_indices) - 1)
        val_indices.extend(class_indices[:num_val].tolist())
        train_indices.extend(class_indices[num_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.asarray(train_indices), np.asarray(val_indices)


def make_dataset(shapes, labels, args, seed, fixed_damage):
    damage_types = [item.strip() for item in args.damage_types.split(",") if item.strip()]
    return DynamicDamageDataset(
        shapes=shapes,
        labels=labels,
        damage_radius_range=(args.damage_radius_min, args.damage_radius_max),
        damage_types=damage_types,
        random_proportion_range=(
            args.random_proportion_min,
            args.random_proportion_max,
        ),
        fixed_damage=fixed_damage,
        augment_rotations=args.augment_rotations,
        return_damage_mask=True,
        seed=seed,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    shapes = np.load(Path(args.shapes_path))
    labels = np.load(Path(args.labels_path)).astype(np.int64)

    num_classes = args.num_classes
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    train_indices, val_indices = stratified_split(labels, args.val_split, args.seed)
    train_dataset = make_dataset(
        shapes[train_indices],
        labels[train_indices].tolist(),
        args,
        seed=args.seed,
        fixed_damage=False,
    )
    val_dataset = None
    if len(val_indices) > 0:
        val_dataset = make_dataset(
            shapes[val_indices],
            labels[val_indices].tolist(),
            args,
            seed=args.seed + 1,
            fixed_damage=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    iterations_per_epoch = args.iterations_per_epoch
    if iterations_per_epoch <= 0:
        iterations_per_epoch = None

    model = NCA3DCombinedDamageClassifier(
        num_hidden_channels=args.num_hidden_channels,
        num_classes=num_classes,
        num_damage_directions=args.num_damage_directions,
        use_tanh=args.use_tanh,
        freeze_perception=not args.train_perception,
    )

    device = torch.device(args.device) if args.device else None
    trainer = CombinedNCA3DTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        iterations_per_epoch=iterations_per_epoch,
        steps_per_sample=args.steps_per_sample,
        damage_loss_weight=args.damage_loss_weight,
        class_loss_weight=args.class_loss_weight,
        buffer_size=args.buffer_size,
        buffer_sampling_prob=args.buffer_sampling_prob,
        grad_clip=args.grad_clip,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
    )

    print(
        "Training combined model "
        f"({len(train_dataset)} train, "
        f"{len(val_dataset) if val_dataset is not None else 0} val, "
        f"{num_classes} classes)"
    )
    trainer.fit(
        epochs=args.epochs,
        save_frequency=args.save_frequency,
        validate_frequency=args.validate_frequency,
    )


if __name__ == "__main__":
    main()
