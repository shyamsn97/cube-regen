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
    apply_multi_spot_damage,
    load_recovery_model,
    run_recovery,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a recovery GIF from a generated table/chair/plane voxel."
    )
    add_common_recovery_args(parser)
    parser.set_defaults(
        repo_id="shyamsn97/cube-regen-combined",
        output_dir=Path("examples/recovery/default_shapes"),
    )
    parser.add_argument(
        "--shape",
        choices=["table", "chair", "plane"],
        default="table",
    )
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--damage-type", choices=["sphere", "cube"], default="sphere")
    parser.add_argument("--damage-radius", type=int, default=3)
    parser.add_argument(
        "--damage-center-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = f"{args.shape}_recovery.gif"
    return args


def main():
    args = parse_args()
    original = make_default_shape(args.shape, args.size)
    damaged = apply_multi_spot_damage(
        original,
        damage_type=args.damage_type,
        radius=args.damage_radius,
        center_fractions=args.damage_center_fractions,
    )

    model = load_recovery_model(args)
    run_recovery(
        model=model,
        original=original,
        damaged=damaged,
        args=args,
        description=f"default {args.shape}",
    )


def make_default_shape(name, size):
    if size < 16:
        raise ValueError("--size must be at least 16 for the generated shapes.")
    shape = np.zeros((size, size, size), dtype=np.uint8)
    if name == "table":
        return make_table(shape)
    if name == "chair":
        return make_chair(shape)
    if name == "plane":
        return make_plane(shape)
    raise ValueError(f"Unsupported default shape: {name}")


def make_table(shape):
    size = shape.shape[0]
    x1, x2 = int(0.18 * size), int(0.82 * size)
    y1, y2 = int(0.18 * size), int(0.82 * size)
    top_z = int(0.62 * size)
    thickness = max(1, size // 16)
    leg = max(2, size // 10)

    shape[x1:x2, y1:y2, top_z : top_z + thickness] = 1
    for lx in (x1, x2 - leg):
        for ly in (y1, y2 - leg):
            shape[lx : lx + leg, ly : ly + leg, int(0.16 * size) : top_z] = 1
    return shape


def make_chair(shape):
    size = shape.shape[0]
    x1, x2 = int(0.24 * size), int(0.76 * size)
    y1, y2 = int(0.24 * size), int(0.76 * size)
    seat_z = int(0.42 * size)
    thickness = max(1, size // 16)
    leg = max(2, size // 10)

    shape[x1:x2, y1:y2, seat_z : seat_z + thickness] = 1
    shape[x1:x2, y2 - leg : y2, seat_z : int(0.85 * size)] = 1
    for lx in (x1, x2 - leg):
        for ly in (y1, y2 - leg):
            shape[lx : lx + leg, ly : ly + leg, int(0.12 * size) : seat_z] = 1
    return shape


def make_plane(shape):
    size = shape.shape[0]
    cx = size // 2
    cy = size // 2
    cz = size // 2
    body = max(1, size // 20)
    wing = max(1, size // 16)

    shape[int(0.14 * size) : int(0.86 * size), cy - body : cy + body + 1, cz] = 1
    shape[
        int(0.38 * size) : int(0.62 * size),
        int(0.16 * size) : int(0.84 * size),
        cz - wing : cz + wing + 1,
    ] = 1
    shape[
        int(0.72 * size) : int(0.86 * size),
        int(0.34 * size) : int(0.66 * size),
        cz : int(0.72 * size),
    ] = 1
    shape[int(0.82 * size) : int(0.9 * size), cy - body : cy + body + 1, cz] = 1
    return shape


if __name__ == "__main__":
    main()
