#!/usr/bin/env python3
"""Generate standing 3D voxel text that reads "SAKANA AI"."""

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
TEXT = "SAKANA AI"
LETTER_SCALE = 2
LETTER_DEPTH = 6
CHAR_SPACING = 2
WORD_SPACING = 6
UNDERLINE_THICKNESS = 2
PADDING_X = 4
PADDING_Y = 2
PADDING_Z = 4


FONT = {
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "I": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "11111",
    ],
    "K": [
        "10001",
        "10010",
        "10100",
        "11000",
        "10100",
        "10010",
        "10001",
    ],
    "N": [
        "10001",
        "11001",
        "10101",
        "10011",
        "10001",
        "10001",
        "10001",
    ],
    "S": [
        "11111",
        "10000",
        "10000",
        "11110",
        "00001",
        "00001",
        "11110",
    ],
}


def text_dimensions(text):
    width = 0
    for char in text:
        if char == " ":
            width += WORD_SPACING
        else:
            width += len(FONT[char][0]) * LETTER_SCALE + CHAR_SPACING

    if text and text[-1] != " ":
        width -= CHAR_SPACING

    height = len(next(iter(FONT.values()))) * LETTER_SCALE
    depth = LETTER_DEPTH
    return width, height, depth


def draw_letter(voxel, char, x_offset, z_offset, y_range):
    pattern = FONT[char]
    letter_height = len(pattern)

    for row_idx, row in enumerate(pattern):
        for col_idx, value in enumerate(row):
            if value != "1":
                continue

            x0 = x_offset + col_idx * LETTER_SCALE
            x1 = x0 + LETTER_SCALE
            z0 = z_offset + (letter_height - row_idx - 1) * LETTER_SCALE
            z1 = z0 + LETTER_SCALE
            voxel[x0:x1, y_range[0] : y_range[1], z0:z1] = 1.0


def draw_underline(voxel, x_start, x_end, z_offset, y_range):
    z0 = z_offset - UNDERLINE_THICKNESS
    z1 = z_offset
    voxel[x_start:x_end, y_range[0] : y_range[1], z0:z1] = 1.0


def generate_sakana_ai_voxel(text=TEXT):
    """Return a binary voxel grid of upright, connected extruded block letters."""
    text_width, text_height, text_depth = text_dimensions(text)
    shape = (
        text_width + 2 * PADDING_X,
        text_depth + 2 * PADDING_Y,
        text_height + 2 * PADDING_Z,
    )
    voxel = np.zeros(shape, dtype=np.float32)

    x_offset = PADDING_X
    z_offset = PADDING_Z
    y_range = (PADDING_Y, PADDING_Y + text_depth)
    underline_start = x_offset
    for char in text:
        if char == " ":
            x_offset += WORD_SPACING
            continue

        draw_letter(voxel, char, x_offset, z_offset, y_range)
        x_offset += len(FONT[char][0]) * LETTER_SCALE + CHAR_SPACING

    underline_end = PADDING_X + text_width
    draw_underline(voxel, underline_start, underline_end, z_offset, y_range)

    return voxel


def render_voxel_text(voxel, size=(12, 4)):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(voxel == 1, facecolors="blue", edgecolor="k", linewidth=0.08, alpha=0.85)

    ax.set_xlim(0, voxel.shape[0])
    ax.set_ylim(0, voxel.shape[1])
    ax.set_zlim(0, voxel.shape[2])
    ax.set_box_aspect(voxel.shape)
    ax.view_init(elev=30, azim=-60)
    ax.set_proj_type("ortho")
    ax.set_axis_off()

    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("white")
    plt.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(
        buf, format="png", facecolor="white", bbox_inches="tight", pad_inches=0.1
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def save_sakana_assets(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    voxel = generate_sakana_ai_voxel()

    voxel_path = output_dir / "sakana_ai_voxel.npy"
    image_path = output_dir / "sakana_ai_voxel.png"

    np.save(voxel_path, voxel)
    image = render_voxel_text(voxel)
    image.save(image_path)

    return voxel_path, image_path, int(voxel.sum())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a 3D Sakana AI-inspired voxel and preview image."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory for sakana_ai_voxel.npy and sakana_ai_voxel.png.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    voxel_path, image_path, live_cells = save_sakana_assets(args.output_dir)
    print(f"Saved voxel: {voxel_path} ({live_cells} live cells)")
    print(f"Saved preview: {image_path}")


if __name__ == "__main__":
    main()
