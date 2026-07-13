import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regen.dataset import DynamicDamageDataset
from regen.device import preferred_device
from regen.model import CellRecoveryModel
from regen.train_config import load_config, load_training_data
from regen.utils import plot_voxels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render damage prediction rows from a pretrained CellRecoveryModel."
    )
    parser.add_argument(
        "--repo-id",
        default="shyamsn97/shapenet-cube-regen-combined-hdim-48",
        help="Hugging Face repo id or local pretrained model directory.",
    )
    parser.add_argument(
        "--weights-filename",
        default="pytorch_model.pt",
        help="Weights filename inside the pretrained model repo/directory.",
    )
    parser.add_argument(
        "--config",
        default="configs/train_shapenet_modal.yaml",
        help="Training YAML used only for data sampling and damage settings.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Local ShapeNet root override for ShapeNet configs.",
    )
    parser.add_argument("--output-dir", default="examples/predictions")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Device override. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument("--image-size", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = preferred_device(args.device)
    config = with_local_data_root(load_config(args.config), args.data_root)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = CellRecoveryModel.from_pretrained(
        args.repo_id,
        device=device,
        filename=args.weights_filename,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Loaded pretrained model: {args.repo_id} on {device}")

    shapes, labels, class_to_idx = load_training_data(config)
    idx_to_class = {idx: name for name, idx in (class_to_idx or {}).items()}
    dataset = make_dataset(config, shapes, labels, seed)
    steps = args.steps or config.get("training", {}).get("max_steps_per_sample", 128)

    saved_paths = []
    for idx in range(min(args.num_samples, len(dataset))):
        saved_paths.append(
            render_sample(
                model=model,
                dataset=dataset,
                sample_idx=idx,
                idx_to_class=idx_to_class,
                steps=steps,
                output_dir=output_dir,
                image_size=args.image_size,
            )
        )

    print(f"Saved {len(saved_paths)} prediction rows to {output_dir}")
    for path in saved_paths:
        print(path)


def with_local_data_root(config, data_root):
    config = copy.deepcopy(config)
    dataset_config = config.setdefault("dataset", {})
    if data_root:
        dataset_config["root"] = data_root
        return config

    configured_root = Path(str(dataset_config.get("root", ""))).expanduser()
    if configured_root.exists():
        return config

    local_root = PROJECT_ROOT / "data" / "shapenet_voxels"
    if local_root.exists():
        dataset_config["root"] = str(local_root)
        return config

    if dataset_config.get("source", "npy") == "shapenet":
        raise FileNotFoundError(
            "Could not find ShapeNet voxel data. Pass --data-root, or run "
            "`make shapenet-subset-download` to create data/shapenet_voxels."
        )
    return config


def make_dataset(config, shapes, labels, seed):
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
        filter_label=dataset_config.get("filter_label"),
    )


def render_sample(
    model,
    dataset,
    sample_idx,
    idx_to_class,
    steps,
    output_dir,
    image_size,
):
    damaged_shape, true_damage, label, original_shape = dataset[sample_idx]
    label_id = int(label.item())
    true_class = idx_to_class.get(label_id, str(label_id))

    prediction = model.predict(
        damaged_shape.numpy(),
        steps=steps,
        class_label=label_id,
        shape_condition=original_shape.numpy(),
    )
    predicted_damage = prediction.damage_labels.squeeze(0).cpu().numpy().astype(np.uint8)
    predicted_class = None
    if prediction.class_label is not None:
        predicted_label = int(prediction.class_label.item())
        predicted_class = idx_to_class.get(predicted_label, str(predicted_label))

    damage_acc = damage_accuracy(
        torch.tensor(predicted_damage),
        true_damage,
        damaged_shape,
    )

    original_np = original_shape.numpy().astype(np.uint8)
    damaged_np = damaged_shape.numpy().astype(np.uint8)
    true_damage_np = true_damage.numpy().astype(np.uint8)
    zeros = np.zeros_like(true_damage_np, dtype=np.uint8)

    prediction_label = f"Predicted\ndamage acc: {damage_acc:.3f}"
    if predicted_class is not None:
        prediction_label = f"Predicted\nclass: {predicted_class}\ndamage acc: {damage_acc:.3f}"

    panels = [
        (
            plot_voxels(original_np, zeros, size=(image_size, image_size)),
            f"Image\ntrue class: {true_class}",
        ),
        (
            plot_voxels(damaged_np, zeros, size=(image_size, image_size)),
            "Damage",
        ),
        (
            plot_voxels(damaged_np, predicted_damage, size=(image_size, image_size)),
            prediction_label,
        ),
        (
            plot_voxels(damaged_np, true_damage_np, size=(image_size, image_size)),
            "True damage",
        ),
    ]

    output_path = output_dir / f"sample_{sample_idx:03d}.png"
    make_labeled_row(panels).save(output_path)
    return output_path


def damage_accuracy(predicted_damage, true_damage, damaged_shape):
    alive_mask = damaged_shape > 0
    if alive_mask.sum().item() == 0:
        return 0.0
    correct = (predicted_damage == true_damage) & alive_mask
    return correct.float().sum().item() / alive_mask.float().sum().item()


def make_labeled_row(panels):
    label_height = 58
    padding = 16
    font = load_font(18)
    panel_images = [image.convert("RGBA") for image, _ in panels]
    width = sum(image.width for image in panel_images) + padding * (len(panels) + 1)
    image_height = max(image.height for image in panel_images)
    height = image_height + label_height + padding * 2

    row = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(row)
    x = padding
    for image, (_, label) in zip(panel_images, panels):
        paste_with_alpha(row, image, (x, padding))
        draw_centered_multiline(
            draw,
            label,
            x,
            padding + image_height + 4,
            image.width,
            font,
        )
        x += image.width + padding
    return row.convert("RGB")


def paste_with_alpha(canvas, image, xy):
    if image.mode == "RGBA":
        canvas.paste(image, xy, image)
    else:
        canvas.paste(image, xy)


def draw_centered_multiline(draw, text, x, y, width, font):
    lines = text.splitlines()
    line_height = max(font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines)
    for offset, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (x + (width - text_width) / 2, y + offset * (line_height + 2)),
            line,
            fill=(0, 0, 0),
            font=font,
        )


def load_font(size):
    for font_name in ("Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
