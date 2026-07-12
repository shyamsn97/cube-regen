import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regen.dataset import DynamicDamageDataset
from regen.model import (
    create_model_from_config,
    load_config_from_json,
    load_weights_from_huggingface,
)
from regen.train_config import load_config, load_training_data
from regen.utils import plot_voxels


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run inference and render prediction images. The default mode renders "
            "ShapeNet combined-model rows: Image | Damage (no class) | Predicted | True."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["shapenet-combined", "legacy-damage"],
        default="shapenet-combined",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face model repo. Defaults depend on --mode.",
    )
    parser.add_argument(
        "--checkpoint",
        default="combined_latest.pt",
        help="Checkpoint filename in the HF repo.",
    )
    parser.add_argument(
        "--config",
        default="configs/train_shapenet_modal.yaml",
        help="YAML config used for data sampling and damage settings.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Local ShapeNet voxel root. Defaults to config root, or data/shapenet_voxels for Modal configs.",
    )
    parser.add_argument("--output-dir", default="examples/shapenet_predictions")
    parser.add_argument("--output", default="combined_img.png")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "legacy-damage":
        legacy_damage_inference(
            model_repo_id=(
                args.repo_id or "shyamsn97/table-cube-regen-damage-detection-20"
            ),
            output_path=args.output,
            steps=args.steps or 128,
        )
        return

    shapenet_combined_inference(args)


def shapenet_combined_inference(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = load_config(args.config)
    config = with_local_data_root(config, args.data_root)

    seed = args.seed if args.seed is not None else config.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)

    repo_id = args.repo_id or "shyamsn97/shapenet-cube-regen-combined-48"
    model = load_combined_model(repo_id, args.checkpoint, device)
    model.eval()

    shapes, labels, class_to_idx = load_training_data(config)
    idx_to_class = {idx: name for name, idx in (class_to_idx or {}).items()}
    dataset_config = config.get("dataset", {})
    dataset = DynamicDamageDataset(
        shapes=shapes,
        labels=labels.tolist(),
        damage_radius_range=tuple(dataset_config.get("damage_radius_range", [1, 3])),
        damage_types=dataset_config.get("damage_types", ["sphere", "cube", "random"]),
        random_proportion_range=tuple(
            dataset_config.get("random_proportion_range", [0.1, 0.2])
        ),
        fixed_damage=True,
        augment_rotations=dataset_config.get("augment_rotations", False),
        return_damage_mask=True,
        seed=seed,
    )

    steps = args.steps or config.get("training", {}).get("steps_per_sample", 64)
    sample_count = min(args.num_samples, len(dataset))
    saved_paths = []

    for idx in range(sample_count):
        saved_paths.append(
            render_shapenet_sample(
                model=model,
                dataset=dataset,
                sample_idx=idx,
                idx_to_class=idx_to_class,
                steps=steps,
                device=device,
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

    local_root = Path("data/shapenet_voxels")
    if local_root.exists():
        dataset_config["root"] = str(local_root)
        return config

    raise FileNotFoundError(
        "Could not find ShapeNet voxel data. Pass --data-root, or run "
        "`make shapenet-subset-download` to create data/shapenet_voxels."
    )


def load_combined_model(repo_id, checkpoint_name, device):
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        repo_type="model",
    )
    model_config = load_config_from_json(config_path)
    model = create_model_from_config(model_config)

    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_name,
        repo_type="model",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"Loaded {checkpoint_name} from {repo_id}")
    return model


def render_shapenet_sample(
    model,
    dataset,
    sample_idx,
    idx_to_class,
    steps,
    device,
    output_dir,
    image_size,
):
    damaged_shape, damage_direction, label, original_shape = dataset[sample_idx]
    label_id = int(label.item())
    true_class = idx_to_class.get(label_id, str(label_id))

    state = model.initialize(damaged_shape.unsqueeze(0).to(device))
    with torch.no_grad():
        for _ in range(steps):
            state = model(state)
        damage_logits = model.damage_logits(state)
        predicted_damage = torch.argmax(damage_logits, dim=-1).squeeze(0)
        class_logits = model.class_logits(state)
        predicted_label = int(torch.argmax(class_logits, dim=-1).item())

    predicted_class = idx_to_class.get(predicted_label, str(predicted_label))
    damage_acc = damage_accuracy(
        predicted_damage.cpu(),
        damage_direction,
        damaged_shape,
    )

    original_np = original_shape.numpy().astype(np.uint8)
    damaged_np = damaged_shape.numpy().astype(np.uint8)
    true_damage_np = damage_direction.numpy().astype(np.uint8)
    predicted_damage_np = predicted_damage.cpu().numpy().astype(np.uint8)
    zeros = np.zeros_like(true_damage_np, dtype=np.uint8)

    panels = [
        (
            plot_voxels(original_np, zeros, size=(image_size, image_size)),
            f"Image\ntrue class: {true_class}",
        ),
        (
            plot_voxels(damaged_np, zeros, size=(image_size, image_size)),
            "Damage (no class)",
        ),
        (
            plot_voxels(damaged_np, predicted_damage_np, size=(image_size, image_size)),
            f"Predicted\nclass: {predicted_class}\ndamage acc: {damage_acc:.3f}",
        ),
        (
            plot_voxels(damaged_np, true_damage_np, size=(image_size, image_size)),
            f"True\nclass: {true_class}",
        ),
    ]

    row = make_labeled_row(panels)
    output_path = output_dir / f"sample_{sample_idx:03d}.png"
    row.save(output_path)
    return output_path


def damage_accuracy(predicted_damage, true_damage, damaged_shape):
    alive_mask = damaged_shape > 0
    if alive_mask.sum().item() == 0:
        return 0.0
    correct = (predicted_damage == true_damage) & alive_mask
    return correct.float().sum().item() / alive_mask.float().sum().item()


def legacy_damage_inference(
    model_repo_id="shyamsn97/table-cube-regen-damage-detection-20",
    output_path="combined_img.png",
    steps=128,
):
    loaded_model, config = load_weights_from_huggingface(
        model=None,  # Will create model from config
        repo_id=model_repo_id,
        filename="pytorch_model.pt",
        load_config=True,
        config_filename="config.json",
    )
    del config
    labels = np.load(PROJECT_ROOT / "data" / "ydata_7class.npy")
    shapes = np.load(PROJECT_ROOT / "data" / "xdata_7class.npy")

    dataset = DynamicDamageDataset(
        shapes,
        labels,
        damage_radius_range=(2, 3),
        damage_types=["sphere", "cube"],
        random_proportion_range=(0.1, 0.2),
        fixed_damage=False,
        augment_rotations=False,
        return_damage_mask=True,
        seed=None,
        filter_label=3,
    )

    damage_mask, damage_direction, label, _ = dataset[0]
    state = loaded_model.initialize(damage_mask.unsqueeze(0))
    with torch.no_grad():
        for _ in range(steps):
            state = loaded_model(state, label.unsqueeze(0))
            predictions = loaded_model.classify(state)
            predictions = torch.argmax(predictions, dim=-1)
    print(predictions.shape)
    print(damage_mask.shape)
    print(damage_direction.shape)
    nonzero_pred = predictions.squeeze() != 0
    zero_pred = predictions.squeeze() == 0
    print(
        "Damage Direction Accuracy: ",
        torch.mean(
            (
                predictions.squeeze()[nonzero_pred]
                == damage_direction.squeeze()[nonzero_pred]
            ).float()
        ),
    )
    print(
        "Undamaged Accuracy: ",
        torch.mean(
            (
                predictions.squeeze()[zero_pred] == damage_direction.squeeze()[zero_pred]
            ).float()
        ),
    )
    ground_truth_img = plot_voxels(
        live_mask=damage_mask.squeeze().detach().cpu().numpy().astype(np.uint8),
        damage_direction=damage_direction.squeeze().detach().cpu().numpy().astype(
            np.uint8
        ),
    )
    predicted_img = plot_voxels(
        live_mask=damage_mask.squeeze().detach().cpu().numpy().astype(np.uint8),
        damage_direction=predictions.squeeze().detach().cpu().numpy().astype(np.uint8),
    )

    combined_img = make_labeled_row(
        [(predicted_img, "Prediction"), (ground_truth_img, "Ground Truth")]
    )
    combined_img.save(output_path)
    print(f"Saved {output_path}")


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