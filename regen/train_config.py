import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from regen.combined_trainer import CombinedNCA3DTrainer
from regen.dataset import DynamicDamageDataset
from regen.model import NCA3DCombinedDamageClassifier, NCA3DDamageDetection
from regen.shapenet import load_shapenet_voxels
from regen.trainer import NCA3DTrainer

Config = Dict[str, Any]


def load_config(path: str) -> Config:
    config_path = Path(path)
    with open(path, "r") as f:
        if config_path.suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "YAML training configs require PyYAML. Install the package with "
                    "`pip install -e .` from this repo."
                ) from exc
            return yaml.safe_load(f)

        return json.load(f)


def load_training_data(
    config: Config,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, int]]]:
    dataset_config = config.get("dataset", {})
    source = dataset_config.get("source", "npy")

    if source == "npy":
        shapes = np.load(
            Path(dataset_config.get("shapes_path", "data/xdata_7class.npy"))
        )
        labels = np.load(
            Path(dataset_config.get("labels_path", "data/ydata_7class.npy"))
        )
        return shapes.astype(np.float32), labels.astype(np.int64), None

    if source == "shapenet":
        shapes, labels, class_to_idx = load_shapenet_voxels(
            root=dataset_config["root"],
            categories=dataset_config.get("categories"),
            sample_categories=dataset_config.get("sample_categories"),
            category_seed=dataset_config.get("category_seed", config.get("seed", 0)),
            shape_seed=dataset_config.get("shape_seed", config.get("seed", 0)),
            max_shapes_per_class=dataset_config.get("max_shapes_per_class"),
            target_size=dataset_config.get("target_size"),
            occupancy_threshold=dataset_config.get("occupancy_threshold", 0.5),
        )
        return shapes, labels, class_to_idx

    raise ValueError(f"Unsupported dataset source: {source}")


def train_from_config(
    config: Config,
    shapes: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
):
    seed = config.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if shapes is None or labels is None:
        shapes, labels, class_to_idx = load_training_data(config)

    labels = np.asarray(labels, dtype=np.int64)
    model_config = config.get("model", {})
    model_type = model_config.get("type", "combined")

    if model_type == "combined":
        return train_combined_from_config(config, shapes, labels, class_to_idx)
    if model_type == "damage":
        return train_damage_from_config(config, shapes, labels)

    raise ValueError(f"Unsupported model type: {model_type}")


def train_combined_from_config(
    config: Config,
    shapes: np.ndarray,
    labels: np.ndarray,
    class_to_idx: Optional[Dict[str, int]] = None,
):
    seed = config.get("seed", 0)
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    output_config = config.get("output", {})

    num_classes = model_config.get("num_classes")
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    train_indices, val_indices = stratified_split(
        labels,
        dataset_config.get("val_split", 0.1),
        seed,
    )
    train_dataset = make_dynamic_damage_dataset(
        shapes[train_indices],
        labels[train_indices].tolist(),
        dataset_config,
        seed=seed,
        fixed_damage=False,
    )

    val_dataset = None
    if len(val_indices) > 0:
        val_dataset = make_dynamic_damage_dataset(
            shapes[val_indices],
            labels[val_indices].tolist(),
            dataset_config,
            seed=seed + 1,
            fixed_damage=True,
        )

    random.seed(seed)
    np.random.seed(seed)

    model = NCA3DCombinedDamageClassifier(
        num_hidden_channels=model_config.get("num_hidden_channels", 20),
        num_classes=num_classes,
        num_damage_directions=model_config.get("num_damage_directions", 7),
        alpha_living_threshold=model_config.get("alpha_living_threshold", 0.1),
        cell_fire_rate=model_config.get("cell_fire_rate", 0.5),
        clip_range=model_config.get("clip_range", 64.0),
        use_tanh=model_config.get("use_tanh", False),
        freeze_perception=not model_config.get("train_perception", False),
    )

    trainer = CombinedNCA3DTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=training_config.get("batch_size", 8),
        lr=training_config.get("lr", 2e-5),
        iterations_per_epoch=_optional_positive(
            training_config.get("iterations_per_epoch", 100)
        ),
        steps_per_sample=training_config.get("steps_per_sample", 96),
        damage_loss_weight=training_config.get("damage_loss_weight", 1.0),
        class_loss_weight=training_config.get("class_loss_weight", 1.0),
        buffer_size=training_config.get("buffer_size", 1000),
        buffer_sampling_prob=training_config.get("buffer_sampling_prob", 0.5),
        grad_clip=training_config.get("grad_clip", 1.0),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        device=_device(training_config.get("device")),
        checkpoint_dir=output_config.get("checkpoint_dir", "combined_nca_models"),
        num_workers=training_config.get("num_workers", 0),
        validate_frequency=output_config.get("validate_frequency", 1),
        repo_id=output_config.get("repo_id"),
        repo_type=output_config.get("repo_type", "model"),
    )

    if class_to_idx is not None:
        _save_class_mapping(
            output_config.get("checkpoint_dir", "combined_nca_models"), class_to_idx
        )

    print(
        "Training combined model "
        f"({len(train_dataset)} train, "
        f"{len(val_dataset) if val_dataset is not None else 0} val, "
        f"{num_classes} classes)"
    )
    trainer.train(
        epochs=training_config.get("epochs", 500),
        save_frequency=output_config.get("save_frequency", 10),
    )
    return trainer


def train_damage_from_config(config: Config, shapes: np.ndarray, labels: np.ndarray):
    seed = config.get("seed", 0)
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    output_config = config.get("output", {})

    dataset = make_dynamic_damage_dataset(
        shapes,
        labels.tolist(),
        dataset_config,
        seed=seed,
        fixed_damage=dataset_config.get("fixed_damage", False),
        filter_label=dataset_config.get("filter_label"),
    )
    model = NCA3DDamageDetection(
        use_class_embeddings=model_config.get("use_class_embeddings", True),
        num_hidden_channels=model_config.get("num_hidden_channels", 20),
        num_classes=model_config.get("num_classes", int(labels.max()) + 1),
        num_damage_directions=model_config.get("num_damage_directions", 7),
        alpha_living_threshold=model_config.get("alpha_living_threshold", 0.1),
        cell_fire_rate=model_config.get("cell_fire_rate", 0.5),
        clip_range=model_config.get("clip_range", 64.0),
        use_tanh=model_config.get("use_tanh", False),
    )
    trainer = NCA3DTrainer(
        model=model,
        dataset=dataset,
        batch_size=training_config.get("batch_size", 8),
        lr=training_config.get("lr", 2e-5),
        iterations_per_epoch=training_config.get("iterations_per_epoch", 100),
        steps_per_sample=training_config.get("steps_per_sample", 96),
        buffer_size=training_config.get("buffer_size", 1000),
        buffer_sampling_prob=training_config.get("buffer_sampling_prob", 0.5),
        grad_clip=training_config.get("grad_clip", 1.0),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        device=_device(training_config.get("device")),
        save_dir=output_config.get("save_dir", "nca_models"),
        repo_id=output_config.get("repo_id", "shyamsn97/cube"),
        repo_type=output_config.get("repo_type", "model"),
        model_repo_id=output_config.get(
            "model_repo_id",
            "shyamsn97/cube-regen-damage-detection",
        ),
    )
    trainer.train(
        epochs=training_config.get("epochs", 500),
        save_frequency=output_config.get("save_frequency", 10),
        visualization_frequency=output_config.get("visualization_frequency", 10),
    )
    return trainer


def make_dynamic_damage_dataset(
    shapes,
    labels,
    dataset_config: Config,
    seed: int,
    fixed_damage: bool,
    filter_label: Optional[int] = None,
):
    return DynamicDamageDataset(
        shapes=shapes,
        labels=labels,
        damage_radius_range=tuple(dataset_config.get("damage_radius_range", [1, 3])),
        damage_types=dataset_config.get(
            "damage_types",
            ["sphere", "cube", "random"],
        ),
        random_proportion_range=tuple(
            dataset_config.get("random_proportion_range", [0.1, 0.2])
        ),
        fixed_damage=fixed_damage,
        augment_rotations=dataset_config.get("augment_rotations", False),
        return_damage_mask=True,
        seed=seed,
        filter_label=filter_label,
    )


def stratified_split(labels, val_split, seed):
    if val_split <= 0:
        return np.arange(len(labels)), np.array([], dtype=np.int64)
    if val_split >= 1:
        raise ValueError("dataset.val_split must be less than 1.0")

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


def _optional_positive(value):
    if value is None or value <= 0:
        return None
    return value


def _device(value):
    return torch.device(value) if value else None


def _save_class_mapping(checkpoint_dir: str, class_to_idx: Dict[str, int]):
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2, sort_keys=True)
