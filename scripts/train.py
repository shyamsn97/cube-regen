import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regen.train_config import load_config, load_training_data, train_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train cube-regen models locally or on Modal from one YAML config."
    )
    parser.add_argument(
        "--config",
        default="configs/train_combined.yaml",
        help="Path to a YAML training config.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "modal"],
        default=None,
        help="Override run.mode from the config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    mode = args.mode or config.get("run", {}).get("mode", "local")
    print_training_summary(config, mode=mode, config_path=args.config)

    if mode == "local":
        train_from_config(config)
    elif mode == "modal":
        run_modal(config)
    else:
        raise ValueError(f"Unsupported training mode: {mode}")


def print_training_summary(config, mode, config_path=None):
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    output_config = config.get("output", {})
    run_config = config.get("run", {})
    model_type = model_config.get("type", "combined")
    default_wandb_project = (
        "nca-3d-combined" if model_type == "combined" else "nca-3d-damage-detection"
    )

    summary = {
        "config": config_path,
        "mode": mode,
        "seed": config.get("seed", 0),
        "run": {
            "app_name": run_config.get("app_name"),
            "gpu": run_config.get("gpu"),
            "detach": run_config.get("detach"),
            "timeout_seconds": run_config.get("timeout_seconds"),
        },
        "dataset": {
            "source": dataset_config.get("source", "npy"),
            "root": dataset_config.get("root"),
            "shapes_path": dataset_config.get("shapes_path"),
            "labels_path": dataset_config.get("labels_path"),
            "sample_categories": dataset_config.get("sample_categories"),
            "max_shapes_per_class": dataset_config.get("max_shapes_per_class"),
            "target_size": dataset_config.get("target_size"),
            "val_split": dataset_config.get("val_split"),
            "damage_radius_range": dataset_config.get("damage_radius_range"),
            "damage_types": dataset_config.get("damage_types"),
            "num_damage_sites_range": dataset_config.get("num_damage_sites_range"),
            "min_damage_target_cells": dataset_config.get("min_damage_target_cells", 0),
            "max_damage_attempts": dataset_config.get("max_damage_attempts", 1),
            "augment_rotations": dataset_config.get("augment_rotations"),
        },
        "model": {
            "type": model_type,
            "num_hidden_channels": model_config.get("num_hidden_channels"),
            "num_damage_directions": model_config.get("num_damage_directions"),
            "clip_range": model_config.get("clip_range"),
            "use_tanh": model_config.get("use_tanh"),
            "train_perception": model_config.get("train_perception"),
        },
        "training": {
            "epochs": training_config.get("epochs"),
            "batch_size": training_config.get("batch_size"),
            "lr": training_config.get("lr"),
            "iterations_per_epoch": training_config.get("iterations_per_epoch"),
            "steps_per_sample": training_config.get("steps_per_sample"),
            "damage_loss_weight": training_config.get("damage_loss_weight", 1.0),
            "class_loss_weight": training_config.get("class_loss_weight", 1.0),
            "damage_class_weight": training_config.get("damage_class_weight", 1.0),
            "damage_loss_type": training_config.get(
                "damage_loss_type",
                "cross_entropy",
            ),
            "focal_gamma": training_config.get("focal_gamma", 2.0),
            "buffer_size": training_config.get("buffer_size"),
            "buffer_sampling_prob": training_config.get("buffer_sampling_prob"),
            "grad_clip": training_config.get("grad_clip"),
            "gradient_checkpointing": training_config.get("gradient_checkpointing"),
            "num_workers": training_config.get("num_workers"),
            "wandb_project": training_config.get(
                "wandb_project",
                default_wandb_project,
            ),
            "wandb_run_name": training_config.get("wandb_run_name"),
            "wandb_watch": training_config.get("wandb_watch", True),
            "wandb_watch_log": training_config.get("wandb_watch_log", "all"),
            "wandb_watch_log_freq": training_config.get("wandb_watch_log_freq", 100),
            "wandb_log_gradient_sums": training_config.get(
                "wandb_log_gradient_sums",
                True,
            ),
            "wandb_gradient_log_freq": training_config.get(
                "wandb_gradient_log_freq", 1
            ),
        },
        "output": {
            "checkpoint_dir": output_config.get("checkpoint_dir")
            or output_config.get("save_dir"),
            "save_frequency": output_config.get("save_frequency"),
            "validate_frequency": output_config.get("validate_frequency"),
            "repo_id": output_config.get("repo_id")
            or output_config.get("model_repo_id"),
            "repo_type": output_config.get("repo_type"),
        },
    }

    print("Resolved training parameters:")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_modal(config):
    try:
        import modal
    except ImportError as exc:
        raise ImportError(
            "Modal training requires the `modal` package. Install it or run with "
            "`--mode local`."
        ) from exc

    run_config = config.get("run", {})
    requirements = run_config.get(
        "requirements",
        [
            "numpy",
            "torch",
            "tensorboard",
            "matplotlib",
            "wandb",
            "Pillow",
            "tqdm",
            "huggingface_hub",
            "PyYAML",
        ],
    )
    env_variables = {
        key: os.environ.get(key)
        for key in run_config.get("secret_env", ["HF_TOKEN", "WANDB_API_KEY"])
        if os.environ.get(key) is not None
    }

    app = modal.App(run_config.get("app_name", "nca-3d-trainer"))
    image = (
        modal.Image.debian_slim()
        .pip_install(requirements)
        .add_local_python_source("regen")
    )
    secrets = [modal.Secret.from_dict(env_variables)] if env_variables else []
    volumes = {
        volume_config["mount_path"]: modal.Volume.from_name(
            volume_config["name"],
            create_if_missing=volume_config.get("create_if_missing", False),
        )
        for volume_config in run_config.get("volumes", [])
    }

    @app.function(
        gpu=run_config.get("gpu", "A10G"),
        image=image,
        secrets=secrets,
        volumes=volumes,
        timeout=run_config.get("timeout_seconds", 60 * 60 * 21),
        serialized=True,
    )
    def train_remote(remote_config, shapes=None, labels=None, class_to_idx=None):
        from regen.train_config import train_from_config

        print_training_summary(remote_config, mode="modal-remote")

        train_from_config(
            remote_config,
            shapes=shapes,
            labels=labels,
            class_to_idx=class_to_idx,
        )

    detach = run_config.get("detach", True)
    if config.get("dataset", {}).get("source") == "shapenet":
        shapes, labels, class_to_idx = None, None, None
    else:
        shapes, labels, class_to_idx = load_training_data(config)

    with app.run(detach=detach):
        if detach:
            call = train_remote.spawn(config, shapes, labels, class_to_idx)
            call_id = getattr(call, "object_id", None) or getattr(
                call, "function_call_id", None
            )
            print(f"Spawned Modal training call in detached app: {call_id or call}")
        else:
            train_remote.remote(config, shapes, labels, class_to_idx)


if __name__ == "__main__":
    main()
