import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regen.train_config import load_config, load_training_data, train_from_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train cube-regen from a YAML config.")
    parser.add_argument("--config", default="configs/train_combined.yaml")
    parser.add_argument("--mode", choices=["local", "modal"], default=None)
    parser.add_argument("--use-adaptive-pooling", type=parse_bool, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    apply_model_overrides(config, args)
    mode = args.mode or config.get("run", {}).get("mode", "local")
    print_summary(config, mode, args.config)

    if mode == "local":
        train_from_config(config)
        return
    if mode == "modal":
        run_modal(config)
        return
    raise ValueError(f"Unsupported training mode: {mode}")


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def apply_model_overrides(config, args):
    if args.use_adaptive_pooling is not None:
        config.setdefault("model", {})["use_adaptive_pooling"] = (
            args.use_adaptive_pooling
        )


def print_summary(config, mode, config_path):
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    training = config.get("training", {})
    output = config.get("output", {})
    summary = {
        "config": config_path,
        "mode": mode,
        "dataset": dataset.get("source", "npy"),
        "model": model.get("type", "combined"),
        "epochs": training.get("epochs"),
        "batch_size": training.get("batch_size"),
        "min_steps_per_sample": training.get("min_steps_per_sample"),
        "max_steps_per_sample": training.get("max_steps_per_sample"),
        "save_dir": output.get("save_dir"),
        "repo_id": output.get("repo_id"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_modal(config):
    try:
        import modal
    except ImportError as exc:
        raise ImportError("Install `modal` or run with `--mode local`.") from exc

    run_config = config.get("run", {})
    app = modal.App(run_config.get("app_name", "cube-regen-train"))
    image = (
        modal.Image.debian_slim()
        .pip_install(
            run_config.get(
                "requirements",
                [
                    "numpy",
                    "torch",
                    "matplotlib",
                    "Pillow",
                    "tqdm",
                    "huggingface_hub",
                    "PyYAML",
                    "wandb",
                ],
            )
        )
        .add_local_python_source("regen")
    )
    secrets = make_secrets(modal, run_config)
    volumes = {
        volume["mount_path"]: modal.Volume.from_name(
            volume["name"],
            create_if_missing=volume.get("create_if_missing", False),
        )
        for volume in run_config.get("volumes", [])
    }

    @app.function(
        gpu=run_config.get("gpu", "H100"),
        image=image,
        secrets=secrets,
        volumes=volumes,
        timeout=run_config.get("timeout_seconds", 60 * 60 * 21),
        serialized=True,
    )
    def train_remote(remote_config, shapes=None, labels=None, class_to_idx=None):
        from regen.train_config import train_from_config

        train_from_config(
            remote_config,
            shapes=shapes,
            labels=labels,
            class_to_idx=class_to_idx,
        )

    shapes, labels, class_to_idx = modal_data_payload(config)
    detach = run_config.get("detach", True)
    with app.run(detach=detach):
        if detach:
            call = train_remote.spawn(config, shapes, labels, class_to_idx)
            call_id = getattr(call, "object_id", None) or getattr(
                call,
                "function_call_id",
                None,
            )
            print(f"Spawned detached Modal training call: {call_id or call}")
        else:
            train_remote.remote(config, shapes, labels, class_to_idx)


def make_secrets(modal, run_config):
    values = {
        key: os.environ[key]
        for key in run_config.get("secret_env", ["HF_TOKEN", "WANDB_API_KEY"])
        if key in os.environ
    }
    return [modal.Secret.from_dict(values)] if values else []


def modal_data_payload(config):
    if config.get("dataset", {}).get("source") == "shapenet":
        return None, None, None
    return load_training_data(config)


if __name__ == "__main__":
    main()
