import argparse
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
        default="configs/train_local.yaml",
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

    if mode == "local":
        train_from_config(config)
    elif mode == "modal":
        run_modal(config)
    else:
        raise ValueError(f"Unsupported training mode: {mode}")


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
    image = modal.Image.debian_slim().pip_install(requirements).add_local_python_source(
        "regen"
    )
    secrets = [modal.Secret.from_dict(env_variables)] if env_variables else []

    @app.function(
        gpu=run_config.get("gpu", "A10G"),
        image=image,
        secrets=secrets,
        timeout=run_config.get("timeout_seconds", 60 * 60 * 21),
    )
    def train_remote(remote_config, shapes, labels, class_to_idx):
        from regen.train_config import train_from_config

        train_from_config(
            remote_config,
            shapes=shapes,
            labels=labels,
            class_to_idx=class_to_idx,
        )

    shapes, labels, class_to_idx = load_training_data(config)
    with app.run():
        train_remote.remote(config, shapes, labels, class_to_idx)


if __name__ == "__main__":
    main()