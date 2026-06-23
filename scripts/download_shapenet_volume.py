import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download voxelized ShapeNet data from a Modal Volume."
    )
    parser.add_argument("--volume", default="shapenet-voxels")
    parser.add_argument("--remote-path", default="/shapenet_voxels")
    parser.add_argument("--local-dir", default="data")
    return parser.parse_args()


def main():
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "modal",
        "volume",
        "get",
        args.volume,
        args.remote_path,
        str(local_dir),
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
