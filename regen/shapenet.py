from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

VOXEL_SUFFIXES = {".npy", ".npz", ".binvox"}
NPZ_VOXEL_KEYS = ("voxels", "occupancy", "x", "arr_0")


def load_shapenet_voxels(
    root: str,
    categories: Optional[Sequence[str]] = None,
    max_shapes_per_class: Optional[int] = None,
    target_size: Optional[int] = None,
    occupancy_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load pre-voxelized ShapeNet shapes from category folders.

    Expected layout is flexible, but each category should be a direct child of
    root and contain .npy, .npz, or .binvox voxel grids somewhere below it:

        root/
          chair/**/*.npy
          table/**/*.npz
          03001627/**/*.binvox

    Returns:
        shapes: [N, D, H, W] float32 occupancy grids.
        labels: [N] int64 category labels.
        class_to_idx: mapping from category folder name to label index.
    """
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"ShapeNet root does not exist: {root_path}")

    category_dirs = _category_dirs(root_path, categories)
    if not category_dirs:
        raise ValueError(f"No ShapeNet category folders found in {root_path}")

    class_to_idx = {
        category_dir.name: idx for idx, category_dir in enumerate(category_dirs)
    }
    shapes: List[np.ndarray] = []
    labels: List[int] = []

    for category_dir in category_dirs:
        files = _voxel_files(category_dir)
        if max_shapes_per_class is not None:
            files = files[:max_shapes_per_class]

        for path in files:
            voxel = load_voxel_file(path)
            voxel = _as_occupancy_grid(voxel, occupancy_threshold)
            if target_size is not None:
                voxel = center_crop_or_pad(voxel, target_size)

            shapes.append(voxel.astype(np.float32))
            labels.append(class_to_idx[category_dir.name])

    if not shapes:
        raise ValueError(f"No voxel files found under {root_path}")

    return np.stack(shapes), np.asarray(labels, dtype=np.int64), class_to_idx


def load_voxel_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        for key in NPZ_VOXEL_KEYS:
            if key in data:
                return data[key]
        raise ValueError(
            f"{path} does not contain one of these voxel keys: {NPZ_VOXEL_KEYS}"
        )
    if suffix == ".binvox":
        return load_binvox(path)
    raise ValueError(f"Unsupported voxel file type: {path}")


def load_binvox(path: Path) -> np.ndarray:
    """Read a BINVOX occupancy grid without adding an external dependency."""
    with open(path, "rb") as f:
        line = f.readline().decode("ascii").strip()
        if not line.startswith("#binvox"):
            raise ValueError(f"Not a binvox file: {path}")

        dims = None
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("dim "):
                dims = tuple(int(item) for item in line.split()[1:4])
            elif line == "data":
                break
            elif line == "":
                raise ValueError(f"Unexpected end of binvox header: {path}")

        if dims is None:
            raise ValueError(f"Missing binvox dimensions: {path}")

        raw = np.frombuffer(f.read(), dtype=np.uint8)
        if len(raw) % 2 != 0:
            raise ValueError(f"Malformed binvox run-length data: {path}")

        values = raw[0::2]
        counts = raw[1::2]
        dense = np.repeat(values, counts).astype(bool)
        expected = int(np.prod(dims))
        if dense.size != expected:
            raise ValueError(
                f"Malformed binvox data in {path}: expected {expected}, got {dense.size}"
            )

        return dense.reshape(dims)


def center_crop_or_pad(voxel: np.ndarray, target_size: int) -> np.ndarray:
    """Center crop/pad a cubic voxel grid to target_size on each axis."""
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    result = voxel
    for axis in range(3):
        size = result.shape[axis]
        if size > target_size:
            start = (size - target_size) // 2
            end = start + target_size
            result = np.take(result, indices=range(start, end), axis=axis)
        elif size < target_size:
            before = (target_size - size) // 2
            after = target_size - size - before
            pad_width = [(0, 0), (0, 0), (0, 0)]
            pad_width[axis] = (before, after)
            result = np.pad(result, pad_width, mode="constant")

    return result


def _category_dirs(root: Path, categories: Optional[Sequence[str]]) -> List[Path]:
    if categories:
        category_dirs = [root / category for category in categories]
        missing = [path for path in category_dirs if not path.is_dir()]
        if missing:
            missing_str = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing ShapeNet category folders: {missing_str}")
        return category_dirs

    return sorted(path for path in root.iterdir() if path.is_dir())


def _voxel_files(category_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in category_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VOXEL_SUFFIXES
    )


def _as_occupancy_grid(voxel: np.ndarray, threshold: float) -> np.ndarray:
    voxel = np.asarray(voxel).squeeze()
    if voxel.ndim != 3:
        raise ValueError(f"Expected a 3D voxel grid, got shape {voxel.shape}")
    return (voxel > threshold).astype(np.float32)
