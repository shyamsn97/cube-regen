import argparse
import csv
import json
import os
import random
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regen.train_config import load_config


URL_FIELDS = ("url", "download_url", "source", "source_url", "href")
PATH_FIELDS = ("path", "file", "filepath", "key", "name")
CATEGORY_FIELDS = ("category", "synset", "synset_id", "class", "label")
DEFAULT_SHAPENET_ARCHIVE_URL = "https://cvgl.stanford.edu/data2/ShapeNetVox32.tgz"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "cube-regen" / "shapenet"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download only a random ShapeNet subset from a manifest."
    )
    parser.add_argument("--config", default=None, help="Training YAML config.")
    parser.add_argument("--manifest", default=None, help="Local path or URL manifest.")
    parser.add_argument(
        "--archive-url",
        default=None,
        help="ShapeNet voxel tar archive URL. Used when no manifest is configured.",
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--sample-categories", type=int, default=None)
    parser.add_argument("--max-shapes-per-class", type=int, default=None)
    parser.add_argument("--category-seed", type=int, default=None)
    parser.add_argument("--shape-seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def value_or_default(value, default):
    return default if value is None else value


def main():
    args = parse_args()
    config = load_config(args.config) if args.config else {}
    dataset_config = config.get("dataset", {})

    manifest = (
        args.manifest
        or os.environ.get("SHAPENET_MANIFEST_URL")
        or os.environ.get("SHAPENET_MANIFEST")
        or dataset_config.get("manifest_url")
        or dataset_config.get("manifest")
        or dataset_config.get("index_url")
    )
    if manifest:
        manifest = os.path.expandvars(str(manifest))
        if "$" in manifest:
            manifest = None

    archive_url = (
        args.archive_url
        or os.environ.get("SHAPENET_ARCHIVE_URL")
        or dataset_config.get("archive_url")
        or dataset_config.get("source_archive_url")
        or DEFAULT_SHAPENET_ARCHIVE_URL
    )
    archive_url = os.path.expandvars(str(archive_url))

    output_root = Path(
        args.output_root or dataset_config.get("root", "data/shapenet_voxels")
    )
    sample_categories = value_or_default(
        args.sample_categories, dataset_config.get("sample_categories", 10)
    )
    max_shapes_per_class = value_or_default(
        args.max_shapes_per_class, dataset_config.get("max_shapes_per_class", 1)
    )
    category_seed = value_or_default(
        args.category_seed, dataset_config.get("category_seed", config.get("seed", 0))
    )
    shape_seed = value_or_default(
        args.shape_seed, dataset_config.get("shape_seed", config.get("seed", 0))
    )

    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_root} already exists. Pass --force to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    archive_path = None
    if manifest:
        records = load_manifest(manifest)
        source_info = {"source_manifest": manifest}
    else:
        archive_path = download_archive(archive_url, Path(args.cache_dir))
        records = load_archive_records(archive_path)
        source_info = {"source_archive": archive_url}

    grouped = group_records(records)
    selected = sample_records(
        grouped,
        sample_categories=sample_categories,
        max_shapes_per_class=max_shapes_per_class,
        category_seed=category_seed,
        shape_seed=shape_seed,
    )

    sample_manifest = {
        **source_info,
        "output_root": str(output_root),
        "sample_categories": sample_categories,
        "max_shapes_per_class": max_shapes_per_class,
        "category_seed": category_seed,
        "shape_seed": shape_seed,
        "samples": [],
    }

    if archive_path:
        with tarfile.open(archive_path, "r:*") as tar:
            write_selected_records(selected, output_root, sample_manifest, tar)
    else:
        write_selected_records(selected, output_root, sample_manifest)

    manifest_path = output_root / "sample_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(sample_manifest, f, indent=2)

    total_files = sum(len(sample["files"]) for sample in sample_manifest["samples"])
    print(
        f"Downloaded {total_files} files across "
        f"{len(sample_manifest['samples'])} categories"
    )
    print(f"sample dir: {output_root}")
    print(f"manifest: {manifest_path}")
    for sample in sample_manifest["samples"]:
        print(sample["category"])
        for file_record in sample["files"]:
            print(f"  {file_record['path']}")


def load_manifest(manifest):
    text = read_text(manifest)
    suffix = Path(urllib.parse.urlparse(manifest).path).suffix.lower()
    if suffix == ".json":
        return normalize_json_records(json.loads(text))
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        return normalize_csv_records(text, delimiter)
    return normalize_line_records(text)


def read_text(path_or_url):
    parsed = urllib.parse.urlparse(path_or_url)
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(path_or_url) as response:
            return response.read().decode("utf-8")
    with open(Path(path_or_url).expanduser(), "r") as f:
        return f.read()


def download_archive(url, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.urlparse(url).path).name or "shapenet_voxels.tgz"
    archive_path = cache_dir / filename
    if archive_path.exists() and archive_path.stat().st_size > 0:
        print(f"Using cached ShapeNet archive: {archive_path}")
        return archive_path

    print(f"Downloading ShapeNet voxel archive: {url}")
    print(f"Cache path: {archive_path}")
    urllib.request.urlretrieve(url, archive_path)
    return archive_path


def load_archive_records(archive_path):
    records = []
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            path = Path(member.name)
            if path.suffix.lower() not in {".binvox", ".npy", ".npz"}:
                continue
            record = archive_record(member.name)
            if record:
                records.append(record)
    if not records:
        raise ValueError(f"No voxel files found in archive: {archive_path}")
    return records


def archive_record(member_name):
    parts = Path(member_name).parts
    if len(parts) < 2:
        return None

    category_index = 1 if parts[0].lower().startswith("shapenet") else 0
    if len(parts) <= category_index:
        return None

    category = parts[category_index]
    relative_parts = parts[category_index:]
    return {
        "category": category,
        "path": str(Path(*relative_parts)),
        "member": member_name,
    }


def normalize_json_records(data):
    if isinstance(data, dict):
        for key in ("files", "records", "items", "shapes"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("JSON manifest must be a list or contain files/records/items")
    return [normalize_record(record) for record in data]


def normalize_csv_records(text, delimiter):
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    if not rows:
        raise ValueError("CSV/TSV manifest has no rows")
    return [normalize_record(row) for row in rows]


def normalize_line_records(text):
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        records.append(normalize_record({"url": line}))
    if not records:
        raise ValueError("Line manifest has no URLs")
    return records


def normalize_record(record):
    url = first_present(record, URL_FIELDS)
    if not url:
        raise ValueError(f"Manifest record is missing a URL field: {record}")
    path = first_present(record, PATH_FIELDS) or infer_path(url)
    category = first_present(record, CATEGORY_FIELDS) or infer_category(path)
    if not category:
        raise ValueError(f"Could not infer category for manifest record: {record}")
    return {"category": str(category), "path": str(path), "url": str(url)}


def first_present(record, fields):
    for field in fields:
        value = record.get(field)
        if value:
            return value
    return None


def infer_path(url):
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.unquote(parsed.path).lstrip("/") or Path(url).name


def infer_category(path):
    parts = Path(path).parts
    return parts[0] if parts else None


def group_records(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["category"], []).append(record)
    if not grouped:
        raise ValueError("Manifest produced no grouped ShapeNet records")
    return grouped


def sample_records(grouped, sample_categories, max_shapes_per_class, category_seed, shape_seed):
    categories = sorted(grouped)
    if sample_categories is not None:
        if sample_categories <= 0:
            raise ValueError("--sample-categories must be positive")
        if sample_categories > len(categories):
            raise ValueError(
                f"Cannot sample {sample_categories} categories from {len(categories)}"
            )
        category_rng = random.Random(category_seed)
        categories = sorted(category_rng.sample(categories, sample_categories))

    shape_rng = random.Random(shape_seed)
    selected = []
    for category in categories:
        records = sorted(grouped[category], key=lambda record: record["path"])
        if max_shapes_per_class is not None and len(records) > max_shapes_per_class:
            records = sorted(
                shape_rng.sample(records, max_shapes_per_class),
                key=lambda record: record["path"],
            )
        elif max_shapes_per_class is not None:
            records = records[:max_shapes_per_class]
        selected.append((category, records))
    return selected


def destination_path(category, record):
    path = Path(record["path"])
    parts = path.parts
    if parts and parts[0] == category:
        return path
    return Path(category) / path.name


def write_selected_records(selected, output_root, sample_manifest, tar=None):
    for category, category_records in selected:
        sample = {"category": category, "files": []}
        for record in category_records:
            destination = output_root / destination_path(category, record)
            destination.parent.mkdir(parents=True, exist_ok=True)

            if tar:
                extract_archive_member(tar, record["member"], destination)
            else:
                fetch(record["url"], destination)

            file_record = {"path": str(destination.relative_to(output_root))}
            if "url" in record:
                file_record["url"] = record["url"]
            if "member" in record:
                file_record["archive_member"] = record["member"]
            sample["files"].append(file_record)
        sample_manifest["samples"].append(sample)


def extract_archive_member(tar, member_name, destination):
    source = tar.extractfile(member_name)
    if source is None:
        raise ValueError(f"Could not extract archive member: {member_name}")
    with source, open(destination, "wb") as output:
        shutil.copyfileobj(source, output)


def fetch(url, destination):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in {"http", "https"}:
        urllib.request.urlretrieve(url, destination)
        return
    if parsed.scheme == "file":
        shutil.copy2(Path(urllib.request.url2pathname(parsed.path)), destination)
        return
    source_path = Path(url).expanduser()
    if source_path.exists():
        shutil.copy2(source_path, destination)
        return
    raise ValueError(f"Unsupported or missing source URL/path: {url}")


if __name__ == "__main__":
    main()
