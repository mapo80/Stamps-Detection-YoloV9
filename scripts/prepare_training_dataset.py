"""
Download and prepare the stamp detection dataset from HuggingFace.
Run this on a training VM to get the dataset ready for YOLO training.

Usage:
    python scripts/prepare_training_dataset.py
    python scripts/prepare_training_dataset.py --output ./data
"""

import argparse
import subprocess
import zipfile
from pathlib import Path


HF_REPO = "mapo80/stamps"
BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
SPLITS = ["train", "val", "test"]


def download_file(url: str, dest: Path):
    """Download a file using wget or curl."""
    if dest.exists():
        print(f"  {dest.name} already exists, skipping download")
        return

    result = subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url])
    if result.returncode != 0:
        subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)


def main():
    parser = argparse.ArgumentParser(description="Download stamp detection dataset from HuggingFace")
    parser.add_argument("--output", type=str, default=".", help="Output directory (default: current dir)")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Dataset directory: {output}\n")

    # Download and extract each split
    for split in SPLITS:
        zip_name = f"{split}.zip"
        zip_path = output / zip_name
        url = f"{BASE_URL}/{zip_name}"

        # Skip if already extracted
        img_dir = output / "images" / split
        if img_dir.exists() and len(list(img_dir.glob("*"))) > 0:
            print(f"[{split}] Already extracted, skipping")
            continue

        print(f"[{split}] Downloading {zip_name}...")
        download_file(url, zip_path)

        print(f"[{split}] Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output)

        zip_path.unlink()
        print(f"[{split}] Done\n")

    # Download dataset.yaml
    yaml_path = output / "dataset.yaml"
    if not yaml_path.exists():
        print("Downloading dataset.yaml...")
        download_file(f"{BASE_URL}/dataset.yaml", yaml_path)

    # Verify
    print("\n" + "=" * 50)
    print("Dataset ready!")
    print("=" * 50)
    for split in SPLITS:
        img_dir = output / "images" / split
        lbl_dir = output / "labels" / split
        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
        print(f"  {split}: {n_img} images, {n_lbl} labels")

    print(f"\nDataset path: {output}")
    print(f"YAML config:  {yaml_path}")


if __name__ == "__main__":
    main()
