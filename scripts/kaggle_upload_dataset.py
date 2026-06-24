#!/usr/bin/env python3
"""Script để upload data lên Kaggle Dataset (requires kaggle CLI).

Usage:
    python scripts/kaggle_upload_dataset.py \
        --checkpoint data/checkpoints/best.pth \
        --data-dir data/train_30k_hard \
        --dataset-name your-username/train-30k-hard

Requirements:
    pip install kaggle
    kaggle datasets --help  # Should work after install
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path


def create_dataset_metadata(dataset_name: str, title: str, description: str) -> dict:
    """Create dataset-metadata.json for Kaggle."""
    return {
        "title": title,
        "id": dataset_name,
        "licenses": [{"name": "CC0-1.0"}],
        "description": description,
    }


def upload_checkpoint(checkpoint_path: Path, dataset_name: str):
    """Upload checkpoint to Kaggle dataset."""
    print(f"\n📦 Uploading checkpoint to {dataset_name}...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Copy checkpoint
        shutil.copy(checkpoint_path, tmpdir / "best.pth")
        
        # Create metadata
        metadata = create_dataset_metadata(
            dataset_name=dataset_name,
            title="X-VLM Checkpoint (mAP 80%)",
            description="Pre-trained X-VLM checkpoint for person re-identification (mAP 80%)",
        )
        with open(tmpdir / "dataset-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Upload
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(tmpdir)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            print(f"✅ Checkpoint uploaded successfully!")
            print(f"   URL: https://www.kaggle.com/datasets/{dataset_name}")
        else:
            print(f"❌ Upload failed:")
            print(result.stderr)
            return False
    
    return True


def upload_training_data(data_dir: Path, dataset_name: str):
    """Upload training data (30K hard subset) to Kaggle dataset."""
    print(f"\n📦 Uploading training data to {dataset_name}...")
    
    # Check required files
    required_files = [
        "train_30k_hard.jsonl",
        "train_30k_hard_vitpose.json",
        "boxes_30k.jsonl",
    ]
    
    missing = [f for f in required_files if not (data_dir / f).exists()]
    if missing:
        print(f"⚠️  Missing files: {missing}")
        print("   Upload will continue with available files.")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Copy data files
        for f in required_files:
            if (data_dir / f).exists():
                shutil.copy(data_dir / f, tmpdir / f)
                print(f"  ✓ Copied {f}")
        
        # Zip image folder (faster upload)
        if (data_dir / "train_webp").exists():
            print("  📦 Zipping train_webp/ (this may take a while)...")
            shutil.make_archive(
                str(tmpdir / "train_webp"),
                "zip",
                data_dir / "train_webp",
            )
            print(f"  ✓ Created train_webp.zip")
        else:
            print("  ⚠️  train_webp/ folder not found, skipping images")
        
        # Create metadata
        metadata = create_dataset_metadata(
            dataset_name=dataset_name,
            title="Train 30K Hard Subset for Person Re-ID",
            description=(
                "Hard negative mining dataset for person re-identification:\n"
                "- 30K challenging samples\n"
                "- VitPose keypoints\n"
                "- Bounding boxes\n"
                "- WebP images"
            ),
        )
        with open(tmpdir / "dataset-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Upload
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(tmpdir)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            print(f"✅ Training data uploaded successfully!")
            print(f"   URL: https://www.kaggle.com/datasets/{dataset_name}")
        else:
            print(f"❌ Upload failed:")
            print(result.stderr)
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload data to Kaggle for STAGE 1 training")
    parser.add_argument("--checkpoint", type=Path, help="Path to best.pth checkpoint")
    parser.add_argument("--data-dir", type=Path, help="Path to train_30k_hard directory")
    parser.add_argument("--checkpoint-dataset", help="Kaggle dataset name for checkpoint (e.g., username/xvlm-checkpoints)")
    parser.add_argument("--data-dataset", help="Kaggle dataset name for training data (e.g., username/train-30k-hard)")
    
    args = parser.parse_args()
    
    # Check Kaggle CLI
    try:
        import subprocess
        result = subprocess.run(["kaggle", "--version"], capture_output=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("❌ Kaggle CLI not found!")
        print("   Install: pip install kaggle")
        print("   Setup: https://www.kaggle.com/docs/api")
        return 1
    
    print("🚀 Kaggle Dataset Upload Tool")
    print("=" * 50)
    
    # Upload checkpoint
    if args.checkpoint and args.checkpoint_dataset:
        if not args.checkpoint.exists():
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            return 1
        
        if not upload_checkpoint(args.checkpoint, args.checkpoint_dataset):
            return 1
    
    # Upload training data
    if args.data_dir and args.data_dataset:
        if not args.data_dir.exists():
            print(f"❌ Data directory not found: {args.data_dir}")
            return 1
        
        if not upload_training_data(args.data_dir, args.data_dataset):
            return 1
    
    print("\n" + "=" * 50)
    print("✅ All uploads completed!")
    print("\nNext steps:")
    print("  1. Go to https://www.kaggle.com/code")
    print("  2. Create new notebook with GPU T4")
    print("  3. Add your datasets")
    print("  4. Follow KAGGLE_SETUP_GUIDE.md")
    
    return 0


if __name__ == "__main__":
    exit(main())
