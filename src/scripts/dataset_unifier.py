"""
File: src/scripts/dataset_unifier.py
Purpose: Unify multiple datasets into a single training dataset
Author: HSE Vision Team
Date: 2024
Project: Construction Safety Detection System
"""

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetUnifier:
    """
    Unifies multiple YOLO format datasets into a single dataset.
    """

    def __init__(self, output_dir: str = "unified_dataset"):
        """
        Initialize dataset unifier.

        Args:
            output_dir: Directory to save unified dataset
        """
        self.output_dir = Path(output_dir)
        self.datasets = []
        self.class_mapping = {}
        self.unified_classes = []

    def add_dataset(self, dataset_path: str, weight: float = 1.0):
        """
        Add a dataset to be unified.

        Args:
            dataset_path: Path to dataset directory containing data.yaml
            weight: Weight for sampling from this dataset
        """
        dataset_path = Path(dataset_path)
        data_yaml_path = dataset_path / "data.yaml"

        if not data_yaml_path.exists():
            logger.error(f"data.yaml not found in {dataset_path}")
            return False

        try:
            with open(data_yaml_path, "r") as f:
                data_config = yaml.safe_load(f)

            # Validate dataset structure
            required_keys = ["train", "val", "names"]
            if not all(key in data_config for key in required_keys):
                logger.error(f"Invalid data.yaml format in {dataset_path}")
                return False

            dataset_info = {
                "path": dataset_path,
                "config": data_config,
                "weight": weight,
                "name": dataset_path.name,
            }

            self.datasets.append(dataset_info)
            logger.info(f"Added dataset: {dataset_path.name}")

            # Update class mapping
            self._update_class_mapping(data_config["names"])

            return True

        except Exception as e:
            logger.error(f"Error adding dataset {dataset_path}: {e}")
            return False

    def add_dataset_info(self, dataset_info: Dict):
        """
        Add a dataset info directly (used by auto-discovery).

        Args:
            dataset_info: Dataset information dictionary with corrected paths
        """
        try:
            # Validate dataset structure
            config = dataset_info["config"]
            required_keys = ["train", "val", "names"]
            if not all(key in config for key in required_keys):
                logger.error(f"Invalid data.yaml format in {dataset_info['name']}")
                return False

            self.datasets.append(dataset_info)
            logger.info(f"Added dataset: {dataset_info['name']}")

            # Update class mapping
            self._update_class_mapping(config["names"])

            return True

        except Exception as e:
            logger.error(f"Error adding dataset {dataset_info['name']}: {e}")
            return False

    def _update_class_mapping(self, class_names):
        """Update unified class mapping."""
        # Handle both list and dict formats
        if isinstance(class_names, list):
            for class_name in class_names:
                if class_name not in self.unified_classes:
                    self.unified_classes.append(class_name)
        elif isinstance(class_names, dict):
            for class_id, class_name in class_names.items():
                if class_name not in self.unified_classes:
                    self.unified_classes.append(class_name)
        else:
            logger.error(f"Unsupported class names format: {type(class_names)}")

    def _get_image_label_pairs(
        self, dataset_info: Dict, split: str
    ) -> List[Tuple[Path, Path]]:
        """
        Get image-label pairs for a specific split.

        Args:
            dataset_info: Dataset information dictionary
            split: Split name ('train', 'val', 'test')

        Returns:
            List of (image_path, label_path) tuples
        """
        pairs = []
        config = dataset_info["config"]
        dataset_path = dataset_info["path"]

        if split not in config:
            return pairs

        # Use the corrected path from config (already fixed in _discover_datasets)
        split_path = Path(config[split])

        if not split_path.exists():
            logger.warning(f"Split path not found: {split_path}")
            return pairs

        # Find images - look in the images directory
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        # Check if this is already an images directory or if we need to look for one
        if split_path.name == "images" or (split_path / "images").exists():
            images_dir = (
                split_path if split_path.name == "images" else split_path / "images"
            )
            labels_dir = images_dir.parent / "labels"
        else:
            # Fallback: treat split_path as containing both images and labels
            images_dir = split_path
            labels_dir = split_path

        for img_path in images_dir.rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                # Find corresponding label
                label_path = labels_dir / f"{img_path.stem}.txt"

                if label_path.exists():
                    pairs.append((img_path, label_path))
                else:
                    logger.warning(f"Label not found for image: {img_path}")

        return pairs

    def _convert_label(self, label_path: Path, source_classes) -> List[str]:
        """
        Convert label file to unified class indices.

        Args:
            label_path: Path to label file
            source_classes: Source dataset class mapping (list or dict)

        Returns:
            List of converted label lines
        """
        converted_lines = []

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class_id = int(parts[0])

                    # Get class name from source dataset
                    class_name = None
                    if isinstance(source_classes, list):
                        if 0 <= old_class_id < len(source_classes):
                            class_name = source_classes[old_class_id]
                    elif isinstance(source_classes, dict):
                        if old_class_id in source_classes:
                            class_name = source_classes[old_class_id]

                    if class_name:
                        # Find new class ID in unified classes
                        if class_name in self.unified_classes:
                            new_class_id = self.unified_classes.index(class_name)

                            # Replace class ID
                            parts[0] = str(new_class_id)
                            converted_lines.append(" ".join(parts) + "\n")
                        else:
                            logger.warning(
                                f"Class {class_name} not found in unified classes"
                            )
                    else:
                        logger.warning(
                            f"Class ID {old_class_id} not found in source classes"
                        )

        except Exception as e:
            logger.error(f"Error converting label {label_path}: {e}")

        return converted_lines

    def _copy_and_convert_data(
        self,
        pairs: List[Tuple[Path, Path]],
        output_split_dir: Path,
        source_classes: Dict[int, str],
        dataset_name: str,
    ):
        """
        Copy images and convert labels to output directory.

        Args:
            pairs: List of (image_path, label_path) tuples
            output_split_dir: Output directory for this split
            source_classes: Source dataset class mapping
            dataset_name: Name of source dataset for unique naming
        """
        images_dir = output_split_dir / "images"
        labels_dir = output_split_dir / "labels"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for i, (img_path, label_path) in enumerate(
            tqdm(pairs, desc=f"Processing {dataset_name}")
        ):
            # Create unique filename
            unique_name = f"{dataset_name}_{i:06d}"

            # Copy image
            output_img_path = images_dir / f"{unique_name}{img_path.suffix}"
            shutil.copy2(img_path, output_img_path)

            # Convert and copy label
            converted_lines = self._convert_label(label_path, source_classes)
            output_label_path = labels_dir / f"{unique_name}.txt"

            with open(output_label_path, "w") as f:
                f.writelines(converted_lines)

    def unify_datasets(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        shuffle: bool = True,
    ) -> bool:
        """
        Unify all added datasets into a single dataset.

        Args:
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            shuffle: Whether to shuffle the data

        Returns:
            True if successful, False otherwise
        """
        if not self.datasets:
            logger.error("No datasets added")
            return False

        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            logger.error("Train, val, and test ratios must sum to 1.0")
            return False

        logger.info(f"Unifying {len(self.datasets)} datasets...")
        logger.info(f"Unified classes: {self.unified_classes}")

        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all data pairs from all datasets
        all_pairs = []

        for dataset_info in self.datasets:
            dataset_name = dataset_info["name"]
            source_classes = dataset_info["config"]["names"]
            weight = dataset_info["weight"]

            # Collect pairs from all splits
            for split in ["train", "val", "test"]:
                pairs = self._get_image_label_pairs(dataset_info, split)

                # Apply weight by duplicating pairs
                weighted_pairs = pairs * int(weight)

                # Add dataset info to each pair
                for img_path, label_path in weighted_pairs:
                    all_pairs.append(
                        (img_path, label_path, source_classes, dataset_name)
                    )

                logger.info(
                    f"Dataset {dataset_name} - {split}: {len(pairs)} samples (weight: {weight})"
                )

        if not all_pairs:
            logger.error("No valid image-label pairs found")
            return False

        # Shuffle if requested
        if shuffle:
            random.shuffle(all_pairs)

        # Split data
        total_samples = len(all_pairs)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        train_pairs = all_pairs[:train_end]
        val_pairs = all_pairs[train_end:val_end]
        test_pairs = all_pairs[val_end:]

        logger.info(
            f"Split sizes - Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}"
        )

        # Process each split
        splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

        for split_name, pairs in splits.items():
            if not pairs:
                continue

            output_split_dir = self.output_dir / split_name

            # Group pairs by dataset for processing
            dataset_pairs = {}
            for img_path, label_path, source_classes, dataset_name in pairs:
                if dataset_name not in dataset_pairs:
                    dataset_pairs[dataset_name] = {
                        "pairs": [],
                        "classes": source_classes,
                    }
                dataset_pairs[dataset_name]["pairs"].append((img_path, label_path))

            # Process each dataset's contribution to this split
            for dataset_name, data in dataset_pairs.items():
                self._copy_and_convert_data(
                    data["pairs"],
                    output_split_dir,
                    data["classes"],
                    f"{split_name}_{dataset_name}",
                )

        # Create unified data.yaml
        self._create_unified_config()

        logger.info(f"✅ Dataset unification completed: {self.output_dir}")
        return True

    def _create_unified_config(self):
        """Create unified data.yaml configuration file."""
        config = {
            "path": str(self.output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": {i: name for i, name in enumerate(self.unified_classes)},
            "nc": len(self.unified_classes),
            # Metadata
            "unified_from": [ds["name"] for ds in self.datasets],
            "source_datasets": len(self.datasets),
            "description": "Unified dataset for helmet detection",
            "created_by": "HSE Vision Dataset Unifier",
        }

        config_path = self.output_dir / "data.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created unified config: {config_path}")

    def get_statistics(self) -> Dict:
        """Get statistics about the unified dataset."""
        if not self.output_dir.exists():
            return {}

        stats = {
            "total_classes": len(self.unified_classes),
            "classes": self.unified_classes,
            "splits": {},
        }

        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split / "images"
            if split_dir.exists():
                image_count = len(list(split_dir.glob("*")))
                stats["splits"][split] = image_count

        return stats

    def _discover_datasets(self, data_dir: Path) -> List[Dict]:
        """Auto-discover datasets in the data directory"""
        datasets = []

        for dataset_path in data_dir.iterdir():
            if not dataset_path.is_dir():
                continue

            # Look for data.yaml file
            data_yaml_path = dataset_path / "data.yaml"
            if not data_yaml_path.exists():
                continue

            try:
                with open(data_yaml_path, "r") as f:
                    config = yaml.safe_load(f)

                # Fix paths - they should point to directories within the dataset
                # The YAML files have paths like "../train/images" but the actual structure is "train/images"
                for split in ["train", "val", "test"]:
                    if split in config:
                        # Always use the dataset-relative path structure
                        # Check if the standard structure exists: dataset/split/images
                        images_path = dataset_path / split / "images"
                        if images_path.exists():
                            config[split] = str(images_path)
                        else:
                            # Fallback: try just the split directory
                            split_path = dataset_path / split
                            if split_path.exists():
                                config[split] = str(split_path)
                            else:
                                logger.warning(
                                    f"Could not find {split} directory in {dataset_path}"
                                )
                                continue

                datasets.append(
                    {"name": dataset_path.name, "path": dataset_path, "config": config}
                )

                print(f"Discovered dataset: {dataset_path.name}")

            except Exception as e:
                print(f"Warning: Could not load dataset {dataset_path.name}: {e}")
                continue

        return datasets


def main():
    """Main function for dataset unification."""
    import argparse

    parser = argparse.ArgumentParser(description="Unify multiple YOLO datasets")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Data",
        help="Directory containing dataset folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="unified_dataset",
        help="Output directory for unified dataset",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training data ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation data ratio"
    )
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test data ratio")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle data before splitting",
    )

    args = parser.parse_args()

    # Initialize unifier
    unifier = DatasetUnifier(args.output_dir)

    # Auto-discover datasets using the improved method
    data_dir = Path(args.data_dir)
    if data_dir.exists():
        discovered_datasets = unifier._discover_datasets(data_dir)
        for dataset_info in discovered_datasets:
            # Add weight to dataset info
            dataset_info["weight"] = 1.0
            unifier.add_dataset_info(dataset_info)
    else:
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Unify datasets
    success = unifier.unify_datasets(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        shuffle=args.shuffle,
    )

    if success:
        # Print statistics
        stats = unifier.get_statistics()
        print("\n" + "=" * 50)
        print("UNIFIED DATASET STATISTICS")
        print("=" * 50)
        print(f"Classes: {stats['total_classes']}")
        print(f"Class names: {', '.join(stats['classes'])}")

        for split, count in stats["splits"].items():
            print(f"{split.capitalize()}: {count} images")

        print(f"\nUnified dataset saved to: {args.output_dir}")
        print("=" * 50)
    else:
        print("❌ Dataset unification failed")


if __name__ == "__main__":
    main()
