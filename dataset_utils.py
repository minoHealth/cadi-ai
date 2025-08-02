#!/usr/bin/env python3
"""
CADI AI Dataset Utilities
========================
Helper functions for dataset management, validation, and preparation.
"""

import os
import yaml
import shutil
from pathlib import Path
from collections import defaultdict


def create_data_yaml(dataset_path: str, output_path: str = "data.yaml", 
                    class_names: list = None) -> str:
    """
    Create a data.yaml file for YOLO training.
    
    Args:
        dataset_path: Path to dataset directory containing train/val/test folders
        output_path: Where to save the data.yaml file
        class_names: List of class names (if None, will try to detect from dataset)
    
    Returns:
        Path to created data.yaml file
    """
    
    # Default class names for CADI AI
    if class_names is None:
        class_names = ['abiotic', 'disease', 'insect']
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Build data configuration
    data_config = {
        'train': f'{dataset_path}/train/images',
        'val': f'{dataset_path}/valid/images',
        'test': f'{dataset_path}/test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    # Verify paths exist
    missing_paths = []
    for split, path in [('train', data_config['train']), ('val', data_config['val'])]:
        if not os.path.exists(path):
            missing_paths.append(f"{split}: {path}")
    
    if missing_paths:
        print("⚠️  Warning: Some paths don't exist:")
        for missing in missing_paths:
            print(f"  - {missing}")
    
    # Save data.yaml
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml at: {output_path}")
    return output_path


def validate_dataset(data_yaml_path: str) -> dict:
    """
    Validate YOLO dataset and return statistics.
    
    Args:
        data_yaml_path: Path to data.yaml file
        
    Returns:
        Dictionary with dataset statistics
    """
    
    # Load data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    stats = {
        'splits': {},
        'classes': data_config.get('names', []),
        'class_counts': defaultdict(int),
        'total_images': 0,
        'total_labels': 0
    }
    
    print("📊 Dataset Validation Report")
    print("=" * 40)
    
    # Check each split
    for split in ['train', 'val', 'test']:
        if split in data_config:
            image_path = data_config[split]
            label_path = image_path.replace('/images', '/labels')
            
            # Count images and labels
            if os.path.exists(image_path):
                images = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                image_count = len(images)
            else:
                image_count = 0
                images = []
            
            if os.path.exists(label_path):
                labels = [f for f in os.listdir(label_path) if f.endswith('.txt')]
                label_count = len(labels)
            else:
                label_count = 0
                labels = []
            
            stats['splits'][split] = {
                'images': image_count,
                'labels': label_count,
                'image_path': image_path,
                'label_path': label_path
            }
            
            stats['total_images'] += image_count
            stats['total_labels'] += label_count
            
            # Print split info
            status = "✅" if image_count > 0 and label_count > 0 else "❌"
            print(f"{status} {split:5s}: {image_count:4d} images, {label_count:4d} labels")
            
            # Count classes in this split
            if os.path.exists(label_path):
                for label_file in labels:
                    try:
                        with open(os.path.join(label_path, label_file), 'r') as f:
                            for line in f:
                                if line.strip():
                                    class_id = int(line.strip().split()[0])
                                    if class_id < len(stats['classes']):
                                        class_name = stats['classes'][class_id]
                                        stats['class_counts'][class_name] += 1
                    except Exception as e:
                        print(f"⚠️  Warning: Error reading {label_file}: {e}")
    
    print("\n📈 Class Distribution:")
    total_objects = sum(stats['class_counts'].values())
    for class_name in stats['classes']:
        count = stats['class_counts'][class_name]
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  {class_name:10s}: {count:5d} objects ({percentage:5.1f}%)")
    
    print(f"\n📋 Summary:")
    print(f"  Total Images: {stats['total_images']}")
    print(f"  Total Labels: {stats['total_labels']}")
    print(f"  Total Objects: {total_objects}")
    print(f"  Classes: {len(stats['classes'])}")
    
    return stats


def setup_kaggle_dataset(kaggle_dataset_path: str, working_dir: str) -> str:
    """
    Set up dataset for Kaggle environment.
    
    Args:
        kaggle_dataset_path: Path to dataset in /kaggle/input/
        working_dir: Working directory path
        
    Returns:
        Path to created data.yaml file
    """
    
    print("🔧 Setting up Kaggle dataset...")
    
    # Create working directory
    os.makedirs(working_dir, exist_ok=True)
    
    # Create data.yaml pointing to Kaggle input paths
    data_yaml_path = create_data_yaml(
        dataset_path=kaggle_dataset_path,
        output_path=f"{working_dir}/data.yaml"
    )
    
    print(f"✅ Kaggle setup complete. Use: {data_yaml_path}")
    return data_yaml_path


def copy_best_weights(source_dir: str, destination: str = "best_model.pt"):
    """
    Copy the best model weights from training results.
    
    Args:
        source_dir: Directory containing training results
        destination: Where to copy the best weights
    """
    
    # Look for best.pt in the results directory
    best_weights_path = None
    
    # Search common locations
    possible_paths = [
        f"{source_dir}/weights/best.pt",
        f"{source_dir}/best.pt",
    ]
    
    # Also search subdirectories
    if os.path.exists(source_dir):
        for root, dirs, files in os.walk(source_dir):
            if "best.pt" in files:
                possible_paths.append(os.path.join(root, "best.pt"))
    
    # Find the first existing path
    for path in possible_paths:
        if os.path.exists(path):
            best_weights_path = path
            break
    
    if best_weights_path:
        shutil.copy2(best_weights_path, destination)
        print(f"✅ Copied best weights: {best_weights_path} → {destination}")
        return destination
    else:
        print("❌ Best weights not found in training results")
        return None


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='CADI AI Dataset Utilities')
    parser.add_argument('--create-yaml', help='Create data.yaml for dataset path')
    parser.add_argument('--output-path', help='Output path for the generated data.yaml', default='data.yaml')
    parser.add_argument('--validate', help='Validate dataset from data.yaml')
    parser.add_argument('--setup-kaggle', nargs=2, metavar=('DATASET_PATH', 'WORKING_DIR'),
                        help='Setup Kaggle dataset')

    args = parser.parse_args()

    if args.create_yaml:
        create_data_yaml(args.create_yaml, args.output_path)
    elif args.validate:
        validate_dataset(args.validate)
    elif args.setup_kaggle:
        setup_kaggle_dataset(args.setup_kaggle[0], args.setup_kaggle[1])
    else:
        print("Please specify an action. Use --help for options.")
