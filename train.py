#!/usr/bin/env python3
"""
CADI AI Model Training Script
=============================
Clean, production-ready training script for YOLO object detection.
Usage: python train.py --config config.yaml
"""

import os
import gc
import sys
import yaml
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO


class ModelTrainer:
    """Main training class with intelligent batch size detection and robust error handling."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = self.load_config(config_path)
        self.setup_directories()
        
    def load_config(self, config_path: str) -> dict:
        """Load and validate configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required fields
        required = ['model', 'data', 'epochs', 'imgsz']
        missing = [field for field in required if field not in config]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
            
        return config
    
    def setup_directories(self):
        """Create necessary output directories."""
        os.makedirs(self.config.get('output_dir', 'runs'), exist_ok=True)
        
    def find_optimal_batch_size(self, max_batch: int = 32, min_batch: int = 2) -> int:
        """
        Dynamically find the largest batch size that fits in memory.
        
        Args:
            max_batch: Starting batch size to try
            min_batch: Minimum batch size to attempt
            
        Returns:
            Optimal batch size that works
        """
        print("🔍 Finding optimal batch size...")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        batch = max_batch
        while batch >= min_batch:
            try:
                print(f"  Testing batch size: {batch}")
                
                # Create model instance
                model = YOLO(self.config['model'])
                
                # Try short training run
                results = model.train(
                    data=self.config['data'],
                    imgsz=self.config['imgsz'],
                    batch=batch,
                    epochs=1,
                    patience=0,
                    verbose=False,
                    plots=False,
                    save=False
                )
                
                print(f"✅ Batch size {batch} works!")
                
                # Cleanup
                del model, results
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                return batch
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "cuda" in error_msg:
                    print(f"  ❌ Batch {batch} failed (OOM)")
                    
                    # Cleanup
                    if 'model' in locals():
                        del model
                    if 'results' in locals():
                        del results
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Try smaller batch
                    batch = max(batch // 2, min_batch)
                    
                else:
                    print(f"❌ Unexpected error: {e}")
                    raise
                    
            except Exception as e:
                print(f"❌ Training test failed: {e}")
                raise
        
        raise RuntimeError(f"No suitable batch size found (tried down to {min_batch})")
    
    def train(self):
        """Main training function."""
        try:
            print("🚀 Starting CADI AI Model Training")
            print("=" * 50)
            
            # Print system info
            self.print_system_info()
            
            # Validate data paths
            self.validate_data_paths()
            
            # Find optimal batch size if not specified
            if 'batch' not in self.config or self.config['batch'] == 'auto':
                optimal_batch = self.find_optimal_batch_size()
                self.config['batch'] = optimal_batch
                print(f"🎯 Using optimal batch size: {optimal_batch}")
            else:
                print(f"📦 Using configured batch size: {self.config['batch']}")
            
            # Initialize model
            print(f"🤖 Loading model: {self.config['model']}")
            model = YOLO(self.config['model'])
            
            # Start training
            print("🏋️ Starting training...")
            results = model.train(
                data=self.config['data'],
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch'],
                device=self.config.get('device', 0),
                workers=self.config.get('workers', 8),
                patience=self.config.get('patience', 100),
                save=self.config.get('save', True),
                plots=self.config.get('plots', True),
                val=self.config.get('val', True),
                project=self.config.get('output_dir', 'runs'),
                name=self.config.get('name', 'train'),
                exist_ok=True
            )
            
            print("✅ Training completed successfully!")
            print(f"📊 Results saved to: {results.save_dir}")
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def print_system_info(self):
        """Print system and environment information."""
        print("💻 System Information:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU Memory: {memory_gb:.1f} GB")
        
        print()
    
    def validate_data_paths(self):
        """Validate that all data paths exist."""
        print("📁 Validating data paths...")
        
        data_file = self.config['data']
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load and check data.yaml
        with open(data_file, 'r') as f:
            data_config = yaml.safe_load(f)
        
        for split in ['train', 'val']:
            if split in data_config:
                path = data_config[split]
                if os.path.exists(path):
                    image_count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  ✅ {split}: {image_count} images")
                else:
                    raise FileNotFoundError(f"{split} path not found: {path}")
        
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train CADI AI Object Detection Model')
    parser.add_argument('--config', '-c', required=True, help='Path to config YAML file')
    parser.add_argument('--batch', '-b', type=int, help='Override batch size from config')
    parser.add_argument('--epochs', '-e', type=int, help='Override epochs from config')
    parser.add_argument('--find-batch', action='store_true', help='Find optimal batch size and exit')
    parser.add_argument('--validate-paths-only', action='store_true', help='Validate data paths and exit')

    args = parser.parse_args()

    try:
        # Initialize trainer
        trainer = ModelTrainer(args.config)

        # Override config with command line args
        if args.batch:
            trainer.config['batch'] = args.batch
        if args.epochs:
            trainer.config['epochs'] = args.epochs

        # Handle action flags
        if args.validate_paths_only:
            print("Running path validation only...")
            trainer.validate_data_paths()
            print("✅ Path validation successful.")
            sys.exit(0)

        if args.find_batch:
            print("Finding optimal batch size...")
            optimal_batch = trainer.find_optimal_batch_size()
            print(f"✅ Optimal batch size found: {optimal_batch}")
            # Optionally, update config on disk
            # with open(args.config, 'w') as f:
            #     yaml.dump(trainer.config, f)
            sys.exit(0)
            
        # Start training
        results = trainer.train()
        
        print("🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"💥 Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

