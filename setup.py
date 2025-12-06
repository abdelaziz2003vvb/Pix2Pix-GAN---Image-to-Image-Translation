"""
Setup script to create necessary directories and verify the environment
"""
import os
import torch


def create_directories():
    """Create all necessary directories for the project"""
    directories = [
        "data/maps/train",
        "data/maps/val",
        "evaluation",
        "checkpoints",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")


def check_environment():
    """Check PyTorch and CUDA availability"""
    print("\n=== Environment Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("⚠ CUDA not available. Training will use CPU (much slower)")


def verify_dataset():
    """Verify dataset structure"""
    print("\n=== Dataset Verification ===")
    train_dir = "data/maps/train"
    val_dir = "data/maps/val"
    
    train_files = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
    val_files = len(os.listdir(val_dir)) if os.path.exists(val_dir) else 0
    
    print(f"Training images: {train_files}")
    print(f"Validation images: {val_files}")
    
    if train_files == 0:
        print("\n⚠ WARNING: No training images found!")
        print("Please add training images to: data/maps/train/")
        print("Images should be in format: [input_image | target_image] concatenated horizontally")
        print("Expected size: 1200px wide (600px input + 600px target)")
    
    if val_files == 0:
        print("\n⚠ WARNING: No validation images found!")
        print("Please add validation images to: data/maps/val/")


def main():
    print("=== Pix2Pix Project Setup ===\n")
    
    # Create directories
    create_directories()
    
    # Check environment
    check_environment()
    
    # Verify dataset
    verify_dataset()
    
    print("\n=== Setup Complete ===")
    print("\nNext steps:")
    print("1. Add your training images to: data/maps/train/")
    print("2. Add your validation images to: data/maps/val/")
    print("3. Run: python train.py")


if __name__ == "__main__":
    main()