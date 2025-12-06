"""
Inference script to generate predictions using trained model
"""
import torch
import config
import os
from PIL import Image
import numpy as np
from generator_model import Generator
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_generator(checkpoint_path, device):
    """Load trained generator model"""
    gen = Generator(in_channels=3, features=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()
    print(f"✓ Loaded generator from {checkpoint_path}")
    return gen


def preprocess_image(image_path):
    """Preprocess input image"""
    image = np.array(Image.open(image_path))
    
    # If image is already concatenated (1200px wide), split it
    if image.shape[1] == 1200:
        image = image[:, :600, :]  # Take only input part
    
    # Resize and normalize
    transform = A.Compose([
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def generate_prediction(gen, input_path, output_path, device):
    """Generate and save prediction"""
    # Preprocess input
    input_tensor = preprocess_image(input_path).to(device)
    
    # Generate prediction
    with torch.no_grad():
        prediction = gen(input_tensor)
        prediction = prediction * 0.5 + 0.5  # Denormalize
    
    # Save output
    save_image(prediction, output_path)
    print(f"✓ Saved prediction to {output_path}")


def batch_inference(gen, input_dir, output_dir, device):
    """Process all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nProcessing {len(image_files)} images...")
    
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"pred_{img_file}")
        
        try:
            generate_prediction(gen, input_path, output_path, device)
        except Exception as e:
            print(f"✗ Error processing {img_file}: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate predictions using trained Pix2Pix model")
    parser.add_argument("--checkpoint", type=str, default="gen.pth.tar", 
                        help="Path to generator checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image or directory")
    parser.add_argument("--output", type=str, default="predictions",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"✗ Checkpoint not found: {args.checkpoint}")
        print("Please train the model first or specify correct checkpoint path")
        return
    
    # Load generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = load_generator(args.checkpoint, device)
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, "prediction.png")
        generate_prediction(gen, args.input, output_path, device)
    elif os.path.isdir(args.input):
        # Directory of images
        batch_inference(gen, args.input, args.output, device)
    else:
        print(f"✗ Input not found: {args.input}")


if __name__ == "__main__":
    main()