"""
CPU-Optimized training script for Pix2Pix
This version is faster on CPU but produces lower quality results
For best results, use train.py on GPU
"""
import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config_cpu as config  # Use CPU-optimized config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

torch.backends.cudnn.benchmark = True


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce):
    """Training function for one epoch (no mixed precision for CPU)"""
    loop = tqdm(loader, leave=True)
    
    d_loss_epoch = 0
    g_loss_epoch = 0

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator (no autocast for CPU)
        y_fake = gen(x)
        D_real = disc(x, y)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(x, y_fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train Generator
        D_fake = disc(x, y_fake)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()
        
        # Accumulate losses
        d_loss_epoch += D_loss.item()
        g_loss_epoch += G_loss.item()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                G_loss=G_loss.item(),
            )
    
    return d_loss_epoch / len(loader), g_loss_epoch / len(loader)


def main():
    print("\n" + "="*60)
    print("CPU-OPTIMIZED PIX2PIX TRAINING")
    print("="*60)
    print("⚠️  This configuration uses reduced settings for CPU training")
    print("    For best quality, use train.py on GPU (Colab/Kaggle)")
    print("="*60 + "\n")
    
    # Create necessary directories
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs(config.TRAIN_DIR, exist_ok=True)
    os.makedirs(config.VAL_DIR, exist_ok=True)
    
    # Initialize models with reduced features for faster CPU training
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=32).to(config.DEVICE)  # Reduced from 64
    
    # Initialize optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    
    # Loss functions
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # Load checkpoints if specified
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    # Create datasets and dataloaders
    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Starting training on {config.DEVICE}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Total epochs: {config.NUM_EPOCHS}\n")

    start_time = time.time()

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        
        d_loss, g_loss = train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE
        )
        
        epoch_time = time.time() - epoch_start
        print(f"D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}, Time: {epoch_time:.1f}s")

        # Save checkpoints more frequently on CPU
        if config.SAVE_MODEL and (epoch % 5 == 0 or epoch == config.NUM_EPOCHS - 1):
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        # Save example predictions
        if epoch % 5 == 0 or epoch == config.NUM_EPOCHS - 1:
            save_some_examples(gen, val_loader, epoch, folder="evaluation")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {total_time/60:.1f} minutes!")
    print(f"Average time per epoch: {total_time/config.NUM_EPOCHS:.1f}s")
    print(f"Checkpoints saved: {config.CHECKPOINT_GEN}, {config.CHECKPOINT_DISC}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()