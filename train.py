import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

torch.backends.cudnn.benchmark = True


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    """Training function for one epoch"""
    loop = tqdm(loader, leave=True)
    
    d_loss_epoch = 0
    g_loss_epoch = 0

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.amp.autocast(device_type=config.DEVICE):
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.amp.autocast(device_type=config.DEVICE):
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
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
    # Create necessary directories
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs(config.TRAIN_DIR, exist_ok=True)
    os.makedirs(config.VAL_DIR, exist_ok=True)
    
    # Initialize models
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    
    # Initialize optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    
    # Loss functions
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # Load checkpoints if specified
    if config.LOAD_MODEL and epoch % 1 == 0:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    # Create datasets and dataloaders
    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Gradient scalers for mixed precision (only on CUDA)
    if config.DEVICE == "cuda":
        g_scaler = torch.amp.GradScaler('cuda')
        d_scaler = torch.amp.GradScaler('cuda')
    else:
        g_scaler = torch.amp.GradScaler('cpu')
        d_scaler = torch.amp.GradScaler('cpu')

    print(f"Starting training on {config.DEVICE}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch}/{config.NUM_EPOCHS}]")
        
        d_loss, g_loss = train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler
        )
        
        print(f"D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")

        # Save checkpoints
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        # Save example predictions
        save_some_examples(gen, val_loader, epoch, folder="evaluation")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()