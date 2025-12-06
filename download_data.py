"""
Script to download the maps dataset for Pix2Pix training
"""
import os
import urllib.request
import tarfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_maps_dataset():
    """Download and extract maps dataset"""
    print("=== Downloading Maps Dataset ===\n")
    
    # Dataset URL (Pix2Pix maps dataset)
    url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz"
    output_file = "maps.tar.gz"
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Download
    print("Downloading dataset...")
    try:
        download_url(url, output_file)
        print(f"✓ Downloaded to {output_file}")
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return
    
    # Extract
    print("\nExtracting dataset...")
    try:
        with tarfile.open(output_file, 'r:gz') as tar:
            tar.extractall('data')
        print("✓ Extracted to data/maps/")
        
        # Remove tar file
        os.remove(output_file)
        print(f"✓ Removed {output_file}")
        
        # Count files
        train_count = len(os.listdir("data/maps/train"))
        val_count = len(os.listdir("data/maps/val"))
        
        print(f"\n=== Dataset Ready ===")
        print(f"Training images: {train_count}")
        print(f"Validation images: {val_count}")
        print("\nYou can now run: python train.py")
        
    except Exception as e:
        print(f"✗ Error extracting: {e}")


def download_facades_dataset():
    """Download and extract facades dataset"""
    print("=== Downloading Facades Dataset ===\n")
    
    url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
    output_file = "facades.tar.gz"
    
    os.makedirs("data", exist_ok=True)
    
    print("Downloading dataset...")
    try:
        download_url(url, output_file)
        print(f"✓ Downloaded to {output_file}")
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return
    
    print("\nExtracting dataset...")
    try:
        with tarfile.open(output_file, 'r:gz') as tar:
            tar.extractall('data')
        print("✓ Extracted to data/facades/")
        
        os.remove(output_file)
        print(f"✓ Removed {output_file}")
        
        train_count = len(os.listdir("data/facades/train"))
        val_count = len(os.listdir("data/facades/val"))
        
        print(f"\n=== Dataset Ready ===")
        print(f"Training images: {train_count}")
        print(f"Validation images: {val_count}")
        
        print("\n⚠ To use this dataset, update config.py:")
        print('TRAIN_DIR = "data/facades/train"')
        print('VAL_DIR = "data/facades/val"')
        
    except Exception as e:
        print(f"✗ Error extracting: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Pix2Pix datasets")
    parser.add_argument("--dataset", type=str, default="maps", 
                        choices=["maps", "facades"],
                        help="Dataset to download (maps or facades)")
    
    args = parser.parse_args()
    
    if args.dataset == "maps":
        download_maps_dataset()
    elif args.dataset == "facades":
        download_facades_dataset()


if __name__ == "__main__":
    main()