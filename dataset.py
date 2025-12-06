import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir,max_samples=100):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)[:max_samples]
        print(f"Found {len(self.list_files)} images in {root_dir}")

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        
        # Split the concatenated image (input | target)
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        # Apply transformations to both images
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        # Apply specific transformations
        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("data/maps/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys
        sys.exit()