import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SatelliteDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = glob.glob(os.path.join(image_dir, '*.png'))  # Adjust the extension as needed

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(np.array(image)).unsqueeze(0)

        return image

# Example of how to use the dataset
if __name__ == "__main__":
    dataset = SatelliteDataset(image_dir='path/to/satellite/images')
    print(f'Dataset size: {len(dataset)}')