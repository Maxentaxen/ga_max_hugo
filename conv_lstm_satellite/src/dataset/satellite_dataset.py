import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SatelliteDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.folder_paths = []
        # Walk through all subdirectories
        for dirpath, dirnames, filenames in os.walk(root_dir):
            image_files = sorted(glob.glob(os.path.join(dirpath, '*.png')))
            if len(image_files) == 4:
                self.folder_paths.append(dirpath)

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder = self.folder_paths[idx]
        image_files = sorted(glob.glob(os.path.join(folder, '*.png')))
        assert len(image_files) == 4, f"Expected 4 images per folder, got {len(image_files)} in {folder}"

        # Load first 3 images as input sequence
        input_seq = []
        for img_path in image_files[:3]:
            img = Image.open(img_path).convert('L')
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
            input_seq.append(img)
        input_seq = torch.stack(input_seq, dim=0)  # (seq_len=3, 1, H, W)

        # Load 4th image as target
        target_img = Image.open(image_files[3]).convert('L')
        target_img = np.array(target_img, dtype=np.float32) / 255.0
        target_img = torch.from_numpy(target_img).unsqueeze(0)  # (1, H, W)

        return input_seq, target_img

# Example usage
if __name__ == "__main__":
    dataset = SatelliteDataset(root_dir='path/to/satellite/root')
    print(f'Dataset size: {len(dataset)}')