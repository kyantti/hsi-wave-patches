import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

NUM_WORKERS = os.cpu_count() if os.cpu_count() is not None else 0

class WaveletDataset(Dataset):
    """Custom Dataset for CWT Morlet images."""

    def __init__(self, csv_file: str, transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_labels = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = str(self.img_labels.iloc[idx, 0])
        # Convert to RGB to ensure 3-channel images for ResNet compatibility
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

