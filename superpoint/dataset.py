import numpy as np
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

category = ['fan', 'frying_pan', 'hairdry', 'mouse', 'running']

class KiprisDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        img_width = 240,
        img_height = 320,
        C = 5,
        transform=None
    ):
        self.annotation = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.img_width = img_width
        self.img_height = img_height
        self.C = C

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        return torch.LongTensor([index])