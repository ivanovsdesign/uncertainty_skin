from torch.utils.data import Dataset
from PIL import Image
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
from tqdm import tqdm

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, images_path, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.images_path = images_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.loc[idx, 'isic_id']
        image = Image.open(f'{self.images_path}/{image_path}.jpg').convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(self.dataframe.loc[self.dataframe.index[idx], 'target'])
    
class ClassDataset(Dataset):
    def __init__(self, dataframe, images_path, transform=None):
        self.dataframe = dataframe.dropna().reset_index()

        self.transform = transform

        self.images_path = images_path
    

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 1]
        image = Image.open(f'{self.images_path}/{image_path}').convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(self.dataframe.loc[idx, 'CTI_label'])