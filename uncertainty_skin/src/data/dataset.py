from torch.utils.data import Dataset
from PIL import Image
import cv2
import torch
import numpy as np

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
    def __init__(self, dataframe, images_path, image_size, preload_transform=None, transform=None):
        self.dataframe = dataframe.dropna().reset_index()
        self.preload_transform = preload_transform
        self.transform = transform

        self.images = torch.zeros(len(dataframe), 3, image_size, image_size).float()
        
        for idx in range(len(self.dataframe)):
            image_path = self.dataframe.loc[idx, 'index']
            image = cv2.imread(f'{images_path}/{image_path}', cv2.IMREAD_GRAYSCALE)
            if self.preload_transform:
                self.images[idx] = self.preload_transform(torch.tensor(image.astype(np.float32) / 255).unsqueeze(0))

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.images[idx])
        else:
            image = self.images[idx]
        return image, int(self.dataframe.loc[idx, 'CTI_label'])