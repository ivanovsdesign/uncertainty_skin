import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .dataset import CustomImageDataset, ClassDataset

from torchvision import transforms

class ISICDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataframe = pd.read_csv(config.annotations_path)
        self.dataframe['path'] = config.path
        self.dataframe['target_name'] = config.target_name
        self.transform = {
            'train': transforms.Compose(
                    [
                    transforms.RandomResizedCrop(config.img_size, scale=config.crop_scale, antialias=True),
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: x / 255.0),
                    #transforms.Normalize(mean, std)
                    ]),
            'test': transforms.Compose(
                    [
                    transforms.RandomResizedCrop(config.img_size ,scale=config.crop_scale_tta, antialias=True),
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: x / 255.0),
                    
                    #transforms.Normalize(mean, std)
                    ]),
            'test_tta':transforms.Compose(
                    [
                    transforms.RandomResizedCrop(config.img_size, scale=config.crop_scale_tta, antialias=True),
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: x / 255.0),
                    #transforms.Normalize(mean, std)
                    ])
                }  # Define your transforms here

    def setup(self, stage=None):
        if self.config.bagging:
            self.dataframe = self.dataframe.sample(n=self.config.bagging_size, replace=True, random_state=self.config.seed).reset_index()
        train_val_df, test_df = train_test_split(self.dataframe, test_size=self.config.test_size, random_state=self.config.seed)
        train_df, val_df = train_test_split(train_val_df, test_size=self.config.val_size / (self.config.train_size + self.config.val_size), random_state=self.config.seed)
        self.train_dataset = CustomImageDataset(train_df, self.config.path, self.transform['train'])
        self.val_dataset = CustomImageDataset(val_df, self.config.path, self.transform['test'])
        self.test_dataset = CustomImageDataset(test_df, self.config.path, self.transform['test'])
        self.test_tta_dataset = CustomImageDataset(test_df, self.config.path, self.transform['test_tta'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def test_dataloader(self):
        if self.config.num_tta > 1:
            return DataLoader(self.test_tta_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
        else:
            return DataLoader(self.test_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
        
class ChestDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataframe = pd.read_csv(config.train_path)
        self.val_dataframe = pd.read_csv(config.val_path)
        self.test_dataframe = pd.read_csv(config.test_path)
        
        self.image_path = config.image_path
        self.image_size = config.img_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            #v2.Lambda(lambda x: x / 255.0),  # Divide by 255
            transforms.Resize((config.img_size, config.img_size)),
            transforms.Grayscale(num_output_channels=3),
            #transforms.RandomResizedCrop(config.img_size, scale=(0.85, 1), antialias=True),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            #v2.Lambda(lambda x: x / 255.0),  # Divide by 255
            transforms.Resize((config.img_size, config.img_size)),
            transforms.Grayscale(num_output_channels=3)
        ])
        
        self.tta_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            #v2.Lambda(lambda x: x / 255.0),  # Divide by 255
            transforms.Resize((config.img_size, config.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(config.img_size, scale=(0.85, 1), antialias=True),
        ])

    def setup(self, stage=None):
        self.train_dataset = ClassDataset(self.train_dataframe, self.image_path, self.transform)
        self.val_dataset = ClassDataset(self.val_dataframe, self.image_path, self.transform)
        self.test_dataset = ClassDataset(self.test_dataframe, self.image_path, self.test_transform)
        self.test_tta_dataset = ClassDataset(self.test_dataframe, self.image_path, self.tta_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def test_dataloader(self):
        if self.config.num_tta > 1:
            return DataLoader(self.test_tta_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
        else:
            return DataLoader(self.test_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)