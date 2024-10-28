import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .dataset import CustomImageDataset

class ISICDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataframe = pd.read_csv(config.annotations_path, index_col=2)
        self.dataframe['path'] = config.path
        self.dataframe['target_name'] = config.target_name
        self.transform = None  # Define your transforms here

    def setup(self, stage=None):
        if self.config.bagging:
            self.dataframe = self.dataframe.sample(n=self.config.bagging_size, random_state=self.config.seed)
        train_val_df, test_df = train_test_split(self.dataframe, test_size=self.config.test_size, random_state=self.config.seed)
        train_df, val_df = train_test_split(train_val_df, test_size=self.config.val_size / (self.config.train_size + self.config.val_size), random_state=self.config.seed)
        self.train_dataset = CustomImageDataset(train_df, self.transform)
        self.val_dataset = CustomImageDataset(val_df, self.transform)
        self.test_dataset = CustomImageDataset(test_df, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)