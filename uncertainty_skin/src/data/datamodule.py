import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data.dataset import CustomImageDataset, ClassDataset
import torch
from torch.utils.data import WeightedRandomSampler
from PIL import Image
import numpy as np

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
                    transforms.Resize((self.config.img_size, self.config.img_size), interpolation=Image.BILINEAR),
                    transforms.ToTensor(),
                    ]),
            'test': transforms.Compose(
                    [
                    #transforms.RandomResizedCrop(config.img_size ,scale=config.crop_scale_tta, antialias=True),
                    transforms.Resize((self.config.img_size, self.config.img_size), interpolation=Image.BILINEAR),
                    transforms.ToTensor(),
                    ]),
            'test_tta':transforms.Compose(
                    [
                    transforms.RandomResizedCrop(config.img_size, scale=config.crop_scale_tta, antialias=True),
                    transforms.Resize((self.config.img_size, self.config.img_size), interpolation=Image.BILINEAR),
                    transforms.ToTensor(),
                    ])
                }  # Define your transforms here

    def add_label_noise(self, labels, noise_level):
        """
        Adds noise to the labels based on the noise_level.
        noise_level: Probability of flipping the label.
        """
        noisy_labels = labels.copy()
        for i in range(len(noisy_labels)):
            if np.random.rand() < noise_level:
                noisy_labels[i] = 1 - noisy_labels[i]  # Flip the label
        return noisy_labels

    def setup(self, stage=None):
        if self.config.bagging == True:
            self.dataframe = self.dataframe.sample(n=self.config.bagging_size, replace=True, random_state=self.config.seed, ignore_index=True).reset_index(drop=True)
            
        # Apply label noise to the training dataset
        if self.config.noise > 0:
            self.dataframe['target'] = self.add_label_noise(self.dataframe['target'].values, self.config.noise)
            
        train_val_df, test_df = train_test_split(self.dataframe, test_size=self.config.test_size, random_state=self.config.seed, stratify=self.dataframe['target'])
        train_df, val_df = train_test_split(train_val_df, test_size=self.config.val_size / (self.config.train_size + self.config.val_size), random_state=self.config.seed, stratify=train_val_df['target'])
        self.train_dataset = CustomImageDataset(train_df, self.config.path, self.transform['train'])
        self.val_dataset = CustomImageDataset(val_df, self.config.path, self.transform['test'])
        
        
        # Fixed test set
        if self.config.fixed == True:
            test_df = pd.read_csv('/repo/uncertainty_skin/data/isic_balanced/test.csv').reset_index(drop=True)
            test_df['path'] = self.config.path
            test_df['target_name'] = self.config.target_name        
        
        self.test_dataset = CustomImageDataset(test_df, self.config.path, self.transform['test'])
        self.test_tta_dataset = CustomImageDataset(test_df, self.config.path, self.transform['test_tta'])
        
        print(self.config.bagging_size)
        print(self.dataframe.shape)
        
        print(train_df['target'].value_counts())
        print(val_df['target'].value_counts())
        print(test_df['target'].value_counts())
        
        print(test_df.index)
        
        #print(f'Train idx: {train_df.reset_index(drop=True).index}')
        #print(f'Dataframe shape: {train_df.shape}')
        #print(f'Dataframe head: {train_df.head()}')
        #print(f'Dataframe head: {train_df.reset_index(drop=True).head()}')
        
        labels = [int(label) for _, label in zip(train_df.index, train_df.target)]
        class_sample_count = [labels.count(i) for i in [0,1]]
        weight = 1. / torch.tensor(class_sample_count, dtype=torch.float)
        samples_weight = torch.tensor([weight[t] for t in labels])
        self.train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        labels = [int(label) for _, label in zip(val_df.index, val_df.target)]
        class_sample_count = [labels.count(i) for i in [0,1]]
        weight = 1. / torch.tensor(class_sample_count, dtype=torch.float)
        samples_weight = torch.tensor([weight[t] for t in labels])
        self.val_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        labels = [int(label) for _, label in zip(test_df.index, test_df.target)]
        class_sample_count = [labels.count(i) for i in [0,1]]
        weight = 1. / torch.tensor(class_sample_count, dtype=torch.float)
        samples_weight = torch.tensor([weight[t] for t in labels])
        self.test_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          sampler=self.train_sampler,
                          pin_memory=True,
                          worker_init_fn=self.config.seed % 2**32,
                          generator=self.g)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          sampler=self.val_sampler,
                          pin_memory=True,
                          worker_init_fn=self.config.seed % 2**32,
                          generator=self.g)

    def test_dataloader(self):
        return DataLoader(self.test_tta_dataset,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers,
                            sampler=self.test_sampler,
                            pin_memory=True,
                            worker_init_fn=self.config.seed % 2**32,
                            generator=self.g)
    def tta_dataloader(self):
        return DataLoader(self.test_dataset,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers,
                            sampler=self.test_sampler,
                            pin_memory=True,
                            worker_init_fn=self.config.seed % 2**32,
                            generator=self.g)
        
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
        
        labels = [label for _, label in self.train_dataset]
        class_sample_count = [labels.count(i) for i in [0,1]]
        weight = 1. / torch.tensor(class_sample_count, dtype=torch.float)
        samples_weight = torch.tensor([weight[t] for t in labels])
        self.train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        labels = [label for _, label in self.val_dataset]
        class_sample_count = [labels.count(i) for i in [0,1]]
        weight = 1. / torch.tensor(class_sample_count, dtype=torch.float)
        samples_weight = torch.tensor([weight[t] for t in labels])
        self.val_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        labels = [label for _, label in self.test_dataset]
        class_sample_count = [labels.count(i) for i in [0,1]]
        weight = 1. / torch.tensor(class_sample_count, dtype=torch.float)
        samples_weight = torch.tensor([weight[t] for t in labels])
        self.test_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          sampler=self.train_sampler,
                          pin_memory=True,
                          worker_init_fn=self.config.seed % 2**32,
                          generator=self.g)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          sampler=self.val_sampler,
                          pin_memory=True,
                          worker_init_fn=self.config.seed % 2**32,
                          generator=self.g)

    def test_dataloader(self):
        if self.config.num_tta > 1:
            return DataLoader(self.test_tta_dataset,
                              batch_size=self.config.batch_size,
                              num_workers=self.config.num_workers,
                              sampler=self.test_sampler,
                              pin_memory=True,
                              worker_init_fn=self.config.seed % 2**32,
                              generator=self.g)
        else:
            return DataLoader(self.test_dataset,
                              batch_size=self.config.batch_size,
                              num_workers=self.config.num_workers,
                              sampler=self.test_sampler,
                              pin_memory=True,
                              worker_init_fn=self.config.seed % 2**32,
                              generator=self.g)