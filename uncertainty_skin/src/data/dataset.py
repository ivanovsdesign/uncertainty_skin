from torch.utils.data import Dataset
from PIL import Image

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