from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, images_path, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.images_path = images_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.index[idx].rsplit('/', 1)[-1]
        image = Image.open(f'{self.images_path}/{image_path}').convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(self.dataframe.loc[self.dataframe.index[idx], 'target'])