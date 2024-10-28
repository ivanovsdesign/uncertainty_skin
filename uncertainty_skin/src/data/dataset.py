from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.index[idx].rsplit('/', 1)[-1]
        image = Image.open(f'{self.dataframe.path}/{image_path}').convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(self.dataframe.loc[self.dataframe.index[idx], self.dataframe.target_name])