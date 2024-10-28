import kagglehub
import shutil as sh
import os

# Download latest version
path = kagglehub.dataset_download("olegopoly/isic-balanced")

if os.path.exists('./data/isic_balanced'):
    print('Dataset was already downloaded')
else:
    new_path = sh.copytree(path, './data/isic_balanced')
    print("Path to dataset files:", new_path)