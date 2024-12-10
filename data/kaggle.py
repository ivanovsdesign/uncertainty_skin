import kagglehub
import shutil as sh
import os

# Download latest version
path = kagglehub.dataset_download("olegopoly/isic-uncertain-balanced/versions/2")

if os.path.exists('./data/isic_uncertain_imbalanced'):
    print('Dataset was already downloaded')
else:
    new_path = sh.copytree(path, './data/isic_uncertain_imbalanced')
    print("Path to dataset files:", new_path)