import torch
import torchvision
import matplotlib.pyplot as plt

import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

def extract_features(model, dataloader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            embeddings = model.intermediate_forward(x)
            features.append(embeddings.cpu().numpy())
            labels.append(y.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels