import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from src.data.datamodule import ISICDataModule
from src.models.cnn_model import CNN
from src.models.timm_model import TimmModel
from src.utils.clearml_logger import ClearMLLogger
from src.utils.utils import set_seed
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

import os

from clearml import InputModel

@hydra.main(config_path="/repo/uncertainty_skin/uncertainty_skin/configs", config_name="config")
def test(config: DictConfig):
    set_seed(config.dataset.seed)
    
    data_module = ISICDataModule(config.dataset)
    data_module.setup()
 
    try:
        if config.model.name == 'CNN':
            model = CNN.load_from_checkpoint(config.model.checkpoint_path, config=config.model).to(config.device)
        elif config.model.name.startswith('timm'):
            model = TimmModel.load_from_checkpoint(config.model.checkpoint_path, config=config.model).to(config.device)
        else:
            raise ValueError(f"Unknown model: {config.model.name}")
    except: 
        print(f'You should provide checkpoint path')

    logger = ClearMLLogger(project_name="ISIC_2024", task_name=f"{config.model.name}_testing")
    trainer = pl.Trainer(logger=logger)

    # Perform TTA
    num_tta = config.dataset.num_tta
    test_predictions_tta = []
    test_labels_tta = []
    test_certainties_s_tta = []
    test_confidences_s_tta = []
    for _ in range(num_tta):
        test_predictions, test_labels, test_certainties_s, test_confidences_s = [], [], [], []
        for batch in data_module.test_dataloader():
            inputs, labels = batch
            inputs = inputs.to(model.device)  # Move inputs to the same device as the model
            outputs = model(inputs)
            preds = torch.softmax(outputs[:, :config.model.num_classes], dim=1)
            certainties_s = torch.exp(-outputs[:, config.model.num_classes:])
            confidences_s = preds.max(dim=1).values
            test_predictions.append(preds)
            test_labels.append(labels)
            test_certainties_s.append(certainties_s)
            test_confidences_s.append(confidences_s)

        test_predictions = torch.cat(test_predictions)
        test_labels = torch.cat(test_labels)
        test_certainties_s = torch.cat(test_certainties_s)
        test_confidences_s = torch.cat(test_confidences_s)

        test_predictions_tta.append(test_predictions)
        test_labels_tta.append(test_labels)
        test_certainties_s_tta.append(test_certainties_s)
        test_confidences_s_tta.append(test_confidences_s)

    # Collect and log metrics
    test_predictions_tta = torch.stack(test_predictions_tta).mean(dim=0)
    test_labels_tta = torch.stack(test_labels_tta).mode(dim=0).values
    test_certainties_s_tta = torch.stack(test_certainties_s_tta).mean(dim=0)
    test_confidences_s_tta = torch.stack(test_confidences_s_tta).mean(dim=0)

    test_predictions_tta = test_predictions_tta.argmax(dim=1).cpu()  # Move to CPU
    test_labels_tta = test_labels_tta.squeeze().cpu()  # Move to CPU

    accuracy = accuracy_score(test_labels_tta, test_predictions_tta)
    f1 = f1_score(test_labels_tta, test_predictions_tta, average='weighted')
    precision = precision_score(test_labels_tta, test_predictions_tta, average='weighted')
    recall = recall_score(test_labels_tta, test_predictions_tta, average='weighted')
    roc_auc = roc_auc_score(test_labels_tta, test_predictions_tta, average='weighted')

    logger.log_metrics({
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    })

    # Save predictions
    predictions_df = pd.DataFrame({
        'true_labels': test_labels_tta.tolist(),
        'predictions': test_predictions_tta.tolist(),
        'certainties_s': test_certainties_s_tta.cpu().tolist(),  # Move to CPU
        'confidences_s': test_confidences_s_tta.cpu().tolist()  # Move to CPU
    })
    predictions_df.to_csv(f"{config.model.name}_predictions.csv", index=False)

    # Plot confusion matrix
    cm = confusion_matrix(test_labels_tta, test_predictions_tta)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=config.dataset.class_names).plot()
    plt.savefig(f"{config.model.name}_confusion_matrix.png")

if __name__ == "__main__":
    test()