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

@hydra.main(config_path="configs", config_name="config")
def test(config: DictConfig):
    set_seed(config.dataset.seed)
    data_module = ISICDataModule(config.dataset)
    if config.model.name == 'CNN':
        model = CNN.load_from_checkpoint(config.model.checkpoint_path, config=config.model)
    elif config.model.name.startswith('timm'):
        model = TimmModel.load_from_checkpoint(config.model.checkpoint_path, config=config.model)
    else:
        raise ValueError(f"Unknown model: {config.model.name}")

    logger = ClearMLLogger(project_name="ISIC_2024", task_name=f"{config.model.name}_testing")
    trainer = pl.Trainer(**config.trainer, logger=logger)

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

    # Calculate TTA metrics
    test_attr_tta = test_vis_tta(model, data_module.test_dataloader(), config.model.thresholds, figs=5, numTTA=config.dataset.num_tta)
    mode_labels = torch.mode(test_attr_tta['labels_tta'], dim=0).values
    mode_predictions = torch.mode(test_attr_tta['predictions_tta'], dim=0).values
    mode_confidences_s_tta = torch.mode(test_attr_tta['confidences_s_tta'], dim=0).values
    mode_certainties_s_tta = torch.mode(test_attr_tta['certainties_s_tta'], dim=0).values

    true_pred = (mode_labels == mode_predictions).tolist()
    df = pd.DataFrame({
        'Mode_label_tta': mode_labels,
        'Mode_prediction_tta': mode_predictions,
        'Mode_confidence_(soft)': mode_confidences_s_tta,
        'Mode_certainty_(soft)': mode_certainties_s_tta,
        'True or false prediction': true_pred
    })

    # Plot histograms
    hist(df, 'Mode_confidence_(soft)', (4, 4))
    hist(df, 'Mode_certainty_(soft)', (4, 4))

    # Confusion matrix for mode-based TTA predictions
    cm = confusion_matrix(mode_labels.cpu(), mode_predictions.cpu())
    print('Mode based TTA predictions (TTAM)')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=config.dataset.class_names).plot()

    # Weighted TTA predictions
    weightedPred = ttaWeightedPred(test_attr_tta['labels_tta'], test_attr_tta['predictions_tta'], test_attr_tta['confidences_s_tta'], test_attr_tta['certainties_s_tta'])
    n_correct_TTAWCo_S = (weightedPred['predictionsCo'] == mode_labels).sum().item()
    n_correct_TTAWCe_S = (weightedPred['predictionsCe'] == mode_labels).sum().item()

    n_samples = mode_labels.shape[0]
    accuracy_TTAWCo_S = n_correct_TTAWCo_S / n_samples
    accuracy_TTAWCe_S = n_correct_TTAWCe_S / n_samples

    print('---------------------- Soft TTA predictions -----------------------------')
    print('n_samples', n_samples)
    print(f'accuracy of confidences based soft TTA predictions = {accuracy_TTAWCo_S}')
    print(f'accuracy of certainties based soft TTA predictions = {accuracy_TTAWCe_S}')

    # Confusion matrix for confidence-based soft TTA predictions
    cm = confusion_matrix(mode_labels.cpu(), weightedPred['predictionsCo'].cpu())
    print('Confidence based soft TTA predictions (TTAWCo-S)')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=config.dataset.class_names).plot()

    # Summary table
    test_accuracy_summary = {
        'ID': logger._task.id,
        'seed': config.dataset.seed,
        '# samples': len(test_labels_tta),
        '# TTA': config.dataset.num_tta,
        'Seed': config.dataset.seed,
        'F1 (without TTA)': f1,
        'Acc (without TTA)': accuracy,
        'Acc TTAM': accuracy_tta(mode_labels, mode_predictions)['acc_without_u'],
        'Acc TTAWCo-S': accuracy_TTAWCo_S,
        'Acc TTAWCe-S': accuracy_TTAWCe_S,
        'ECE': ECE
    }

    summary_df = pd.DataFrame([test_accuracy_summary])
    summary_df.to_csv(f"{config.model.name}_summary.csv", index=False)

    # Calculate std/error for metrics across seeds
    # Assuming you have a list of summary dataframes from different seeds
    # summary_dfs = [summary_df1, summary_df2, ...]
    # combined_summary_df = pd.concat(summary_dfs)
    # metrics_std = combined_summary_df.std()
    # metrics_error = combined_summary_df.sem()

if __name__ == "__main__":
    test()