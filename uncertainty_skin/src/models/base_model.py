from pytorch_metric_learning import distances, losses, miners, reducers, testers, regularizers

from src.functional.criterion import UANLLloss
from src.data.datamodule import ISICDataModule

import logging
import sys

from typing import List, Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader

import numpy as np

from src.utils.metrics import calculate_ece, calculate_accuracy, calculate_f1_score_binary, certain_predictions, accuracy_tta, test_vis_tta, ttac, ttaWeightedPred, hist

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


logging.basicConfig(
    stream=sys.stderr, 
    level=logging.DEBUG, 
    format="%(asctime)s %(levelname)s: %(message)s"
)

class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.build_model()
        self.loss_fun, self.loss_module_1 = self.build_loss()
        self.mining_func = miners.TripletMarginMiner(margin=0.2, distance=distances.CosineSimilarity(), type_of_triplets="semihard")

        self.epoch_val_loss = []
        self.epoch_train_loss = []

    def build_model(self):
        raise NotImplementedError

    def build_loss(self):
        if self.config.model.loss_fun == 'TM+UANLL':
            distance = distances.CosineSimilarity()
            reducer = reducers.ThresholdReducer(low=0)
            eRegularizer = regularizers.LpRegularizer()
            loss_fun = losses.TripletMarginLoss(margin=self.config.model.margin, distance=distance, reducer=reducer, embedding_regularizer=eRegularizer)
            loss_module_1 = UANLLloss(smoothing=self.config.model.label_smoothing)
            print('UANLL loss is an additional loss term (module 1)')
        elif self.config.model.loss_fun == 'TM+CE':
            distance = distances.CosineSimilarity()
            reducer = reducers.ThresholdReducer(low=0)
            eRegularizer = regularizers.LpRegularizer()
            loss_fun = losses.TripletMarginLoss(margin=self.config.model.margin, distance=distance, reducer=reducer, embedding_regularizer=eRegularizer)
            loss_module_1 = nn.CrossEntropyLoss(label_smoothing=self.config.model.label_smoothing)
            print('CE loss is an additional loss term (module 1)')
        elif self.config.model.loss_fun == 'CE':
            loss_fun = None
            loss_module_1 = nn.CrossEntropyLoss(label_smoothing=self.config.model.label_smoothing)
            print('Single CE loss')
        elif self.config.model.loss_fun == 'UANLL':
            loss_fun = None
            loss_module_1 = UANLLloss(smoothing=self.config.model.label_smoothing)
            print('Single UANLL loss')
        else:
            raise ValueError(f"Unknown loss function: {self.config.model.loss_fun}")
        return loss_fun, loss_module_1

    def forward(self, x):
        return self.model(x)
    
    def lr_lambda(self, epoch, n=150, delay=30, stop_lr=0):
        n = self.config.trainer.max_epochs
        start_lr = self.config.model.lr
        learning_rate = start_lr if epoch < delay else start_lr - (epoch - delay) * (start_lr - stop_lr) / (n - 1 - delay)
        return learning_rate / start_lr

    def configure_optimizers(self):
        if self.config.model.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.config.model.optimizer_hparams)
        elif self.config.model.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.config.model.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.config.model.optimizer_name}"'

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        if self.loss_fun is not None:
            indices_tuple = self.mining_func(embeddings[:, :self.config.model.num_classes], y)
            loss = self.loss_fun(embeddings[:, :self.config.model.num_classes], y, indices_tuple)
        else:
            loss = 0
        loss += self.loss_module_1(embeddings, y)
        if self.config.model.loss_fun in ['TM+CE', 'CE']:
            preds = nn.functional.softmax(embeddings, 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        elif self.config.model.loss_fun in ['TM+UANLL', 'UANLL']:
            preds = nn.functional.softmax(embeddings[:, :self.config.model.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        self.log("train_loss", loss.float(), on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        if self.loss_fun is not None:
            indices_tuple = self.mining_func(embeddings[:, :self.config.model.num_classes], y)
            loss = self.loss_fun(embeddings[:, :self.config.model.num_classes], y, indices_tuple)
        else:
            loss = 0
        loss += self.loss_module_1(embeddings, y)
        if self.config.model.loss_fun in ['TM+CE', 'CE']:
            preds = nn.functional.softmax(embeddings[:, :self.config.model.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        else:
            acc = (embeddings[:, :self.config.model.num_classes].argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss.float(), on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        if self.config.model.loss_fun in ['TM+CE', 'CE']:
            preds = nn.functional.softmax(embeddings[:, :self.config.model.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        else:
            acc = (embeddings[:, :self.config.model.num_classes].argmax(dim=-1) == y).float().mean()
        print("Test set accuracy without TTA (Precision@1) = {}".format(acc))
        self.log("test_acc", acc, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """
        Perform predictions, TTA, ensembling, and weighted predictions at the end of the test epoch.
        """
        # Collect predictions without TTA
        test_predictions_no_tta, test_labels_no_tta = self.collect_predictions_no_tta()
        self.evaluate_no_tta(test_predictions_no_tta, test_labels_no_tta)

        # Perform TTA and ensembling
        test_predictions_tta, test_labels_tta, test_confidences_tta, test_uncertainties_tta = self.perform_tta_and_ensembling()
        self.evaluate_with_tta(test_predictions_tta, test_labels_tta)

        # Save predictions to CSV
        self.save_predictions_to_csv(test_predictions_no_tta, test_labels_no_tta, test_predictions_tta, test_labels_tta, test_confidences_tta, test_uncertainties_tta)

        # Handle weighted predictions based on confidence and certainty
        self.handle_weighted_predictions(test_predictions_tta, test_labels_tta, test_confidences_tta, test_uncertainties_tta)

        # Compare metrics for different approaches
        self.compare_metrics(test_predictions_no_tta, test_labels_no_tta, test_predictions_tta, test_labels_tta, test_confidences_tta, test_uncertainties_tta)

    def compare_metrics(self, test_predictions_no_tta: torch.Tensor, test_labels_no_tta: torch.Tensor, test_predictions_tta: torch.Tensor, test_labels_tta: torch.Tensor, test_confidences_tta: torch.Tensor, test_uncertainties_tta: torch.Tensor) -> None:
        """
        Compare metrics for different approaches:
        - Without TTA
        - TTAM (Mode-based TTA)
        - TTAWCo-S (Weighted with Confidences)
        - TTAWCe-S (Weighted with Certainties)
        - Ensembling (Simple)
        - Ensembling with Confidences
        - Ensembling with Certainties
        - Ensembling with TTA

        Args:
            test_predictions_no_tta (torch.Tensor): Predictions without TTA.
            test_labels_no_tta (torch.Tensor): True labels without TTA.
            test_predictions_tta (torch.Tensor): Predictions with TTA.
            test_labels_tta (torch.Tensor): True labels with TTA.
            test_confidences_tta (torch.Tensor): Confidences with TTA.
            test_uncertainties_tta (torch.Tensor): Uncertainties with TTA.
        """
        #mode_predictions_tta = self.mode_based_tta(test_predictions_tta)
        
        # Convert tensors to numpy for sklearn metrics
        test_labels_no_tta = test_labels_no_tta.cpu().numpy()
        test_labels_tta = test_labels_tta.cpu().numpy()
        test_predictions_no_tta = test_predictions_no_tta.argmax(dim=1).cpu().numpy()
        test_predictions_tta = test_predictions_tta.argmax(dim=1).cpu().numpy()
        test_confidences_tta = test_confidences_tta.cpu().numpy()
        test_uncertainties_tta = test_uncertainties_tta.cpu().numpy()

        # 1. Without TTA
        accuracy_no_tta = accuracy_score(test_labels_no_tta, test_predictions_no_tta)
        f1_no_tta = f1_score(test_labels_no_tta, test_predictions_no_tta, average='weighted')
        roc_auc_no_tta = roc_auc_score(test_labels_no_tta, test_predictions_no_tta, average='weighted', multi_class='ovr')

        # 2. TTAM (Mode-based TTA)
        # accuracy_ttam = accuracy_score(test_labels_tta, mode_predictions_tta)
        # f1_ttam = f1_score(test_labels_tta, mode_predictions_tta, average='weighted')
        # roc_auc_ttam = roc_auc_score(test_labels_tta, mode_predictions_tta, average='weighted', multi_class='ovr')
        accuracy_ttam = 0
        f1_ttam = 0
        roc_auc_ttam = 0

        # 3. TTAWCo-S (Weighted with Confidences)
        weighted_predictions_co = self.weighted_predictions_with_confidence(test_predictions_tta, test_confidences_tta)
        accuracy_tta_co = accuracy_score(test_labels_tta, weighted_predictions_co)
        f1_tta_co = f1_score(test_labels_tta, weighted_predictions_co, average='weighted')
        roc_auc_tta_co = roc_auc_score(test_labels_tta, weighted_predictions_co, average='weighted', multi_class='ovr')

        # 4. TTAWCe-S (Weighted with Certainties)
        weighted_predictions_ce = self.weighted_predictions_with_certainty(test_predictions_tta, test_uncertainties_tta)
        accuracy_tta_ce = accuracy_score(test_labels_tta, weighted_predictions_ce)
        f1_tta_ce = f1_score(test_labels_tta, weighted_predictions_ce, average='weighted')
        roc_auc_tta_ce = roc_auc_score(test_labels_tta, weighted_predictions_ce, average='weighted', multi_class='ovr')

        # 5. Ensembling (Simple)
        ensembled_predictions = self.ensemble_predictions(test_predictions_tta)
        accuracy_ensemble = accuracy_score(test_labels_tta, ensembled_predictions)
        f1_ensemble = f1_score(test_labels_tta, ensembled_predictions, average='weighted')
        roc_auc_ensemble = roc_auc_score(test_labels_tta, ensembled_predictions, average='weighted', multi_class='ovr')

        # 6. Ensembling with Confidences
        ensembled_predictions_co = self.ensemble_predictions_with_confidence(test_predictions_tta, test_confidences_tta)
        accuracy_ensemble_co = accuracy_score(test_labels_tta, ensembled_predictions_co)
        f1_ensemble_co = f1_score(test_labels_tta, ensembled_predictions_co, average='weighted')
        roc_auc_ensemble_co = roc_auc_score(test_labels_tta, ensembled_predictions_co, average='weighted', multi_class='ovr')

        # 7. Ensembling with Certainties
        ensembled_predictions_ce = self.ensemble_predictions_with_certainty(test_predictions_tta, test_uncertainties_tta)
        accuracy_ensemble_ce = accuracy_score(test_labels_tta, ensembled_predictions_ce)
        f1_ensemble_ce = f1_score(test_labels_tta, ensembled_predictions_ce, average='weighted')
        roc_auc_ensemble_ce = roc_auc_score(test_labels_tta, ensembled_predictions_ce, average='weighted', multi_class='ovr')

        # 8. Ensembling with TTA
        ensembled_tta_predictions = self.ensemble_tta_predictions(test_predictions_tta)
        accuracy_ensemble_tta = accuracy_score(test_labels_tta, ensembled_tta_predictions)
        f1_ensemble_tta = f1_score(test_labels_tta, ensembled_tta_predictions, average='weighted')
        roc_auc_ensemble_tta = roc_auc_score(test_labels_tta, ensembled_tta_predictions, average='weighted', multi_class='ovr')

        # Create a metrics comparison table
        metrics_table = pd.DataFrame({
            'Approach': ['Without TTA', 'TTAM', 'TTAWCo-S', 'TTAWCe-S', 'Ensembling (Simple)', 'Ensembling with Confidences', 'Ensembling with Certainties', 'Ensembling with TTA'],
            'Accuracy': [accuracy_no_tta, accuracy_ttam, accuracy_tta_co, accuracy_tta_ce, accuracy_ensemble, accuracy_ensemble_co, accuracy_ensemble_ce, accuracy_ensemble_tta],
            'F1 Score': [f1_no_tta, f1_ttam, f1_tta_co, f1_tta_ce, f1_ensemble, f1_ensemble_co, f1_ensemble_ce, f1_ensemble_tta],
            'ROC-AUC': [roc_auc_no_tta, roc_auc_ttam, roc_auc_tta_co, roc_auc_tta_ce, roc_auc_ensemble, roc_auc_ensemble_co, roc_auc_ensemble_ce, roc_auc_ensemble_tta]
        })

        # Save the metrics table to a CSV file
        metrics_table.to_csv(f"{self.config.model.name}_{self.config.dataset.seed}_metrics_comparison.csv", index=False)

        # Print the metrics table
        print(metrics_table)

    def mode_based_tta(self, test_predictions_tta: torch.Tensor) -> np.ndarray:
        """
        Compute mode-based TTA predictions.

        Args:
            test_predictions_tta (torch.Tensor): Predictions with TTA.

        Returns:
            np.ndarray: Mode-based TTA predictions.
        """
        return torch.mode(test_predictions_tta, dim=0).values.cpu().numpy()

    def weighted_predictions_with_confidence(self, test_predictions_tta: torch.Tensor, test_confidences_tta: torch.Tensor) -> np.ndarray:
        """
        Compute weighted predictions based on confidence.

        Args:
            test_predictions_tta (torch.Tensor): Predictions with TTA.
            test_confidences_tta (torch.Tensor): Confidences with TTA.

        Returns:
            np.ndarray: Weighted predictions based on confidence.
        """
        weighted_predictions_co = test_predictions_tta
        weighted_predictions_co[test_confidences_tta < 0.5] = -1  # Ignore low-confidence predictions
        return weighted_predictions_co
    
    def weighted_predictions_with_certainty(self, test_predictions_tta: torch.Tensor, test_uncertainties_tta: torch.Tensor) -> np.ndarray:
        """
        Compute weighted predictions based on certainty.

        Args:
            test_predictions_tta (torch.Tensor): Predictions with TTA.
            test_uncertainties_tta (torch.Tensor): Uncertainties with TTA.

        Returns:
            np.ndarray: Weighted predictions based on certainty.
        """
        weighted_predictions_ce = test_predictions_tta
        weighted_predictions_ce[test_uncertainties_tta > 0.5] = -1  # Ignore high-uncertainty predictions
        return weighted_predictions_ce

    def ensemble_predictions(self, test_predictions_tta: torch.Tensor) -> np.ndarray:
        """
        Compute simple ensembled predictions.

        Args:
            test_predictions_tta (torch.Tensor): Predictions with TTA.

        Returns:
            np.ndarray: Ensembled predictions.
        """
        return test_predictions_tta.mean(dim=0).argmax(dim=1).cpu().numpy()

    def ensemble_predictions_with_confidence(self, test_predictions_tta: torch.Tensor, test_confidences_tta: torch.Tensor) -> np.ndarray:
        """
        Compute ensembled predictions weighted by confidence.

        Args:
            test_predictions_tta (torch.Tensor): Predictions with TTA.
            test_confidences_tta (torch.Tensor): Confidences with TTA.

        Returns:
            np.ndarray: Ensembled predictions weighted by confidence.
        """
        weighted_predictions = test_predictions_tta * test_confidences_tta.unsqueeze(1)
        return weighted_predictions.sum(dim=0).argmax(dim=1).cpu().numpy()

    def ensemble_predictions_with_certainty(self, test_predictions_tta: torch.Tensor, test_uncertainties_tta: torch.Tensor) -> np.ndarray:
        """
        Compute ensembled predictions weighted by certainty.

        Args:
            test_predictions_tta (torch.Tensor): Predictions with TTA.
            test_uncertainties_tta (torch.Tensor): Certainties with TTA.

        Returns:
            np.ndarray: Ensembled predictions weighted by certainty.
        """
        weighted_predictions = test_predictions_tta * test_uncertainties_tta.unsqueeze(1)
        return weighted_predictions.sum(dim=0).argmax(dim=1).cpu().numpy()

    def ensemble_tta_predictions(self, test_predictions_tta: torch.Tensor) -> np.ndarray:
        """
        Compute ensembled predictions with TTA.

        Args:
            test_predictions_tta (torch.Tensor): Predictions with TTA.

        Returns:
            np.ndarray: Ensembled predictions with TTA.
        """
        return test_predictions_tta.mean(dim=0).argmax(dim=1).cpu().numpy()

    def collect_predictions_no_tta(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Collect predictions without Test-Time Augmentation (TTA).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Predictions and labels without TTA.
        """
        test_predictions_no_tta, test_labels_no_tta = [], []
        for batch in self.trainer.datamodule.test_dataloader():
            inputs, labels = batch
            inputs = inputs.to(self.device)
            outputs = self(inputs)

            # Handle model output as in test_step
            if self.config.model.loss_fun in ['TM+CE', 'CE']:
                preds = torch.softmax(outputs[:, :self.config.model.num_classes], dim=1)
            else:
                preds = torch.softmax(outputs[:, :self.config.model.num_classes], dim=1)

            test_predictions_no_tta.append(preds)
            test_labels_no_tta.append(labels)

        test_predictions_no_tta = torch.cat(test_predictions_no_tta)
        test_labels_no_tta = torch.cat(test_labels_no_tta)
        return test_predictions_no_tta, test_labels_no_tta

    def evaluate_no_tta(self, test_predictions_no_tta: torch.Tensor, test_labels_no_tta: torch.Tensor) -> None:
        """
        Evaluate predictions without TTA.

        Args:
            test_predictions_no_tta (torch.Tensor): Predictions without TTA.
            test_labels_no_tta (torch.Tensor): True labels without TTA.
        """
        test_predictions_no_tta = test_predictions_no_tta.argmax(dim=1).cpu()
        test_labels_no_tta = test_labels_no_tta.squeeze().cpu()

        accuracy_no_tta = accuracy_score(test_labels_no_tta, test_predictions_no_tta)
        f1_no_tta = f1_score(test_labels_no_tta, test_predictions_no_tta, average='weighted')
        roc_auc_no_tta = roc_auc_score(test_labels_no_tta, test_predictions_no_tta, average='weighted', multi_class='ovr')

        print(f"Metrics without TTA: Accuracy={accuracy_no_tta}, F1={f1_no_tta}, ROC-AUC={roc_auc_no_tta}")

    def perform_tta_and_ensembling(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Test-Time Augmentation (TTA) and ensembling.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Ensembled predictions, labels, confidences, and uncertainties.
        """
        num_tta = self.config.dataset.num_tta
        checkpoint_paths = list(self.config.model.checkpoint_path)

        all_test_predictions_tta = []
        all_test_labels_tta = []
        all_test_confidences_tta = []
        all_test_uncertainties_tta = []

        for checkpoint_path in checkpoint_paths:
            # Reinitialize the model and datamodule for each checkpoint
            self.build_model()
            self.load_state_dict(torch.load(checkpoint_path, weights_only=True)['state_dict'])
            self.to(self.device)
            self.eval()

            datamodule = self.trainer.datamodule  # Reinitialize datamodule if needed
            test_loader = datamodule.test_dataloader()

            test_predictions_tta, test_labels_tta, test_confidences_tta, test_uncertainties_tta = self.collect_tta_predictions(self, test_loader, num_tta)
            all_test_predictions_tta.append(test_predictions_tta)
            all_test_labels_tta.append(test_labels_tta)
            all_test_confidences_tta.append(test_confidences_tta)
            all_test_uncertainties_tta.append(test_uncertainties_tta)

        # Ensemble predictions
        all_test_predictions_tta = torch.stack(all_test_predictions_tta).mean(dim=0)
        all_test_labels_tta = torch.stack(all_test_labels_tta).mode(dim=0).values
        all_test_confidences_tta = torch.stack(all_test_confidences_tta).mean(dim=0)
        all_test_uncertainties_tta = torch.stack(all_test_uncertainties_tta).mean(dim=0)

        return all_test_predictions_tta, all_test_labels_tta, all_test_confidences_tta, all_test_uncertainties_tta

    def collect_tta_predictions(self, model: torch.nn.Module, test_loader: DataLoader, num_tta: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collect predictions with Test-Time Augmentation (TTA) for a single checkpoint.

        Args:
            model (torch.nn.Module): The model to use for predictions.
            test_loader (DataLoader): The test dataloader.
            num_tta (int): Number of TTA iterations.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Predictions, labels, confidences, and uncertainties with TTA.
        """
        test_predictions_tta = []
        test_labels_tta = []
        test_confidences_tta = []
        test_uncertainties_tta = []

        for _ in range(num_tta):
            for batch in test_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                outputs = model(inputs)

                # Handle model output as in test_step
                if self.config.model.loss_fun in ['TM+CE', 'CE']:
                    preds = torch.softmax(outputs[:, :self.config.model.num_classes], dim=1)
                else:
                    preds = torch.softmax(outputs[:, :self.config.model.num_classes], dim=1)

                confidences = preds.max(dim=1).values
                uncertainties = outputs[:, -1] if self.config.model.loss_fun in ['UANLL', 'TM+UANLL'] else torch.zeros_like(confidences)

                test_predictions_tta.append(preds)
                test_labels_tta.append(labels)
                test_confidences_tta.append(confidences)
                test_uncertainties_tta.append(uncertainties)

        test_predictions_tta = torch.cat(test_predictions_tta)
        test_labels_tta = torch.cat(test_labels_tta)
        test_confidences_tta = torch.cat(test_confidences_tta)
        test_uncertainties_tta = torch.cat(test_uncertainties_tta)

        return test_predictions_tta, test_labels_tta, test_confidences_tta, test_uncertainties_tta

    def evaluate_with_tta(self, test_predictions_tta: torch.Tensor, test_labels_tta: torch.Tensor) -> None:
        """
        Evaluate predictions with TTA and ensembling.

        Args:
            test_predictions_tta (torch.Tensor): Predictions with TTA.
            test_labels_tta (torch.Tensor): True labels with TTA.
        """
        print(test_predictions_tta)
        test_predictions_tta = test_predictions_tta.argmax(dim=1).cpu()
        test_labels_tta = test_labels_tta.squeeze().cpu()

        accuracy_tta = accuracy_score(test_labels_tta, test_predictions_tta)
        f1_tta = f1_score(test_labels_tta, test_predictions_tta, average='weighted')
        roc_auc_tta = roc_auc_score(test_labels_tta, test_predictions_tta, average='weighted', multi_class='ovr')

        print(f"Metrics with TTA: Accuracy={accuracy_tta}, F1={f1_tta}, ROC-AUC={roc_auc_tta}")

    def save_predictions_to_csv(self, test_predictions_no_tta: torch.Tensor, test_labels_no_tta: torch.Tensor, test_predictions_tta: torch.Tensor, test_labels_tta: torch.Tensor, test_confidences_tta: torch.Tensor, test_uncertainties_tta: torch.Tensor) -> None:
        """
        Save predictions (without TTA and with TTA) to CSV files.

        Args:
            test_predictions_no_tta (torch.Tensor): Predictions without TTA.
            test_labels_no_tta (torch.Tensor): True labels without TTA.
            test_predictions_tta (torch.Tensor): Predictions with TTA.
            test_labels_tta (torch.Tensor): True labels with TTA.
            test_confidences_tta (torch.Tensor): Confidences with TTA.
            test_uncertainties_tta (torch.Tensor): Uncertainties with TTA.
        """
        # Save predictions without TTA
        no_tta_df = pd.DataFrame({
            'true_labels': test_labels_no_tta.tolist(),
            'predictions': test_predictions_no_tta.argmax(dim=1).tolist()
        })
        no_tta_df.to_csv(f"{self.config.model.name}_{self.config.dataset.seed}_predictions_no_tta.csv", index=False)

        # Save predictions with TTA
        tta_df = pd.DataFrame({
            'true_labels': test_labels_tta.tolist(),
            'predictions': test_predictions_tta.argmax(dim=1).tolist(),
            'confidences': test_confidences_tta.tolist(),
            'uncertainties': test_uncertainties_tta.tolist()
        })
        tta_df.to_csv(f"{self.config.model.name}_{self.config.dataset.seed}_predictions_tta.csv", index=False)

    def handle_weighted_predictions(self, test_predictions_tta: torch.Tensor, test_labels_tta: torch.Tensor, test_confidences_tta: torch.Tensor, test_uncertainties_tta: torch.Tensor) -> None:
        """
        Handle weighted predictions based on confidence and certainty.

        Args:
            test_predictions_tta (torch.Tensor): Predictions with TTA.
            test_labels_tta (torch.Tensor): True labels with TTA.
            test_confidences_tta (torch.Tensor): Confidences with TTA.
            test_uncertainties_tta (torch.Tensor): Uncertainties with TTA.
        """
        # Weighted predictions based on confidence
        weighted_predictions_co = test_predictions_tta.argmax(dim=1)
        weighted_predictions_co[test_confidences_tta < 0.5] = -1  # Example: Ignore low-confidence predictions

        # Weighted predictions based on certainty
        weighted_predictions_ce = test_predictions_tta.argmax(dim=1)
        weighted_predictions_ce[test_uncertainties_tta > 0.5] = -1  # Example: Ignore high-uncertainty predictions

        # Evaluate weighted predictions
        self.evaluate_weighted_predictions(weighted_predictions_co, weighted_predictions_ce, test_labels_tta)

    def evaluate_weighted_predictions(self, weighted_predictions_co: torch.Tensor, weighted_predictions_ce: torch.Tensor, test_labels_tta: torch.Tensor) -> None:
        """
        Evaluate weighted predictions based on confidence and certainty.

        Args:
            weighted_predictions_co (torch.Tensor): Weighted predictions based on confidence.
            weighted_predictions_ce (torch.Tensor): Weighted predictions based on certainty.
            test_labels_tta (torch.Tensor): True labels with TTA.
        """
        # Filter out ignored predictions
        valid_indices_co = weighted_predictions_co != -1
        valid_indices_ce = weighted_predictions_ce != -1
        
        valid_indices_ce = valid_indices_ce.cpu().numpy()
        valid_indices_co = valid_indices_co.cpu().numpy()
        
        test_labels_tta = test_labels_tta.cpu().numpy()
        weighted_predictions_co = weighted_predictions_co.cpu().numpy()
        weighted_predictions_ce = weighted_predictions_ce.cpu().numpy()

        # Evaluate confidence-weighted predictions
        accuracy_co = accuracy_score(test_labels_tta[valid_indices_co], weighted_predictions_co[valid_indices_co])
        f1_co = f1_score(test_labels_tta[valid_indices_co], weighted_predictions_co[valid_indices_co], average='weighted')
        roc_auc_co = roc_auc_score(test_labels_tta[valid_indices_co], weighted_predictions_co[valid_indices_co], average='weighted', multi_class='ovr')

        # Evaluate certainty-weighted predictions
        accuracy_ce = accuracy_score(test_labels_tta[valid_indices_ce], weighted_predictions_ce[valid_indices_ce])
        f1_ce = f1_score(test_labels_tta[valid_indices_ce], weighted_predictions_ce[valid_indices_ce], average='weighted')
        roc_auc_ce = roc_auc_score(test_labels_tta[valid_indices_ce], weighted_predictions_ce[valid_indices_ce], average='weighted', multi_class='ovr')

        print(f"Confidence-Weighted Metrics: Accuracy={accuracy_co}, F1={f1_co}, ROC-AUC={roc_auc_co}")
        print(f"Certainty-Weighted Metrics: Accuracy={accuracy_ce}, F1={f1_ce}, ROC-AUC={roc_auc_ce}")