import torch
import torchmetrics
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #inp = ((std * inp) + mean)
    inp = np.clip(inp, 0, 1)
    #plt.grid(visible=None, which='major',axis='both')
    plt.axis('off')
    plt.imshow(inp)
    plt.title(title)
    plt.savefig(f'{title}.png')
    plt.close()

def hist(df, values, histSize, name):
    
    plt.figure(figsize=(8,8))
    
    sns.set(style="darkgrid")
    sns.set(rc={'figure.figsize': histSize})

    n = 50

    sns.histplot(data=df[df['True or false prediction'] == True], x = values, color="skyblue", 
                 label='True predictions',  bins=n, kde=True)
    sns.histplot(data=df[df['True or false prediction'] == False],x = values, color="red",     
                 label='False predictions', bins=n, kde=True)
    plt.legend()
    plt.title(f'{name}')
    plt.savefig(f'hist_{name}.png')
    plt.close()

def calculate_ece(probabilities, labels, num_classes=2, n_bins=32, norm='l1'):
    """
    Calculate the Expected Calibration Error (ECE).

    Args:
        probabilities (torch.Tensor): Predicted probabilities.
        labels (torch.Tensor): True labels.
        num_classes (int): Number of classes.
        n_bins (int): Number of bins for calibration error calculation.
        norm (str): Norm to use for calibration error calculation.

    Returns:
        float: Expected Calibration Error.
    """
    ece_metric = torchmetrics.classification.MulticlassCalibrationError(
        num_classes=num_classes, n_bins=n_bins, norm=norm
    )
    return ece_metric(probabilities, labels).item()

def calculate_accuracy(preds, labels):
    """
    Calculate the accuracy.

    Args:
        preds (torch.Tensor): Predicted labels.
        labels (torch.Tensor): True labels.

    Returns:
        float: Accuracy.
    """
    return torch.sum(labels == preds).item() / torch.sum(labels == labels).item()

def calculate_f1_score_binary(preds, labels):
    """
    Calculate the F1 score for binary classification where 1 is the rare class.

    Args:
        preds (torch.Tensor): Predicted labels.
        labels (torch.Tensor): True labels.

    Returns:
        dict: Dictionary containing precision, recall, F1 score, sensitivity, specificity, and balanced accuracy.
    """
    tp = torch.logical_and((preds == labels), (preds == 1)).sum().item()
    tn = torch.logical_and((preds == labels), (preds == 0)).sum().item()
    fp = torch.logical_and((preds != labels), (preds == 1)).sum().item()
    fn = torch.logical_and((preds != labels), (preds == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    b_acc = (sensitivity + specificity) / 2
    f1 = 2 * precision * recall / (precision + recall) if precision or recall != 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_accuracy': b_acc,
    }

def certain_predictions(x, probv, nc, loss_fun):
    """
    Calculate certainty and confidence predictions.

    Args:
        x (torch.Tensor): Model outputs.
        probv (torch.Tensor): Maximum probabilities.
        nc (int): Number of classes.
        loss_fun (str): Loss function type.

    Returns:
        dict: Dictionary containing soft and hard confidence and certainty values.
    """
    if loss_fun == 'UACS':
        unc = torch.sigmoid(x[:, nc:])
        certainty_s = 1 - torch.squeeze(unc)
        confidence_s = probv
    elif loss_fun[-5:] == 'UANLL':
        certainty = torch.exp(-x[:, nc:])
        confidence_s = probv
        certainty_s = torch.squeeze(certainty)
    else:
        confidence_s = probv
        certainty_s = torch.zeros_like(confidence_s)


    return {
        'confidence_s': confidence_s,
        'certainty_s': certainty_s,
    }

def accuracy_tta(labels, pred, values_h=None):
    """
    Calculate accuracy with and without uncertainty estimation.

    Args:
        labels (torch.Tensor): True labels.
        pred (torch.Tensor): Predicted labels.
        values_h (torch.Tensor): Hard values for uncertainty estimation.

    Returns:
        dict: Dictionary containing accuracy metrics.
    """
    with torch.no_grad():
        if values_h is None:
            values_h = pred
        n_samples = labels.shape[0]
        n_correct = (pred == labels).sum().item()
        n_samples_cer = (values_h == 1).sum().item()
        n_correct_cer = torch.logical_and((pred == labels), (values_h == 1)).sum().item()

        acc_without_u = n_correct / n_samples
        acc_with_u = n_correct_cer / n_samples_cer if n_samples_cer > 0 else torch.tensor(0.0)

        return {
            'pred': pred,
            'values_h': values_h,
            'n_samples': n_samples,
            'n_correct': n_correct,
            'n_samples_cer': n_samples_cer,
            'n_correct_cer': n_correct_cer,
            'acc_without_u': acc_without_u,
            'acc_with_u': acc_with_u,
        }

def test_vis(model, loader, device, nc, loss_fun):
    """
    Visualize test set predictions.

    Args:
        model (torch.nn.Module): Model to use for predictions.
        loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): Device to use for computation.
        nc (int): Number of classes.

    Returns:
        tuple: Tuple containing test inputs and test attributes as a DataFrame.
    """
    test_pred = []
    test_prob = []
    test_prob_comp = []
    test_labe = []
    test_cert_s = []
    test_conf_s = []
    test_X = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            x = model(inputs).to(device)
            prob = torch.nn.functional.softmax(x[:, :nc], 1)
            probv, pred = torch.max(prob, 1)

            test_pred.append(pred)
            test_prob.append(probv)
            test_prob_comp.append(prob)
            test_labe.append(labels)
            test_X.append(inputs)

            cert_attr = certain_predictions(x, probv, nc, loss_fun)
            test_cert_s.append(cert_attr['certainty_s'])
            test_conf_s.append(cert_attr['confidence_s'])

    test_labels = torch.cat(test_labe, dim=0)
    test_predictions = torch.cat(test_pred, dim=0)
    test_prob_complete = torch.cat(test_prob_comp, dim=0)
    test_inputs = torch.cat(test_X, dim=0)
    test_certainties_s = torch.cat(test_cert_s, dim=0)
    test_confidences_s = torch.cat(test_conf_s, dim=0)

    ece = calculate_ece(test_prob_complete, test_labels, num_classes=nc)
    accu = calculate_accuracy(test_predictions, test_labels)
    some_metric = calculate_f1_score_binary(test_predictions, test_labels)
    f1 = some_metric['f1']
    b_acc = some_metric['balanced_accuracy']

    test_attr = {
        'test_labels': test_labels.cpu().numpy(),
        'test_predictions': test_predictions.cpu().numpy(),
        'test_certainties_s': test_certainties_s.cpu().numpy(),
        'test_confidences_s': test_confidences_s.cpu().numpy(),
        'ECE': ece,
        'acc': accu,
        'balanced_acc': b_acc,
        'F1': f1,
    }

    return test_inputs, pd.DataFrame(data=test_attr)

def test_vis_tta(model, loader, num_classes, loss_fun, seed, figs=1, fSize=(8, 8), nSamples=4, device='cuda', numTTA=10):
    """
    Perform test-time augmentation (TTA) and visualize test set predictions.

    Args:
        model (torch.nn.Module): Model to use for predictions.
        loader (torch.utils.data.DataLoader): DataLoader for the test set.
        figs (int): Number of figures to plot.
        fSize (tuple): Figure size.
        nSamples (int): Number of samples to visualize.
        device (torch.device): Device to use for computation.
        numTTA (int): Number of TTA iterations.

    Returns:
        dict: Dictionary containing TTA attributes.
    """
    test_labels_tta = []
    test_predictions_tta = []
    test_certainties_s_tta = []
    test_confidences_s_tta = []

    for i in range(numTTA):
        test_inputs, test_attr = test_vis(model.to(device), loader, device, num_classes, loss_fun)
        if i < figs:
            out = torchvision.utils.make_grid(test_inputs[:nSamples].cpu(), nrow=4)
            fig = plt.figure(figsize=fSize)
            imshow(out, title=f'Test-time augmentation {loss_fun} {seed}')
        test_labels_tta.append(test_attr['test_labels'])
        test_predictions_tta.append(test_attr['test_predictions'])
        test_certainties_s_tta.append(test_attr['test_certainties_s'])
        test_confidences_s_tta.append(test_attr['test_confidences_s'])
        del test_attr

    labels_tta = torch.as_tensor(test_labels_tta)
    predictions_tta = torch.as_tensor(test_predictions_tta)
    certainties_s_tta = torch.as_tensor(test_certainties_s_tta)
    confidences_s_tta = torch.as_tensor(test_confidences_s_tta)

    return {
        'labels_tta': labels_tta,
        'predictions_tta': predictions_tta,
        'certainties_s_tta': certainties_s_tta,
        'confidences_s_tta': confidences_s_tta,
    }

def ttac(mode_template, predictions_tta, values_tta, labels, class_names):
    """
    Calculate weighted predictions based on TTA.

    Args:
        mode_template (torch.Tensor): Zero-like tensor.
        predictions_tta (torch.Tensor): TTA predictions.
        values_tta (torch.Tensor): TTA values.
        labels (torch.Tensor): True labels.
        class_names (list): List of class names.

    Returns:
        torch.Tensor: Weighted predictions.
    """
    l = mode_template.size()
    predictions_w = mode_template * 0
    for i in range(l[0]):
        ttap = predictions_tta[:, i]
        ttaw = values_tta[:, i]
        freq_w = torch.bincount(ttap, weights=ttaw)
        _, predictions_w[i] = torch.max(freq_w, 0)
        if i < 4:
            print(ttap, ttaw, predictions_w[i].item(), labels[i].item(),
                  class_names[predictions_w[i].item()], class_names[labels[i].item()])
    return predictions_w

def ttaWeightedPred(labels_tta, predictions_tta, confidences_tta, certainties_tta, class_names):
    """
    Calculate weighted predictions based on TTA.

    Args:
        labels_tta (torch.Tensor): TTA labels.
        predictions_tta (torch.Tensor): TTA predictions.
        confidences_tta (torch.Tensor): TTA confidences.
        certainties_tta (torch.Tensor): TTA certainties.

    Returns:
        dict: Dictionary containing weighted predictions.
    """
    mode_labels, _ = torch.mode(labels_tta, dim=0)
    mode_template = torch.zeros_like(mode_labels)

    predictionsCo = ttac(mode_template, predictions_tta, confidences_tta, mode_labels, class_names)
    predictionsCe = ttac(mode_template, predictions_tta, certainties_tta, mode_labels, class_names)

    return {
        'predictionsCo': predictionsCo,
        'predictionsCe': predictionsCe
    }