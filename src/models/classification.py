import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

class GaussianNoise(nn.Module):
    """
    Module that adds Gaussian noise to inputs during training.

    During training, this layer perturbs the input tensor by adding random noise 
    drawn from a normal distribution with a specified mean and standard deviation. 
    This can help improve model robustness and prevent overfitting.

    The noise is only added when the module is in training mode. In evaluation mode,
    the input is returned as-is.

    Parameters:
        mean (float): The mean of the Gaussian noise. Default is 0.0.
        std (float): The standard deviation of the Gaussian noise. Default is 0.2.

    """
    def __init__(self, mean=0.0, std=0.2):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std + self.mean
        return x

    
class ClassificationModel(nn.Module):
    """
    A fully connected neural network for classification tasks with noise injection and regularization.

    This model consists of several linear layers interleaved with batch normalization, dropout, 
    ReLU activations, and Gaussian noise layers for improved generalization. It supports 
    configurable input feature size and number of output classes.

    The network architecture includes:
        - Input BatchNorm and GaussianNoise layers
        - Multiple hidden layers with linear transformations, batch normalization,
          dropout, ReLU activations, and Gaussian noise
        - Final linear layer producing logits for classification

    Weights of all linear layers are initialized using Xavier uniform initialization,
    and biases are initialized to zero.

    Args:
        num_features (int): Number of input features. Default is 44.
        num_classes (int): Number of output classes. Default is 6.

    """
    def __init__(self, num_features=44, num_classes=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_features),
            GaussianNoise(0.0, 0.2),
            nn.Linear(num_features, 1000),
            nn.ReLU(),

            nn.BatchNorm1d(1000),
            nn.Dropout(0.2),
            GaussianNoise(0.0, 0.2),
            nn.Linear(1000, 200),
            nn.ReLU(),

            nn.BatchNorm1d(200),
            nn.Dropout(0.2),
            GaussianNoise(0.0, 0.2),
            nn.Linear(200, 200),
            nn.ReLU(),

            nn.BatchNorm1d(200),
            nn.Dropout(0.2),
            GaussianNoise(0.0, 0.2),
            nn.Linear(200, 200),
            nn.ReLU(),

            nn.BatchNorm1d(200),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.ReLU(),

            nn.Linear(200, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)
    
    
class TabularDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(df['features'].tolist(), dtype=torch.float32)
        self.y = torch.tensor(df['class_label'].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
    
def train(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch on the given dataset loader.

    Performs a full pass through the data loader, computing predictions,
    calculating the loss, performing backpropagation, and updating model parameters.

    Args:
        model (nn.Module): The neural network model to train.
        loader (DataLoader): DataLoader providing training data batches.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        criterion (callable): Loss function to compute the training loss.
        device (torch.device): Device on which to perform computations (CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - average_loss (float): Average loss over all batches.
            - accuracy (float): Accuracy computed on the training data for this epoch.
    """
    model.train()
    running_loss = 0
    y_true, y_pred = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    return running_loss / len(loader), acc


def evaluate_binary(model, loader, criterion, device):
    """
    Evaluate a binary classification model on a validation or test dataset.

    Performs a forward pass without gradient computation and calculates multiple
    performance metrics including loss, accuracy, F1 score, sensitivity, specificity,
    and AUC-ROC score.

    Args:
        model (nn.Module): The trained model to evaluate.
        loader (DataLoader): DataLoader providing evaluation data batches.
        criterion (callable): Loss function used to compute the loss.
        device (torch.device): Device on which to perform computations (CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - average_loss (float): Average loss over the dataset.
            - accuracy (float): Classification accuracy.
            - f1_score (float): Weighted F1 score.
            - sensitivity (float or None): True positive rate, or None if undefined.
            - specificity (float or None): True negative rate, or None if undefined.
            - auc (float or None): Area under the ROC curve, or None if not computable.
    """
    model.eval()
    y_true, y_pred = [], []
    y_prob = []  # For AUC
    val_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            val_loss += criterion(outputs, y).item()

            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())  # prob for positive class

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    else:
        sensitivity, specificity = None, None

    # Compute AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = None

    return val_loss / len(loader), acc, f1, sensitivity, specificity, auc

def evaluate_multiclass(model, loader, criterion, device):
    model.eval()
    y_true, y_pred = [], []
    y_prob = []  # collect probabilities for AUC
    val_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            val_loss += criterion(outputs, y).item()

            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    # Sensitivity and Specificity
    if n_classes == 2:
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        sensitivity_arr = np.array([sensitivity, sensitivity])
        specificity_arr = np.array([specificity, specificity])
    else:
        sensitivity_arr = []
        specificity_arr = []
        for i in range(n_classes):
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = np.sum(cm) - (TP + FN + FP)

            sens = TP / (TP + FN) if (TP + FN) > 0 else 0
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0

            sensitivity_arr.append(sens)
            specificity_arr.append(spec)

        sensitivity_arr = np.array(sensitivity_arr)
        specificity_arr = np.array(specificity_arr)
        sensitivity = sensitivity_arr.mean()
        specificity = specificity_arr.mean()

    # AUC-ROC (Multiclass)
    try:
        y_true_one_hot = np.eye(n_classes)[y_true]
        auc = roc_auc_score(y_true_one_hot, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = None

    return (
        val_loss / len(loader),
        acc,
        f1,
        sensitivity,
        specificity,
        sensitivity_arr,
        specificity_arr,
        auc
    )

class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.

    This class monitors the validation loss during training and triggers early stopping
    if the loss does not improve by at least `min_delta` for a specified number of 
    consecutive epochs (`patience`).

    Args:
        patience (int): Number of epochs to wait without improvement before stopping. Default is 7.
        min_delta (float): Minimum change in validation loss to qualify as an improvement. Default is 0.001.
    """
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience
