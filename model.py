"""
Multitask Learning Framework for Clinical Text and Structured Data

This script defines a hybrid multitask model that integrates:
- A feedforward neural network for classification on structured features.
- A T5-based conditional generation model for synthetic clinical descriptions.

It supports synthetic text augmentation by training on synthetic descriptions
before actual clinical text is introduced. The setup includes:
- A custom PyTorch model class (MultitaskModel).
- A custom dataset class (MultitaskDataset).
- Training and evaluation loops for multitask learning.

Dependencies:
    - pandas
    - torch
    - transformers
    - sklearn
    - torchmetrics
"""

import pandas as pd
import torch
import torch.nn as nn # neural network layers and modules
import torch.optim as optim # optimisers
from torch.utils.data import DataLoader, Dataset # data batching tools
from transformers import T5Tokenizer, T5ForConditionalGeneration # huggingface T5 model + tokenizer
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix # sklearn metrics for evaluation
import torchmetrics # extra metrics

class MultitaskModel(nn.Module):
    """
    A multitask model combining structured data classification and
    T5-based text generation for synthetic clinical descriptions.

    Components:
        - A feedforward encoder for tabular features.
        - A classifier branch for predicting discrete labels.
        - A T5 decoder for conditional text generation.

    Args:
        num_features (int): Number of input features.
        embedding_dim (int): Dimension of the intermediate embedding.
        num_classes (int): Number of classification categories.
        t5_model_name (str): HuggingFace model name for the T5 generator.
    """
    def __init__(self, num_features=44, embedding_dim=256, num_classes=6, t5_model_name='t5-small'):
        super().__init__()
        
        # shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),  # Add batch norm for stability
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification branch
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Smaller T5 model or consider freezing some layers
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        
        # Optional: Freeze T5 encoder to reduce computation/memory
        for param in self.t5.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, features, input_ids=None, attention_mask=None, labels=None):
        # Pass features through encoder
        embedding = self.encoder(features)
        
        # Classification output
        class_logits = self.classifier(embedding)
        
        # T5 generation (only when text inputs provided)
        if input_ids is not None and labels is not None:
            t5_output = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return class_logits, t5_output.loss, t5_output.logits
        else:
            return class_logits, None, None

# Define Dataset class
class MultitaskDataset(Dataset):
    """
    A custom PyTorch Dataset for multitask learning that combines structured features 
    and tokenized text data.

    Args:
        df (pandas.DataFrame): The input dataframe containing the dataset. Each row should 
            include 'features' (a list or array of floats), 'class_label' (an integer), and 
            'synthetic_description' (a string).
        tokenizer (transformers.PreTrainedTokenizer): A HuggingFace tokenizer used to 
            tokenize the text descriptions.
        max_length (int, optional): The maximum sequence length for tokenized text. Defaults to 64.

    Returns:
        Tuple containing:
            - features (torch.FloatTensor): Structured input features.
            - label (torch.LongTensor): Class label.
            - input_ids (torch.LongTensor): Token IDs from the tokenizer.
            - attention_mask (torch.LongTensor): Attention mask from the tokenizer.
            - input_ids (torch.LongTensor): Token IDs (repeated, may be redundant).
    """
    def __init__(self, df, tokenizer, max_length=64):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row['features'], dtype=torch.float)
        label = torch.tensor(row['class_label'], dtype=torch.long)
        text = row['synthetic_description']

        tokenized = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)

        return features, label, input_ids, attention_mask, tokenized['input_ids'].squeeze(0)

# Training loop skeleton
def train(model, dataloader, optimizer, device):
    """
    Trains the multitask model on classification and (optionally) text generation.

    Args:
        model (nn.Module): The multitask model.
        dataloader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device to run training on (CPU or GPU).

    Returns:
        float: Average training loss over the dataset.
    """
    model.train() # set model to train mode
    clf_loss_fn = nn.CrossEntropyLoss() # classification loss function
    total_loss = 0

    for features, labels, input_ids, attention_mask, target_ids in dataloader:
        features, labels = features.to(device), labels.to(device)
        input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(device)

        optimizer.zero_grad() # clear gradients

        class_logits, gen_loss, _ = model(features, input_ids, attention_mask, target_ids) # forward pass
        clf_loss = clf_loss_fn(class_logits, labels) # compute classification loss

        if gen_loss is not None:
            total_batch_loss = clf_loss + gen_loss
        else:
            total_batch_loss = clf_loss
        total_batch_loss.backward() # backward pass
        optimizer.step() # update parameters

        total_loss += total_batch_loss.item() # track loss

    return total_loss / len(dataloader) # return average loss

def evaluate(model, dataloader, device):
    """
    Evaluates the multitask model on classification and optional generation loss.

    Args:
        model (nn.Module): The multitask model.
        dataloader (DataLoader): Validation/test data loader.
        device (torch.device): Device to run evaluation on.

    Returns:
        Tuple:
            - float: Average evaluation loss.
            - dict: Dictionary containing evaluation metrics (accuracy, f1_score).
    """
    model.eval()
    clf_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, input_ids, attention_mask, target_ids in dataloader:
            features, labels = features.to(device), labels.to(device)
            input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(device)
            
            class_logits, gen_loss, _ = model(features, input_ids, attention_mask, target_ids)
            clf_loss = clf_loss_fn(class_logits, labels)
            
            # Handle case where gen_loss might be None
            if gen_loss is not None:
                batch_loss = clf_loss + gen_loss
            else:
                batch_loss = clf_loss
                
            total_loss += batch_loss.item()
            
            predictions = torch.argmax(class_logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    return total_loss / len(dataloader), {'accuracy': accuracy, 'f1_score': f1}