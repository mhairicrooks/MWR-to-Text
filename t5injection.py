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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import torchmetrics
import random
import copy
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import random
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, confusion_matrix
)
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


class GaussianNoise(nn.Module):
    """Gaussian noise layer with specified mean and standard deviation"""
    def __init__(self, mean=0.0, std=0.2):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x
    
class MultitaskModel(nn.Module):
    """
    A multitask model that fuses T5 encoder representations with 
    raw temperature features for classification.
    """
    def __init__(self, num_features=44, embedding_dim=512, num_classes=6, t5_model_name='t5-small'):
        super().__init__()
        
        # T5 model for both encoding and generation
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name, local_files_only=True)
        
        # Get T5 encoder hidden size
        t5_hidden_size = self.t5.config.d_model
        
        # Raw feature encoder (your original encoder)
        self.feature_encoder = nn.Sequential(
            nn.BatchNorm1d(num_features),
            GaussianNoise(0.0, 0.2),
            nn.Linear(num_features, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.2),
            nn.Linear(1000, 512),          
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layer - combines T5 encoder output with raw features
        fusion_input_size = t5_hidden_size + 512  # T5 hidden + feature encoder output
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1)
        )
        
        # Classification branch - takes fused representation
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(512),
            GaussianNoise(0.0, 0.2),
            nn.Linear(512, 1000),  
            nn.ReLU(),
            
            nn.BatchNorm1d(1000, eps=1e-5),
            nn.Dropout(0.2),
            GaussianNoise(mean=0.0, std=0.2),
            nn.Linear(1000, 200),
            nn.ReLU(),
            
            nn.BatchNorm1d(200, eps=1e-5),
            nn.Dropout(0.2),
            GaussianNoise(mean=0.0, std=0.2),
            nn.Linear(200, 200),
            nn.ReLU(),
            
            nn.BatchNorm1d(200, eps=1e-5),
            nn.Dropout(0.2),
            GaussianNoise(mean=0.0, std=0.2),
            nn.Linear(200, 200),
            nn.ReLU(),
            
            nn.BatchNorm1d(200, eps=1e-5),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.ReLU(),
            
            nn.Linear(200, num_classes)
        )
        
        # Loss weights
        self.classification_weight = 0.1
        self.generation_weight = 10
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in [self.feature_encoder, self.fusion_layer, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, features, input_ids=None, attention_mask=None, labels=None):
        # Process raw features
        feature_embedding = self.feature_encoder(features)
        
        if input_ids is not None and attention_mask is not None:
            # Get T5 encoder representations
            encoder_outputs = self.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Pool encoder representations (mean pooling across sequence length)
            encoder_hidden = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
            # Mean pooling with attention mask
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_hidden.size()).float()
            sum_embeddings = torch.sum(encoder_hidden * attention_mask_expanded, dim=1)
            sum_mask = torch.sum(attention_mask_expanded, dim=1)
            pooled_encoder = sum_embeddings / sum_mask  # [batch_size, hidden_size]
            
            # Fuse T5 encoder output with raw features
            fused_representation = torch.cat([pooled_encoder, feature_embedding], dim=1)
            fused_embedding = self.fusion_layer(fused_representation)
            
            # Classification branch
            class_logits = self.classifier(fused_embedding)
            
            # Text generation branch
            if labels is not None:
                t5_output = self.t5(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                return (
                    class_logits, 
                    t5_output.loss, 
                    t5_output.logits,
                    self.classification_weight,
                    self.generation_weight
                )
            else:
                return (
                    class_logits, 
                    None, 
                    None, 
                    self.classification_weight, 
                    self.generation_weight
                )
        else:
            # Fallback: use only raw features if no text input
            class_logits = self.classifier(feature_embedding)
            return (
                class_logits, 
                None, 
                None, 
                self.classification_weight, 
                self.generation_weight
            )
        

# define Dataset class
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
        max_length (int, optional): The maximum sequence length for tokenized text. Defaults to 128.
        text_mode (str, optional): Text input mode - 'full' or 'minimal'. Defaults to 'full'.

    Returns:
        Tuple containing:
            - features (torch.FloatTensor): Structured input features.
            - label (torch.LongTensor): Class label.
            - input_ids (torch.LongTensor): Token IDs from the tokenizer.
            - attention_mask (torch.LongTensor): Attention mask from the tokenizer.
            - target_ids (torch.LongTensor): Target token IDs for generation.
    """
    def __init__(self, df, tokenizer, max_length=128, text_mode='full'):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_mode = text_mode
        
        # Validate text_mode
        if text_mode not in ['full', 'minimal']:
            raise ValueError("text_mode must be 'full' or 'minimal'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row['features'], dtype=torch.float)
        label = torch.tensor(row['class_label'], dtype=torch.long)
        
        # CREATE INPUT FROM TEMPERATURE READINGS based on text_mode
        input_text = self.create_temperature_input(row)
        
        # TARGET IS THE SYNTHETIC CONCLUSION
        target_text = row['synthetic_description']
        
        # Tokenize input (what the model sees)
        input_tokenized = self.tokenizer(
            input_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        # Tokenize target (what the model should generate)
        target_tokenized = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=64,  # Conclusions are shorter
            return_tensors='pt'
        )
        
        return (
            features,
            label,
            input_tokenized['input_ids'].squeeze(0),
            input_tokenized['attention_mask'].squeeze(0),
            target_tokenized['input_ids'].squeeze(0)
        )

    def create_temperature_input(self, row):
        """
        Convert temperature readings into a text prompt for T5 using your exact column format
        """
        if self.text_mode == 'minimal':
            return "Generate thermal assessment"
        
        elif self.text_mode == 'full':
            # Column names in order
            temp_columns = [
                'R1 int', 'L1 int', 'R2 int', 'L2 int', 'R3 int', 'L3 int', 'R4 int',
                'L4 int', 'R5 int', 'L5 int', 'R6 int', 'L6 int', 'R7 int', 'L7 int',
                'R8 int', 'L8 int', 'R9 int', 'L9 int', 'T1 int', 'T2 int', 'R0 int',
                'L0 int', 'R1 sk', 'L1 sk', 'R2 sk', 'L2 sk', 'R3 sk', 'L3 sk', 'R4 sk',
                'L4 sk', 'R5 sk', 'L5 sk', 'R6 sk', 'L6 sk', 'R7 sk', 'L7 sk', 'R8 sk',
                'L8 sk', 'R9 sk', 'L9 sk', 'T1 sk', 'T2 sk', 'R0 sk', 'L0 sk'
            ]
            
            # Create the temperature reading string
            temp_readings = []
            for col in temp_columns:
                value = row[col]
                # Clean up column name for display (remove spaces, make consistent)
                clean_name = col.replace(' ', '_')
                temp_readings.append(f"{clean_name}={value:.1f}")
            
            # Create the full input prompt
            input_text = "Generate thermal assessment from readings: " + ", ".join(temp_readings)
            return input_text


def create_weighted_sampler(dataset):
    """
    Create a weighted sampler to handle class imbalance.
    """
    labels = []
    for i in range(len(dataset)):
        _, label, _, _, _ = dataset[i]
        labels.append(label.item())
    
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# Training loop skeleton
def train(model, dataloader, optimizer, device):
    """
    Trains the multitask model on classification and text generation.

    Returns:
        tuple:
            - Average total loss
            - Average classification loss
            - Average generation loss (or None if no generation loss)
    """
    model.train()
    clf_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    total_clf_loss = 0
    total_gen_loss = 0
    total_gen_loss_count = 0  # To handle cases where gen_loss is None

    for features, labels, input_ids, attention_mask, target_ids in dataloader:
        features, labels = features.to(device), labels.to(device)
        input_ids, attention_mask, target_ids = (
            input_ids.to(device),
            attention_mask.to(device),
            target_ids.to(device),
        )

        optimizer.zero_grad()

        class_logits, gen_loss, gen_logits, clf_weight, gen_weight = model(
            features, input_ids, attention_mask, target_ids
        )
        clf_loss = clf_loss_fn(class_logits, labels)

        if gen_loss is not None:
            total_batch_loss = clf_weight * clf_loss + gen_weight * gen_loss
            total_gen_loss += gen_loss.item()
            total_gen_loss_count += 1
        else:
            total_batch_loss = clf_loss

        total_clf_loss += clf_loss.item()
        total_loss += total_batch_loss.item()

        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    avg_clf_loss = total_clf_loss / len(dataloader)
    avg_gen_loss = (
        total_gen_loss / total_gen_loss_count if total_gen_loss_count > 0 else None
    )

    return avg_loss, avg_clf_loss, avg_gen_loss



def evaluate(model, dataloader, device, threshold=0.5):
    """
   Evaluates a multi-task model on classification and generation tasks.
   
   This function performs evaluation on a model that handles both classification
   and text generation tasks. It computes classification metrics using a specified
   probability threshold and handles cases where generation loss may be None.
   
   Args:
       model: Multi-task model with classification and generation capabilities
       dataloader: DataLoader containing evaluation data with batches of:
           (features, labels, input_ids, attention_mask, target_ids)
       device: PyTorch device (CPU/GPU) for tensor operations
       threshold (float, optional): Probability threshold for binary classification.
           Defaults to 0.5.
   
   Returns:
       tuple: A 2-tuple containing:
           - avg_loss (float): Average total loss across all batches
           - metrics (dict): Dictionary containing evaluation metrics:
               - 'accuracy': Classification accuracy
               - 'f1_score': Weighted F1 score
               - 'sensitivity': Recall for positive class (true positive rate)
               - 'specificity': Recall for negative class (true negative rate)
               - 'auc_roc': Area under ROC curve (None if cannot be computed)
               - 'confusion_matrix': Confusion matrix as nested list
               - 'threshold': Threshold used for predictions
   
   Note:
       - Model is set to evaluation mode during execution
       - Generation loss is weighted and combined with classification loss when available
       - AUC-ROC may be None if all samples belong to one class
       - Uses softmax probabilities from class logits for threshold-based predictions
   """
    
    model.eval()
    clf_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    all_predictions = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for features, labels, input_ids, attention_mask, target_ids in dataloader:
            features, labels = features.to(device), labels.to(device)
            input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(device)

            class_logits, gen_loss, _, clf_weight, gen_weight = model(features, input_ids, attention_mask, target_ids)
            clf_loss = clf_loss_fn(class_logits, labels)

            if gen_loss is not None:
                clf_weight = clf_weight
                gen_weight = gen_weight
                batch_loss = clf_weight * clf_loss + gen_weight * gen_loss
            else:
                batch_loss = clf_loss

            total_loss += batch_loss.item()

            probs = torch.softmax(class_logits, dim=1)
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='weighted')
    sensitivity = recall_score(all_labels, preds, pos_label=1)
    specificity = recall_score(all_labels, preds, pos_label=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = None
    conf_matrix = confusion_matrix(all_labels, preds)

    return total_loss / len(dataloader), {
        'accuracy': accuracy,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc_roc': auc,
        'confusion_matrix': conf_matrix.tolist(),
        'threshold': threshold
    }

def evaluate_with_sampling(model, dataloader, device, tokenizer, threshold=0.5, text_sample_size=None):
    """
    Evaluate classification metrics over full dataset and METEOR + ROUGE-L on full dataset or sample.
    
    Args:
        text_sample_size: If None, evaluates on full dataset. If int, samples that many examples.
    """
    model.eval()
    clf_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    all_probs = []
    all_labels = []

    generated_texts = []
    reference_texts = []
    total_samples = len(dataloader.dataset)

    # CHANGE 2: Handle None case for full dataset evaluation
    if text_sample_size is None or text_sample_size >= total_samples:
        # Evaluate on ALL samples
        sample_indices = set(range(total_samples))
        print(f"Evaluating text generation on FULL dataset ({total_samples} samples)")
    else:
        # Original sampling behavior
        sample_indices = set(random.sample(range(total_samples), text_sample_size))
        print(f"Sampling {len(sample_indices)} out of {total_samples} samples for text generation metrics")

    current_idx = 0

    # REST OF THE FUNCTION STAYS EXACTLY THE SAME
    with torch.no_grad():
        for features, labels, input_ids, attention_mask, target_ids in dataloader:
            features, labels = features.to(device), labels.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)

            class_logits, gen_loss, _, clf_weight, gen_weight = model(features, input_ids, attention_mask, target_ids)
            clf_loss = clf_loss_fn(class_logits, labels)

            if gen_loss is not None:
                batch_loss = clf_weight * clf_loss + gen_weight * gen_loss
            else:
                batch_loss = clf_loss

            total_loss += batch_loss.item()

            probs = torch.softmax(class_logits, dim=1)
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_size = features.size(0)
            for i in range(batch_size):
                if current_idx + i in sample_indices:
                    sample_input_ids = input_ids[i:i+1]
                    sample_attention_mask = attention_mask[i:i+1]
                    sample_target_ids = target_ids[i:i+1]

                    generated_ids = model.t5.generate(
                        input_ids=sample_input_ids,
                        attention_mask=sample_attention_mask,
                        max_length=64,
                        num_beams=1,
                        do_sample=False,
                        #early_stopping=True
                    )

                    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    ref_text = tokenizer.decode(sample_target_ids[0], skip_special_tokens=True)
                    generated_texts.append(gen_text)
                    reference_texts.append(ref_text)

            current_idx += batch_size

    # Classification metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > threshold).astype(int)

    accuracy = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='weighted')
    sensitivity = recall_score(all_labels, preds, pos_label=1)
    specificity = recall_score(all_labels, preds, pos_label=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = None
    conf_matrix = confusion_matrix(all_labels, preds)

    # Text generation metrics
    meteor_mean_score = None
    rougeL_score = None

    if generated_texts and reference_texts:
        print(f"Computing text metrics on {len(generated_texts)} samples...")

        try:
            meteor_scores = [
                meteor_score([ref.lower().split()], gen.lower().split())
                for ref, gen in zip(reference_texts, generated_texts)
            ]
            meteor_mean_score = np.mean(meteor_scores)
        except Exception as e:
            print(f"Error calculating METEOR Score: {e}")

        try:
            rougeL_score = compute_rougeL(reference_texts, generated_texts)
        except Exception as e:
            print(f"Error calculating ROUGE-L Score: {e}")

    return total_loss / len(dataloader), {
        'accuracy': accuracy,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc_roc': auc,
        'confusion_matrix': conf_matrix.tolist(),
        'threshold': threshold,
        'avg_meteor': meteor_mean_score,
        'avg_rougeL_f1': rougeL_score,
        'text_samples_used': len(generated_texts),
        'total_samples': total_samples,
        'generated_texts': generated_texts,
        'reference_texts': reference_texts,
    }

def setup_training_pipeline(
    df_train, df_val, df_test,
    multitask_model_class,
    multitask_dataset_class,
    tokenizer_path='./t5-small-local/',
    batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    device_override=None,
    text_mode='full'  # New parameter: 'full' or 'minimal'
):
    """
    Set up tokenizer, datasets, dataloaders, model, and optimizer for multitask training.
    
    Args:
        df_train, df_val, df_test: DataFrames for training, validation, and testing
        multitask_model_class: The model class to instantiate
        multitask_dataset_class: The dataset class to instantiate
        tokenizer_path: Path to the tokenizer
        batch_size: Batch size for dataloaders
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        device_override: Override device selection
        text_mode: Text input mode - 'full' (with temperature readings) or 'minimal' (simple prompt)
    
    Returns:
        tokenizer, train_loader, val_loader, test_loader, model, optimizer, device
    """
    print("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    
    print(f"Creating datasets with text_mode='{text_mode}'...")
    train_dataset = multitask_dataset_class(df_train, tokenizer, text_mode=text_mode)
    val_dataset = multitask_dataset_class(df_val, tokenizer, text_mode=text_mode)
    test_dataset = multitask_dataset_class(df_test, tokenizer, text_mode=text_mode)
    
    print("Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Setting up device...")
    device = torch.device(device_override if device_override else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    print("Initializing model...")
    model = multitask_model_class(num_classes=2, t5_model_name=tokenizer_path)
    print(f"Initial weights - CLF: {model.classification_weight}, GEN: {model.generation_weight}")
    
    print("Moving model to device...")
    model = model.to(device)
    
    print("Setting up optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    print(f"After .to(device) - CLF: {model.classification_weight}, GEN: {model.generation_weight}")
    print(f"✓ Setup complete with text_mode='{text_mode}'!\n")
    
    return tokenizer, train_loader, val_loader, test_loader, model, optimizer, device


def train_and_validate_model_with_meteor_rouge(
    model, train_loader, val_loader, optimizer, device, tokenizer, num_epochs=30, meteor_every=5
):
    """
    Trains and evaluates the model across epochs with periodic METEOR and ROUGE-L scoring.
    Now includes plotting of training curves.
    
    Args:
        meteor_every (int): Calculate METEOR and ROUGE every N epochs to save computation
    """
    import nltk
    from nltk.translate.meteor_score import meteor_score
    
    # Initialize tracking variables
    best_accuracy = 0
    best_f1 = 0
    best_auc = 0
    best_meteor = 0
    best_rougeL = 0
    
    # Lists to store metrics for plotting
    epochs_list = []
    train_losses = []
    val_losses = []
    clf_losses = []
    gen_losses = []
    accuracies = []
    f1_scores = []
    aucs = []
    
    # Lists for text generation metrics (only updated every meteor_every epochs)
    meteor_epochs = []
    meteor_scores = []
    rouge_scores = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, clf_loss, gen_loss = train(model, train_loader, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")
        print(f"Classification loss: {clf_loss:.4f}")
        if gen_loss is not None:
            print(f"Generation loss: {gen_loss:.4f}")
            print(f"Weighted gen loss: {model.generation_weight * gen_loss:.6f}")
        
        # Regular validation (fast)
        val_loss, val_metrics = evaluate(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}")
        print("Validation metrics:")
        print(f"  Accuracy     : {val_metrics['accuracy']:.4f}")
        print(f"  F1-Score     : {val_metrics['f1_score']:.4f}")
        print(f"  Sensitivity  : {val_metrics['sensitivity']:.4f}")
        print(f"  Specificity  : {val_metrics['specificity']:.4f}")
        if val_metrics['auc_roc'] is not None:
            print(f"  AUC-ROC      : {val_metrics['auc_roc']:.4f}")
        else:
            print("  AUC-ROC      : N/A")
        
        # Store metrics for plotting
        epochs_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        clf_losses.append(clf_loss)
        gen_losses.append(gen_loss if gen_loss is not None else 0)
        accuracies.append(val_metrics['accuracy'])
        f1_scores.append(val_metrics['f1_score'])
        aucs.append(val_metrics['auc_roc'] if val_metrics['auc_roc'] is not None else 0)
        
        # METEOR + ROUGE evaluation every N epochs
        if (epoch + 1) % meteor_every == 0:
            print(f"\n--- Text Evaluation (Epoch {epoch + 1}) ---")
            val_loss_sampled, val_metrics_sampled = evaluate_with_sampling(
                model, val_loader, device, tokenizer, text_sample_size=None
            )
            
            meteor_epochs.append(epoch + 1)
            
            # METEOR
            if val_metrics_sampled['avg_meteor'] is not None:
                current_meteor = val_metrics_sampled['avg_meteor']
                meteor_scores.append(current_meteor)
                print(f"  METEOR Score : {current_meteor:.4f}")
                if current_meteor > best_meteor:
                    best_meteor = current_meteor
                    print(f"  ✓ New best METEOR: {best_meteor:.4f}")
            else:
                meteor_scores.append(0)
                print("  METEOR Score : N/A")
            
            # ROUGE-L
            if val_metrics_sampled.get('avg_rougeL_f1') is not None:
                current_rougeL = val_metrics_sampled['avg_rougeL_f1']
                rouge_scores.append(current_rougeL)
                print(f"  ROUGE-L F1   : {current_rougeL:.4f}")
                if current_rougeL > best_rougeL:
                    best_rougeL = current_rougeL
                    print(f"  ✓ New best ROUGE-L F1: {best_rougeL:.4f}")
            else:
                rouge_scores.append(0)
                print("  ROUGE-L F1   : N/A")
        
        # Track best classification metrics
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            print(f"✓ New best accuracy: {best_accuracy:.4f}")
        if val_metrics['f1_score'] > best_f1:
            best_f1 = val_metrics['f1_score']
            print(f"✓ New best F1: {best_f1:.4f}")
        if val_metrics['auc_roc'] is not None and val_metrics['auc_roc'] > best_auc:
            best_auc = val_metrics['auc_roc']
            print(f"✓ New best AUC-ROC: {best_auc:.4f}")
        
        if (epoch + 1) % 3 == 0:
            print(f"Current loss weights - CLF: {model.classification_weight:.3f}, GEN: {model.generation_weight:.3f}")
    
    # Create comprehensive plots
    plot_training_curves(
        epochs_list, train_losses, val_losses, clf_losses, gen_losses,
        accuracies, f1_scores, aucs, meteor_epochs, meteor_scores, rouge_scores
    )
    
    return best_accuracy, best_f1, best_auc, best_meteor, best_rougeL

def plot_training_curves(epochs, train_losses, val_losses, clf_losses, gen_losses,
                        accuracies, f1_scores, aucs, meteor_epochs, meteor_scores, rouge_scores):
    """
    Create comprehensive training curve plots for multitask model
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multitask Model Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Overall Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Task-specific losses
    ax2 = axes[0, 1]
    ax2.plot(epochs, clf_losses, 'g-', label='Classification Loss', linewidth=2)
    ax2.plot(epochs, gen_losses, 'm-', label='Generation Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Task-Specific Losses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Classification metrics
    ax3 = axes[0, 2]
    ax3.plot(epochs, accuracies, 'b-', label='Accuracy', linewidth=2)
    ax3.plot(epochs, f1_scores, 'r-', label='F1-Score', linewidth=2)
    # Only plot AUC if we have valid values
    valid_aucs = [auc for auc in aucs if auc > 0]
    if valid_aucs:
        ax3.plot(epochs, aucs, 'g-', label='AUC-ROC', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Classification Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 4. Text generation metrics
    ax4 = axes[1, 0]
    if meteor_scores and any(score > 0 for score in meteor_scores):
        ax4.plot(meteor_epochs, meteor_scores, 'o-', color='orange', 
                label='METEOR', linewidth=2, markersize=6)
    if rouge_scores and any(score > 0 for score in rouge_scores):
        ax4.plot(meteor_epochs, rouge_scores, 's-', color='purple', 
                label='ROUGE-L F1', linewidth=2, markersize=6)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.set_title('Text Generation Metrics')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # 5. Loss comparison (normalized)
    ax5 = axes[1, 1]
    # Normalize losses for comparison
    if max(train_losses) > 0:
        norm_train = np.array(train_losses) / max(train_losses)
        ax5.plot(epochs, norm_train, 'b-', label='Train Loss (norm)', linewidth=2)
    
    if max(clf_losses) > 0:
        norm_clf = np.array(clf_losses) / max(clf_losses)
        ax5.plot(epochs, norm_clf, 'g--', label='CLF Loss (norm)', linewidth=2)
    
    if max(gen_losses) > 0:
        norm_gen = np.array(gen_losses) / max(gen_losses)
        ax5.plot(epochs, norm_gen, 'm--', label='GEN Loss (norm)', linewidth=2)
    
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Normalized Loss')
    ax5.set_title('Normalized Loss Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance summary
    ax6 = axes[1, 2]
    
    # Create a summary table
    final_metrics = {
        'Accuracy': f"{accuracies[-1]:.3f}",
        'F1-Score': f"{f1_scores[-1]:.3f}",
        'AUC-ROC': f"{aucs[-1]:.3f}" if aucs[-1] > 0 else "N/A",
        'METEOR': f"{meteor_scores[-1]:.3f}" if meteor_scores and meteor_scores[-1] > 0 else "N/A",
        'ROUGE-L': f"{rouge_scores[-1]:.3f}" if rouge_scores and rouge_scores[-1] > 0 else "N/A"
    }
    
    # Display as text
    ax6.axis('off')
    ax6.text(0.1, 0.9, 'Final Performance:', fontsize=14, fontweight='bold', transform=ax6.transAxes)
    
    y_pos = 0.75
    for metric, value in final_metrics.items():
        ax6.text(0.1, y_pos, f'{metric}: {value}', fontsize=12, transform=ax6.transAxes)
        y_pos -= 0.12
    
    # Add training info
    ax6.text(0.1, 0.2, f'Total Epochs: {len(epochs)}', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.1, 0.1, f'Final Train Loss: {train_losses[-1]:.4f}', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.1, 0.0, f'Final Val Loss: {val_losses[-1]:.4f}', fontsize=10, transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Best Accuracy: {max(accuracies):.4f}")
    print(f"Best F1-Score: {max(f1_scores):.4f}")
    print(f"Best AUC-ROC: {max(aucs):.4f}" if max(aucs) > 0 else "Best AUC-ROC: N/A")
    if meteor_scores:
        print(f"Best METEOR: {max(meteor_scores):.4f}")
    if rouge_scores:
        print(f"Best ROUGE-L: {max(rouge_scores):.4f}")
    print("="*60)

import copy
import torch
import nltk
from nltk.translate.meteor_score import meteor_score

class METEOREarlyStopping:
    """
    Early stopping for METEOR scores with model state management.
    """
    def __init__(self, patience=3, min_delta=0.01, restore_best=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.verbose = verbose
        
        self.best_score = None
        self.best_model_state = None
        self.counter = 0
        self.early_stop = False
        self.generation_frozen = False
        
    def __call__(self, score, model):
        """
        Check if early stopping should be triggered.

        Args:
            score: Current METEOR score
            model: The model to potentially save state from

        Returns:
            bool: True if early stopping should be triggered
        """
        if self.generation_frozen:
            return True

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
            if self.verbose:
                print(f"  Initial METEOR baseline: {score:.4f}")
            return False

        if score > self.best_score:
            # Actual improvement — update best
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
            if self.verbose:
                print(f"✓ METEOR improved to: {score:.4f}")
        elif self.best_score - score <= self.min_delta:
            # Slight drop (within tolerance) — don't count as bad
            if self.verbose:
                print(f"  METEOR change within tolerance (best: {self.best_score:.4f}, current: {score:.4f})")
        else:
            # Significant drop — count toward early stopping
            self.counter += 1
            if self.verbose:
                print(f"  No METEOR improvement ({self.counter}/{self.patience}) — current: {score:.4f}, best: {self.best_score:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered for generation component!")
                return True

        return False

    
    def _save_checkpoint(self, model):
        """Save the current model state."""
        if self.restore_best:
            # Save only text generation-related parameters
            self.best_model_state = {}
            for name, param in model.named_parameters():
                if 't5' in name or 'feature_projection' in name:
                    self.best_model_state[name] = param.data.clone()

    def restore_and_freeze_generation(self, model):
        """
        Restore best generation weights and freeze generation parameters.
        """
        if self.best_model_state is not None and self.restore_best:
            if self.verbose:
                print("Restoring best generation weights...")
        
            for name, param in model.named_parameters():
                if name in self.best_model_state:
                    param.data.copy_(self.best_model_state[name])
    
        # Freeze generation parameters (T5 and feature projection)
        frozen_count = 0
        for name, param in model.named_parameters():
            if 't5' in name or 'feature_projection' in name:
                param.requires_grad = False
                frozen_count += 1
    
        self.generation_frozen = True
        if self.verbose:
            print(f"Froze {frozen_count} generation parameters")
            
        old_clf_weight = model.classification_weight
        model.classification_weight = 1.0
        model.generation_weight = 0.0
    
        if self.verbose:
            print(f"Classification weight: {old_clf_weight:.1f} → 1.0")
            print("Switched to classification-only training")
        
    def should_evaluate_meteor(self):
        """Check if METEOR evaluation should continue."""
        return not self.generation_frozen
    
    def should_train_generation(self):
        """Check if generation component should still be trained."""
        return not self.generation_frozen

def train_and_validate_model_with_meteor_stopping(
    model, train_loader, val_loader, optimizer, device, tokenizer, 
    num_epochs=30, meteor_every=5, patience=3, min_delta=0.01
):
    """
    Trains and evaluates the model with robust METEOR-based early stopping.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        tokenizer: Tokenizer for text generation
        num_epochs: Maximum number of epochs
        meteor_every: Calculate METEOR every N epochs
        patience: Early stopping patience (in METEOR evaluation cycles)
        min_delta: Minimum improvement threshold for METEOR
    
    Returns:
        dict: Training results including best metrics
    """
    # Initialize early stopping
    early_stopper = METEOREarlyStopping(
        patience=patience, 
        min_delta=min_delta, 
        restore_best=True,
        verbose=True
    )
    
    # Track best metrics
    best_metrics = {
        'accuracy': 0,
        'f1_score': 0,
        'auc_roc': 0,
        'meteor': 0,
        'rougeL_f1': 0
    }
    
    print(f"Starting training with METEOR early stopping (patience={patience}, min_delta={min_delta})")
    print(f"METEOR evaluation every {meteor_every} epochs")
    print(f"GEN weight: {model.generation_weight}, CLASS weight: {model.classification_weight}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        try:
            train_loss, clf_loss, gen_loss = train(model, train_loader, optimizer, device)
            print(f"Training loss: {train_loss:.4f}")
            print(f"Classification loss: {clf_loss:.4f}")
            
            if gen_loss is not None:
                gen_status = "frozen" if early_stopper.generation_frozen else "active"
                print(f"Generation loss: {gen_loss:.4f} ({gen_status})")
                print(f"Generation weighting: {gen_loss}*{model.generation_weight}")
                    
        except Exception as e:
            print(f"Training failed at epoch {epoch + 1}: {e}")
            break
        
        # Validation phase
        try:
            val_loss, val_metrics = evaluate(model, val_loader, device)
            print(f"Validation loss: {val_loss:.4f}")
            print("Validation metrics:")
            print(f"  Accuracy     : {val_metrics['accuracy']:.4f}")
            print(f"  F1-Score     : {val_metrics['f1_score']:.4f}")
            print(f"  Sensitivity  : {val_metrics['sensitivity']:.4f}")
            print(f"  Specificity  : {val_metrics['specificity']:.4f}")
            
            if val_metrics.get('auc_roc') is not None:
                print(f"  AUC-ROC      : {val_metrics['auc_roc']:.4f}")
            else:
                print("  AUC-ROC      : N/A")
                
        except Exception as e:
            print(f"Validation failed at epoch {epoch + 1}: {e}")
            continue
        
        # METEOR evaluation and early stopping
        if (epoch + 1) % meteor_every == 0:
            print(f"\n--- METEOR Evaluation (Epoch {epoch + 1}) ---")
            
            try:
                _, val_metrics_sampled = evaluate_with_sampling(
                    model, val_loader, device, tokenizer, text_sample_size=None
                )
                
                current_meteor = val_metrics_sampled.get("avg_meteor")
                current_meteor = val_metrics_sampled.get("avg_meteor")
                current_rouge = val_metrics_sampled.get("avg_rougeL_f1")
                
                if current_meteor is not None:
                    print(f"  METEOR Score : {current_meteor:.4f}")
                    
                    # Check for early stopping
                    should_stop = early_stopper(current_meteor, model)
                    
                    # Update best METEOR
                    if current_meteor > best_metrics['meteor']:
                        best_metrics['meteor'] = current_meteor
                        
                if current_rouge is not None:
                    print(f"  ROUGE-L F1   : {current_rouge:.4f}")
                    if current_rouge > best_metrics['rougeL_f1']:
                            best_metrics['rougeL_f1'] = current_rouge
                    
                    # Apply early stopping if triggered
                    if should_stop and not early_stopper.generation_frozen:
                        early_stopper.restore_and_freeze_generation(model)
                        print("  Generation training stopped. Classification continues.")
                        
                else:
                    print("  ROGUE-L F1 : N/A (evaluation failed)")
                    
            except Exception as e:
                print(f"  METEOR evaluation failed: {e}")
        
        # Update best classification metrics
        if val_metrics['accuracy'] > best_metrics['accuracy']:
            best_metrics['accuracy'] = val_metrics['accuracy']
            print(f"✓ New best accuracy: {best_metrics['accuracy']:.4f}")
            
        if val_metrics['f1_score'] > best_metrics['f1_score']:
            best_metrics['f1_score'] = val_metrics['f1_score']
            print(f"✓ New best F1: {best_metrics['f1_score']:.4f}")
            
        if val_metrics.get('auc_roc') is not None and val_metrics['auc_roc'] > best_metrics['auc_roc']:
            best_metrics['auc_roc'] = val_metrics['auc_roc']
            print(f"✓ New best AUC-ROC: {best_metrics['auc_roc']:.4f}")
        
        # Log current state
        if (epoch + 1) % 3 == 0:
            gen_status = "frozen" if early_stopper.generation_frozen else f"{model.generation_weight:.3f}"
            print(f"Current state - CLF weight: {model.classification_weight:.3f}, "
                  f"GEN weight: {gen_status}")
    
    # Final summary
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Best metrics achieved:")
    print(f"  Accuracy : {best_metrics['accuracy']:.4f}")
    print(f"  F1-Score : {best_metrics['f1_score']:.4f}")
    print(f"  AUC-ROC  : {best_metrics['auc_roc']:.4f}")
    print(f"  METEOR   : {best_metrics['meteor']:.4f}")
    print(f"  ROUGE-L  : {best_metrics['rougeL_f1']:.4f}")
    
    if early_stopper.generation_frozen:
        print(f"Generation training stopped early after {early_stopper.counter} cycles without improvement")
    
    return best_metrics['accuracy'], best_metrics['f1_score'], best_metrics['auc_roc'], best_metrics['meteor']

def display_generated_texts(generated_texts, reference_texts, num_samples=5):
    """
    Display a few examples of generated texts alongside their reference texts.

    Args:
        generated_texts (list): List of generated text strings.
        reference_texts (list): List of reference (target) text strings.
        num_samples (int): Number of examples to display.
    """
    assert len(generated_texts) == len(reference_texts), "Mismatched list lengths."

    sample_size = min(num_samples, len(generated_texts))
    sample_indices = random.sample(range(len(generated_texts)), sample_size)

    print(f"\nDisplaying {sample_size} randomly selected text generation samples:\n")
    for idx in sample_indices:
        print(f"Sample {idx + 1}")
        print("-" * 40)
        print(f"Generated : {generated_texts[idx]}")
        print(f"Reference : {reference_texts[idx]}")
        print("-" * 40 + "\n")
        
        
def print_test_evaluation(model, test_loader, device, tokenizer, test_simple_df):
    print(f"\nTraining completed!")
    print(f"Best results from validation:")
    # Assuming best_acc, best_f1, best_auc are tracked elsewhere, you might want to pass them as arguments if needed
    
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)

    # Quick classification metrics on full test set
    print("\n1. Full Classification Performance:")
    test_loss, test_metrics = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Classification metrics (all {len(test_simple_df)} samples):")
    print(f"  Accuracy     : {test_metrics['accuracy']:.4f}")
    print(f"  F1-Score     : {test_metrics['f1_score']:.4f}")
    print(f"  Sensitivity  : {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity  : {test_metrics['specificity']:.4f}")
    if test_metrics['auc_roc'] is not None:
        print(f"  AUC-ROC      : {test_metrics['auc_roc']:.4f}")
    else:
        print("  AUC-ROC      : N/A")

    # Sampled text generation metrics
    print("\n2. Sampled Text Generation Performance:")
    # Use 1000 samples for text metrics (adjust as needed)
    test_loss_sampled, test_metrics_sampled = evaluate_with_sampling(
        model, test_loader, device, tokenizer, text_sample_size=None
    )

    print(f"Text generation metrics ({test_metrics_sampled['text_samples_used']} samples):")
    if test_metrics_sampled.get('avg_rougeL_f1') is not None:
        print(f"  ROUGE-L F1   : {test_metrics_sampled['avg_rougeL_f1']:.4f}")
    else:
        print(f"  ROUGE-L F1   : N/A")

    if test_metrics_sampled.get('avg_meteor') is not None:
        print(f"  METEOR       : {test_metrics_sampled['avg_meteor']:.4f}")
    else:
        print(f"  METEOR       : N/A")

    print(f"\nSample efficiency: {test_metrics_sampled['text_samples_used']}/{test_metrics_sampled['total_samples']} samples used for text metrics")
    
    return test_loss_sampled, test_metrics_sampled
    
def compute_rougeL(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_score = 0.0
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        total_score += score['rougeL'].fmeasure
    return total_score / len(predictions)