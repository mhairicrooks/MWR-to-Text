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
import torch.nn as nn # neural network layers and modules
import torch.nn.init as init
import torch.optim as optim # optimisers
from torch.utils.data import DataLoader, Dataset # data batching tools
from transformers import T5Tokenizer, T5ForConditionalGeneration # huggingface T5 model + tokenizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix, classification_report # sklearn metrics for evaluation
import torchmetrics # extra metrics
import random


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
    A multitask model combining structured data classification and
    T5-based text generation with proper shared encoder architecture.

    Architecture:
        temperature readings → shared encoder → embedding
                                               ├── classification branch
                                               └── T5 text generation branch

    Args:
        num_features (int): Number of input features.
        embedding_dim (int): Dimension of the intermediate embedding.
        num_classes (int): Number of classification categories.
        t5_model_name (str): HuggingFace model name for the T5 generator.
    """
    def __init__(self, num_features=44, embedding_dim=512, num_classes=6, t5_model_name='t5-small'):
        super().__init__()
        
        # Shared encoder - maps temperature readings to embedding space
        #self.encoder = nn.Sequential(
        #    nn.Linear(num_features, 128),
        #    nn.BatchNorm1d(128),
        #    nn.ReLU(),
        #    nn.Dropout(0.2),
        #    nn.Linear(128, embedding_dim),
        #   nn.BatchNorm1d(embedding_dim),
        #    nn.ReLU(),
        #    nn.Dropout(0.1)
        #)
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features),
            GaussianNoise(0.0, 0.2),
            nn.Linear(num_features, 1000),  # Same as original first layer
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.2),
            nn.Linear(1000, 512),          # Larger embedding
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        
        # Classification branch
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),  # Just 256
            GaussianNoise(0.0, 0.2),
            nn.Linear(embedding_dim, 1000),  # 256 → 1000
            nn.ReLU(),
            
            #nn.BatchNorm1d(1000, momentum=0.99, eps=1e-5),
            nn.BatchNorm1d(1000, eps=1e-5),
            nn.Dropout(0.2),
            GaussianNoise(mean=0.0, std=0.2),
            nn.Linear(1000, 200),
            nn.ReLU(),
            
            #nn.BatchNorm1d(200, momentum=0.99, eps=1e-5),
            nn.BatchNorm1d(200, eps=1e-5),
            nn.Dropout(0.2),
            GaussianNoise(mean=0.0, std=0.2),
            nn.Linear(200, 200),
            nn.ReLU(),
            
            #nn.BatchNorm1d(200, momentum=0.99, eps=1e-5),
            nn.BatchNorm1d(200, eps=1e-5),
            nn.Dropout(0.2),
            GaussianNoise(mean=0.0, std=0.2),
            nn.Linear(200, 200),
            nn.ReLU(),
            
            #nn.BatchNorm1d(200, momentum=0.99, eps=1e-5),
            nn.BatchNorm1d(200, eps=1e-5),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.ReLU(),
            
            nn.Linear(200, num_classes)
        )
        
        # T5 model for text generation
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name, local_files_only=True)
        
        # feature projection layer to inject temperature features into T5
        self.feature_projection = nn.Linear(embedding_dim, self.t5.config.d_model)
        
        # freeze T5 encoder to reduce computation/memory
        for param in self.t5.encoder.parameters():
            param.requires_grad = False
            
        # Loss weights
        self.classification_weight = 1.0
        self.generation_weight = 0.001
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize feature projection layer
        nn.init.xavier_uniform_(self.feature_projection.weight)
        if self.feature_projection.bias is not None:
            nn.init.zeros_(self.feature_projection.bias)
    
    def forward(self, features, input_ids=None, attention_mask=None, labels=None):
        # Pass temperature readings through shared encoder
        embedding = self.encoder(features)
        
        # new
        # Concatenate original features with embedding
        #combined_input = torch.cat([features, embedding], dim=1)
        #class_logits = self.classifier(combined_input)
        
        
        # Classification branch
        class_logits = self.classifier(embedding)
        
        # Text generation branch, using shared features
        if input_ids is not None and labels is not None:
            # Project feature embedding to T5's embedding dimension
            feature_embedding = self.feature_projection(embedding)  # [batch_size, d_model]
            
            # Get T5's input embeddings
            input_embeddings = self.t5.encoder.embed_tokens(input_ids)  # [batch_size, seq_len, d_model]
            
            # Add feature information to input embeddings
            # Option 1: Add to all positions (broadcasting)
            enhanced_embeddings = input_embeddings + feature_embedding.unsqueeze(1)
            
            # Forward through T5 with enhanced embeddings
            t5_output = self.t5(
                inputs_embeds=enhanced_embeddings,
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

def get_parameter_groups(model):
    # Shared encoder parameters
    encoder_params = list(model.encoder.parameters())
    
    # Classification branch parameters
    classification_params = list(model.classifier.parameters()) + list(model.feature_projection.parameters())
    
    # T5 generation parameters (only unfrozen decoder parts)
    generation_params = list(model.t5.decoder.parameters())
    # Note: T5 encoder is frozen, so it is skipped
    
    return encoder_params, classification_params, generation_params

# Training loop skeleton
def train(model, dataloader, optimizer, device):
    """
    Trains the multitask model on classification and text generation.

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

        class_logits, gen_loss, gen_logits, clf_weight, gen_weight = model(features, input_ids, attention_mask, target_ids) # forward pass
        clf_loss = clf_loss_fn(class_logits, labels) # compute classification loss

        if gen_loss is not None:
            clf_weight = clf_weight
            gen_weight = gen_weight
            total_batch_loss = clf_weight * clf_loss + gen_weight * gen_loss
        else:
            total_batch_loss = clf_loss
    
        total_batch_loss.backward() # backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # update parameters

        total_loss += total_batch_loss.item() # track loss

    return total_loss / len(dataloader) # return average loss


def evaluate(model, dataloader, device, threshold=0.5):
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

def evaluate_with_sampling(model, dataloader, device, tokenizer, threshold=0.5, text_sample_size=500):
    """
    Evaluate with full classification metrics but sampled text generation metrics.
    
    Args:
        text_sample_size: Number of samples to use for BERT/METEOR computation
    """
    model.eval()
    clf_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    # For text generation sampling
    generated_texts = []
    reference_texts = []
    sample_indices = set()
    total_samples = len(dataloader.dataset)
    
    # Randomly select indices for text generation
    if text_sample_size < total_samples:
        sample_indices = set(random.sample(range(total_samples), text_sample_size))
        print(f"Sampling {text_sample_size} out of {total_samples} samples for text generation metrics")
    else:
        sample_indices = set(range(total_samples))
        print(f"Using all {total_samples} samples for text generation metrics")
    
    current_idx = 0
    
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
            
            # Classification metrics for ALL samples
            probs = torch.softmax(class_logits, dim=1)
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Text generation for SAMPLED samples only
            batch_size = features.size(0)
            for i in range(batch_size):
                if current_idx + i in sample_indices:
                    # Generate text for this sample
                    sample_input_ids = input_ids[i:i+1]
                    sample_attention_mask = attention_mask[i:i+1]
                    sample_target_ids = target_ids[i:i+1]
                    
                    generated_ids = model.t5.generate(
                        input_ids=sample_input_ids,
                        attention_mask=sample_attention_mask,
                        max_length=64,
                        do_sample=False,  # Greedy for speed
                        early_stopping=True
                    )
                    
                    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    ref_text = tokenizer.decode(sample_target_ids[0], skip_special_tokens=True)
                    generated_texts.append(gen_text)
                    reference_texts.append(ref_text)
            
            current_idx += batch_size
    
    # Classification metrics (full dataset)
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
    
    # Text generation metrics (sampled)
    bert_f1_score = None
    meteor_mean_score = None
    
    if generated_texts and reference_texts:
        print(f"Computing text metrics on {len(generated_texts)} samples...")
        
        # BERT Score
        try:
            from bert_score import score as bert_score
            P, R, F1 = bert_score(
                generated_texts,
                reference_texts,
                model_type="/home/s2080063/MWR-to-Text/models/roberta-large",  # local path here
                lang="en",
                verbose=False
            )
            bert_f1_score = F1.mean().item()
        except Exception as e:
            print(f"Error calculating BERT Score: {e}")
        
        # METEOR Score
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score
            nltk.data.path.append('/home/s2080063/nltk_data')
            meteor_scores = []
            for gen, ref in zip(generated_texts, reference_texts):
                gen_tokens = nltk.word_tokenize(gen.lower())
                ref_tokens = nltk.word_tokenize(ref.lower())
                meteor_scores.append(meteor_score([ref_tokens], gen_tokens))
            meteor_mean_score = np.mean(meteor_scores)
        except Exception as e:
            print(f"Error calculating METEOR Score: {e}")
    
    return total_loss / len(dataloader), {
        'accuracy': accuracy,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc_roc': auc,
        'confusion_matrix': conf_matrix.tolist(),
        'threshold': threshold,
        'avg_bertscore_f1': bert_f1_score,
        'avg_meteor': meteor_mean_score,
        'text_samples_used': len(generated_texts),
        'total_samples': total_samples
    }

def train_and_validate_model(
    model, train_loader, val_loader, optimizer, device, num_epochs=30
):
    """
    Trains and evaluates the model across epochs.

    Returns:
        best_accuracy, best_f1, best_auc
    """
    best_accuracy = 0
    best_f1 = 0
    best_auc = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train(model, train_loader, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Validate
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
        print(f"  Confusion Matrix: {val_metrics['confusion_matrix']}")

        # Track best metrics
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            print(f"✓ New best accuracy: {best_accuracy:.4f}")

        if val_metrics['f1_score'] > best_f1:
            best_f1 = val_metrics['f1_score']
            print(f"✓ New best F1: {best_f1:.4f}")

        if val_metrics['auc_roc'] is not None and val_metrics['auc_roc'] > best_auc:
            best_auc = val_metrics['auc_roc']
            print(f"✓ New best AUC-ROC: {best_auc:.4f}")

        # Periodic logging of dynamic loss weights
        if (epoch + 1) % 3 == 0:
            print(f"Current loss weights - CLF: {model.classification_weight:.3f}, GEN: {model.generation_weight:.3f}")

    return best_accuracy, best_f1, best_auc

def setup_training_pipeline(
    df_train, df_val, df_test,
    multitask_model_class,
    multitask_dataset_class,
    tokenizer_path='./t5-small-local/',
    batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    device_override=None
):
    """
    Set up tokenizer, datasets, dataloaders, model, and optimizer for multitask training.

    Returns:
        tokenizer, train_loader, val_loader, test_loader, model, optimizer, device
    """

    print("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    print("Creating datasets...")
    train_dataset = multitask_dataset_class(df_train, tokenizer)
    val_dataset = multitask_dataset_class(df_val, tokenizer)
    test_dataset = multitask_dataset_class(df_test, tokenizer)

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
    print("✓ Setup complete!\n")

    return tokenizer, train_loader, val_loader, test_loader, model, optimizer, device
