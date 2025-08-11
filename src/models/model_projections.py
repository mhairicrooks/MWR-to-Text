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
    
class FeatureProjector(nn.Module):
    """
    Projects structured features into T5's embedding space for injection
    """
    def __init__(self, feature_dim=512, t5_hidden_dim=512, projection_type='linear'):
        super().__init__()
        self.projection_type = projection_type
        
        if projection_type == 'linear':
            self.projector = nn.Linear(feature_dim, t5_hidden_dim)
        elif projection_type == 'mlp':
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, t5_hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(t5_hidden_dim * 2, t5_hidden_dim)
            )
        elif projection_type == 'attention':
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, t5_hidden_dim),
                nn.LayerNorm(t5_hidden_dim)
            )
            self.attention = nn.MultiheadAttention(t5_hidden_dim, num_heads=8, batch_first=True)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
    
    def forward(self, features, t5_embeddings=None):
        """
        Args:
            features: [batch_size, feature_dim] - structured features from encoder
            t5_embeddings: [batch_size, seq_len, hidden_dim] - T5 input embeddings (for attention)
        
        Returns:
            projected_features: [batch_size, 1, t5_hidden_dim] or modified t5_embeddings
        """
        if self.projection_type == 'attention' and t5_embeddings is not None:
            # Project features and use as query to attend to T5 embeddings
            projected = self.projector(features)  # [batch, t5_hidden_dim]
            projected = projected.unsqueeze(1)     # [batch, 1, t5_hidden_dim]
            
            # Attention between projected features and T5 embeddings
            attended, _ = self.attention(projected, t5_embeddings, t5_embeddings)
            return attended  # [batch, 1, t5_hidden_dim]
        else:
            # Simple projection
            projected = self.projector(features)  # [batch, t5_hidden_dim]
            return projected.unsqueeze(1)         # [batch, 1, t5_hidden_dim]

class MultitaskModelWithProjection(nn.Module):
    """
    Multitask model with configurable feature projection strategies
    
    Args:
        projection_mode: str, one of:
            - 'none': No feature projection (original approach)
            - 'prefix': Prepend projected features as prefix tokens
            - 'concat': Concatenate projected features with input embeddings
            - 'attention': Use attention mechanism to blend features with text
    """
    def __init__(self, num_features=44, embedding_dim=512, num_classes=6, 
                 t5_model_name='t5-small', projection_mode='none', 
                 projection_type='linear', use_minimal_text=False):
        super().__init__()
        
        self.projection_mode = projection_mode
        self.use_minimal_text = use_minimal_text
        
        # Shared encoder - maps temperature readings to embedding space
        self.encoder = nn.Sequential(
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
        
        # Classification branch
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            GaussianNoise(0.0, 0.2),
            nn.Linear(embedding_dim, 1000),  
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
        
        # T5 model for text generation
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name, local_files_only=True)
        
        # Feature projection (if needed)
        if projection_mode != 'none':
            t5_hidden_dim = self.t5.config.d_model
            self.feature_projection = FeatureProjector(
                feature_dim=embedding_dim,
                t5_hidden_dim=t5_hidden_dim,
                projection_type=projection_type
            )
        else:
            self.feature_projection = None
            
        # Loss weights
        self.classification_weight = 0.1
        self.generation_weight = 10
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        if self.feature_projection is not None:
            for m in self.feature_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _inject_features_into_t5(self, features, input_ids, attention_mask):
        """
        Inject projected features into T5 based on projection_mode
        """
        batch_size = features.shape[0]
        
        if self.projection_mode == 'none':
            # No injection, use T5 normally
            return input_ids, attention_mask
            
        elif self.projection_mode == 'prefix':
            # Add projected features as prefix tokens
            t5_embeddings = self.t5.encoder.embed_tokens(input_ids)
            projected_features = self.feature_projection(features)  # [batch, 1, hidden_dim]
            
            # Concatenate projected features at the beginning
            combined_embeddings = torch.cat([projected_features, t5_embeddings], dim=1)
            
            # Extend attention mask
            feature_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            extended_attention_mask = torch.cat([feature_mask, attention_mask], dim=1)
            
            return combined_embeddings, extended_attention_mask, True  # True indicates we're passing embeddings
            
        elif self.projection_mode == 'concat':
            # Average pool projected features and add to each token
            t5_embeddings = self.t5.encoder.embed_tokens(input_ids)
            projected_features = self.feature_projection(features)  # [batch, 1, hidden_dim]
            
            # Broadcast and add to all tokens
            projected_features = projected_features.expand(-1, t5_embeddings.shape[1], -1)
            combined_embeddings = t5_embeddings + projected_features
            
            return combined_embeddings, attention_mask, True
            
        elif self.projection_mode == 'attention':
            # Use attention to blend features with text embeddings
            t5_embeddings = self.t5.encoder.embed_tokens(input_ids)
            attended_features = self.feature_projection(features, t5_embeddings)  # [batch, 1, hidden_dim]
            
            # Concatenate attended features at the beginning
            combined_embeddings = torch.cat([attended_features, t5_embeddings], dim=1)
            
            # Extend attention mask
            feature_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            extended_attention_mask = torch.cat([feature_mask, attention_mask], dim=1)
            
            return combined_embeddings, extended_attention_mask, True
            
        else:
            raise ValueError(f"Unknown projection mode: {self.projection_mode}")
    
    def forward(self, features, input_ids=None, attention_mask=None, labels=None):
        # Pass temperature readings through shared encoder
        embedding = self.encoder(features)
    
        # Classification branch
        class_logits = self.classifier(embedding)
    
        # Text generation branch with feature injection
        if input_ids is not None and labels is not None:
            if self.projection_mode != 'none':
                # Inject features into T5
                modified_input, modified_attention, use_embeddings = self._inject_features_into_t5(
                    embedding, input_ids, attention_mask
                )
                
                if use_embeddings:
                    # Pass embeddings directly to encoder
                    encoder_outputs = self.t5.encoder(
                        inputs_embeds=modified_input,
                        attention_mask=modified_attention
                    )
                    
                    # Generate with modified encoder outputs
                    t5_output = self.t5(
                        encoder_outputs=encoder_outputs,
                        attention_mask=modified_attention,
                        labels=labels
                    )
                else:
                    t5_output = self.t5(
                        input_ids=modified_input,
                        attention_mask=modified_attention,
                        labels=labels
                    )
            else:
                # Original approach - no feature injection
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

class ProjectionExperimentDataset(Dataset):
    """
    Dataset that supports both full and minimal text modes for experimentation
    """
    def __init__(self, df, tokenizer, max_length=128, use_minimal_text=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_minimal_text = use_minimal_text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row['features'], dtype=torch.float)
        label = torch.tensor(row['class_label'], dtype=torch.long)
        
        # Create input text based on mode
        if self.use_minimal_text:
            # Minimal prompt - forces model to rely more on injected features
            input_text = "Generate thermal assessment:"
        else:
            # Full prompt with all temperature readings
            input_text = self.create_temperature_input(row)
        
        # Target is always the synthetic conclusion
        target_text = row['synthetic_description']
        
        # Tokenize input and target
        input_tokenized = self.tokenizer(
            input_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        target_tokenized = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=64,
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
        """Same as original - creates full temperature reading prompt"""
        temp_columns = [
            'R1 int', 'L1 int', 'R2 int', 'L2 int', 'R3 int', 'L3 int', 'R4 int',
            'L4 int', 'R5 int', 'L5 int', 'R6 int', 'L6 int', 'R7 int', 'L7 int',
            'R8 int', 'L8 int', 'R9 int', 'L9 int', 'T1 int', 'T2 int', 'R0 int',
            'L0 int', 'R1 sk', 'L1 sk', 'R2 sk', 'L2 sk', 'R3 sk', 'L3 sk', 'R4 sk',
            'L4 sk', 'R5 sk', 'L5 sk', 'R6 sk', 'L6 sk', 'R7 sk', 'L7 sk', 'R8 sk',
            'L8 sk', 'R9 sk', 'L9 sk', 'T1 sk', 'T2 sk', 'R0 sk', 'L0 sk'
        ]
        
        temp_readings = []
        for col in temp_columns:
            value = row[col]
            clean_name = col.replace(' ', '_')
            temp_readings.append(f"{clean_name}={value:.1f}")
        
        input_text = "Generate thermal assessment from readings: " + ", ".join(temp_readings)
        return input_text

def create_experiment_pipeline(
    df_train, df_val, df_test,
    projection_mode='none',
    projection_type='linear', 
    use_minimal_text=False,
    tokenizer_path='./t5-small-local/',
    batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    device_override=None
):
    """
    Set up a complete experiment pipeline for different projection modes
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {projection_mode.upper()} projection + {'MINIMAL' if use_minimal_text else 'FULL'} text")
    print(f"Projection type: {projection_type}")
    print(f"{'='*60}")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    
    # Create datasets
    train_dataset = ProjectionExperimentDataset(df_train, tokenizer, use_minimal_text=use_minimal_text)
    val_dataset = ProjectionExperimentDataset(df_val, tokenizer, use_minimal_text=use_minimal_text)
    test_dataset = ProjectionExperimentDataset(df_test, tokenizer, use_minimal_text=use_minimal_text)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up device
    device = torch.device(device_override if device_override else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Initialize model
    model = MultitaskModelWithProjection(
        num_classes=2, 
        t5_model_name=tokenizer_path,
        projection_mode=projection_mode,
        projection_type=projection_type,
        use_minimal_text=use_minimal_text
    )
    
    print(f"Model configuration:")
    print(f"  Projection mode: {projection_mode}")
    print(f"  Projection type: {projection_type}")
    print(f"  Minimal text: {use_minimal_text}")
    print(f"  Feature projection: {'Yes' if model.feature_projection is not None else 'No'}")
    
    model = model.to(device)
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    return tokenizer, train_loader, val_loader, test_loader, model, optimizer, device

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
    
    print("=== ENCODER EMBEDDING DIAGNOSTIC ===")
    diagnostic_done = False
    
    with torch.no_grad():
        for features, labels, input_ids, attention_mask, target_ids in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            # ADD THIS DIAGNOSTIC CODE - ONLY RUN ONCE
            if not diagnostic_done:
                embeddings = model.encoder(features)
                print(f"Embedding variance: {embeddings.var(dim=0).mean():.6f}")
                print(f"Embedding std: {embeddings.std(dim=0).mean():.6f}")
                print(f"Embedding range: {embeddings.max() - embeddings.min():.6f}")
                if embeddings.shape[0] > 1:
                    cos_sim = torch.cosine_similarity(embeddings[0:1], embeddings[1:2])
                    print(f"Cosine similarity between samples 0 and 1: {cos_sim.item():.6f}")
                print(f"First 2 samples, first 5 dims:\n{embeddings[:2, :5]}")
                print("=== END DIAGNOSTIC ===\n")
                diagnostic_done = True

    # Handle None case for full dataset evaluation
    if text_sample_size is None or text_sample_size >= total_samples:
        # Evaluate on ALL samples
        sample_indices = set(range(total_samples))
        print(f"Evaluating text generation on FULL dataset ({total_samples} samples)")
    else:
        # Original sampling behavior
        sample_indices = set(random.sample(range(total_samples), text_sample_size))
        print(f"Sampling {len(sample_indices)} out of {total_samples} samples for text generation metrics")

    current_idx = 0

    
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
                                        do_sample=True,        # Enable sampling
                                        temperature=0.8,       # Add randomness
                                        top_p=0.9,            # Nucleus sampling
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
        train_loss, clf_loss, gen_loss = train(model, train_loader, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")
        print(f"Classification loss: {clf_loss:.4f}")
        if gen_loss is not None:
            print(f"Generation loss: {gen_loss:.4f}")
            print(f"Weighted gen loss: {model.generation_weight * gen_loss:.6f}")

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
