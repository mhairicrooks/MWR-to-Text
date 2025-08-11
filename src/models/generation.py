import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt



class T5GenerationOnlyModel(nn.Module):
    def __init__(self, model_name='t5-small'):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss, output.logits

    def generate(self, input_ids, attention_mask=None, max_length=64):
        return self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length
        )
    
class TextOnlyDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length=140, max_target_length=140):#64):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_text = create_temperature_input(row)
        target_text = row['synthetic_description']

        input_enc = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        target_enc = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_target_length,
            return_tensors='pt'
        )

        return (
            input_enc['input_ids'].squeeze(0),
            input_enc['attention_mask'].squeeze(0),
            target_enc['input_ids'].squeeze(0)
        )

def create_temperature_input(row):
        """
        Convert temperature readings into a text prompt for T5 using your exact column format
        """
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
    
def compute_rougeL(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_score = 0.0
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        total_score += score['rougeL'].fmeasure
    return total_score / len(predictions)

def train_and_validate_t5_with_metrics(
    model, train_loader, val_loader, optimizer,
    device, tokenizer, num_epochs=30, meteor_every=5
):
    best_meteor = 0.0
    best_rougeL = 0.0
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    meteor_scores = []
    rougeL_scores = []
    meteor_epochs = []  # Track which epochs had METEOR evaluation
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc="Training")
        for input_ids, attention_mask, labels in progress:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Validation loss
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # METEOR + ROUGE Evaluation every N epochs
        if (epoch + 1) % meteor_every == 0:
            print(f"\n--- Text Generation Evaluation (Epoch {epoch + 1}) ---")
            references = []
            predictions = []
            with torch.no_grad():
                for input_ids, attention_mask, labels in val_loader:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device) 
                    labels = labels.to(device)
                    gen_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    
                    # Process ALL samples in the batch, not just the first one
                    for i in range(len(gen_ids)):
                        pred_text = tokenizer.decode(gen_ids[i], skip_special_tokens=True)
                        ref_text = tokenizer.decode(labels[i], skip_special_tokens=True)
                        predictions.append(pred_text)
                        references.append(ref_text)
            
            # Compute METEOR with tokenized inputs
            meteor_score_values = [
                meteor_score([ref.lower().split()], pred.lower().split())
                for ref, pred in zip(references, predictions)
            ]
            avg_meteor = sum(meteor_score_values) / len(meteor_score_values)
            
            # Compute ROUGE-L
            avg_rougeL = compute_rougeL(references, predictions)
            
            # Store metrics
            meteor_scores.append(avg_meteor)
            rougeL_scores.append(avg_rougeL)
            meteor_epochs.append(epoch + 1)
            
            print(f"  METEOR Score : {avg_meteor:.4f}")
            print(f"  ROUGE-L Score: {avg_rougeL:.4f}")
            
            # Update bests
            if avg_meteor > best_meteor:
                best_meteor = avg_meteor
                print(f"  ✓ New best METEOR: {best_meteor:.4f}")
            if avg_rougeL > best_rougeL:
                best_rougeL = avg_rougeL
                print(f"  ✓ New best ROUGE-L: {best_rougeL:.4f}")
    
    # Plot metrics
    plot_training_metrics(train_losses, val_losses, meteor_scores, rougeL_scores, meteor_epochs)
    
    return best_meteor, best_rougeL, avg_train_loss, avg_val_loss

def train_and_validate_t5_with_early_stopping(
    model, train_loader, val_loader, optimizer,
    device, tokenizer, num_epochs=30, meteor_every=5,
    patience=5, min_delta=0.001, save_best_model=True, model_save_path="best_model.pt"
):
    """
    Train T5 model with early stopping based on validation loss.
    
    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as improvement
        save_best_model: Whether to save the best model
        model_save_path: Path to save the best model
    """
    best_meteor = 0.0
    best_rougeL = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    meteor_scores = []
    rougeL_scores = []
    meteor_epochs = []
    
    print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc="Training")
        
        for input_ids, attention_mask, labels in progress:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Validation loss
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss - min_delta:
            # Significant improvement
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model state
            if save_best_model:
                best_model_state = model.state_dict().copy()
            
            print(f"✓ New best validation loss: {best_val_loss:.4f}")
        else:
            # No significant improvement
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        # METEOR + ROUGE Evaluation every N epochs
        if (epoch + 1) % meteor_every == 0:
            print(f"\n--- Text Generation Evaluation (Epoch {epoch + 1}) ---")
            references = []
            predictions = []
            with torch.no_grad():
                for input_ids, attention_mask, labels in val_loader:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device) 
                    labels = labels.to(device)
                    gen_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    
                    # Process ALL samples in the batch, not just the first one
                    for i in range(len(gen_ids)):
                        pred_text = tokenizer.decode(gen_ids[i], skip_special_tokens=True)
                        ref_text = tokenizer.decode(labels[i], skip_special_tokens=True)
                        predictions.append(pred_text)
                        references.append(ref_text)
            
            # Compute METEOR with tokenized inputs
            meteor_score_values = [
                meteor_score([ref.lower().split()], pred.lower().split())
                for ref, pred in zip(references, predictions)
            ]
            avg_meteor = sum(meteor_score_values) / len(meteor_score_values)
            
            # Compute ROUGE-L
            avg_rougeL = compute_rougeL(references, predictions)
            
            # Store metrics
            meteor_scores.append(avg_meteor)
            rougeL_scores.append(avg_rougeL)
            meteor_epochs.append(epoch + 1)
            
            print(f"  METEOR Score : {avg_meteor:.4f}")
            print(f"  ROUGE-L Score: {avg_rougeL:.4f}")
            
            # Update bests
            if avg_meteor > best_meteor:
                best_meteor = avg_meteor
                print(f"  ✓ New best METEOR: {best_meteor:.4f}")
            if avg_rougeL > best_rougeL:
                best_rougeL = avg_rougeL
                print(f"  ✓ New best ROUGE-L: {best_rougeL:.4f}")
    
    # Load best model if early stopping occurred and we saved it
    if save_best_model and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation loss: {best_val_loss:.4f}")
        
        # Optionally save to disk
        if model_save_path:
            torch.save(best_model_state, model_save_path)
            print(f"Best model saved to: {model_save_path}")
    
    # Plot metrics
    plot_training_metrics(train_losses, val_losses, meteor_scores, rougeL_scores, meteor_epochs)
    
    return best_meteor, best_rougeL, avg_train_loss, avg_val_loss, best_val_loss


def evaluate_on_test_set(model, test_loader, device, tokenizer):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained T5 model
        test_loader: DataLoader for test set
        device: torch device (cuda/cpu)
        tokenizer: T5 tokenizer
    
    Returns:
        dict: Dictionary containing test metrics
    """
    print("=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)
    
    model.eval()
    total_test_loss = 0.0
    references = []
    predictions = []
    
    with torch.no_grad():
        # Calculate test loss and collect predictions
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Calculate loss
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_test_loss += loss.item()
            
            # Generate predictions
            gen_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            
            # Process ALL samples in the batch
            for i in range(len(gen_ids)):
                pred_text = tokenizer.decode(gen_ids[i], skip_special_tokens=True)
                ref_text = tokenizer.decode(labels[i], skip_special_tokens=True)
                predictions.append(pred_text)
                references.append(ref_text)
    
    # Calculate metrics
    avg_test_loss = total_test_loss / len(test_loader)
    
    # Compute METEOR scores
    meteor_scores = [
        meteor_score([ref.lower().split()], pred.lower().split())
        for ref, pred in zip(references, predictions)
    ]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    
    # Compute ROUGE-L
    avg_rougeL = compute_rougeL(references, predictions)
    
    # Print results
    print(f"Test Set Results:")
    print(f"  Samples evaluated: {len(predictions)}")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  METEOR Score: {avg_meteor:.4f}")
    print(f"  ROUGE-L Score: {avg_rougeL:.4f}")
    
    # Show some example predictions
    print(f"\nSample Predictions:")
    print("-" * 30)
    for i in range(min(5, len(predictions))):  # Show first 5 examples
        print(f"Example {i+1}:")
        print(f"  Reference: {references[i]}")
        print(f"  Prediction: {predictions[i]}")
        print(f"  METEOR: {meteor_scores[i]:.4f}")
        print()
    
    # Return metrics dictionary
    return {
        'test_loss': avg_test_loss,
        'meteor_score': avg_meteor,
        'rougeL_score': avg_rougeL,
        'num_samples': len(predictions),
        'all_references': references,
        'all_predictions': predictions,
        'individual_meteor_scores': meteor_scores
    }

def plot_training_metrics(train_losses, val_losses, meteor_scores, rougeL_scores, meteor_epochs):
    """
    Plot training metrics in two subplots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Training and Validation Loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: METEOR and ROUGE-L Scores
    ax2.plot(meteor_epochs, meteor_scores, 'g-o', label='METEOR Score', linewidth=2, markersize=6)
    ax2.plot(meteor_epochs, rougeL_scores, 'm-s', label='ROUGE-L Score', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('METEOR and ROUGE-L Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)  # Scores are typically between 0 and 1
    
    plt.tight_layout()
    plt.show()
    
    # Print final metrics summary
    print("\n" + "="*50)
    print("TRAINING COMPLETED - FINAL METRICS SUMMARY")
    print("="*50)
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    if meteor_scores:
        print(f"Final METEOR Score: {meteor_scores[-1]:.4f}")
        print(f"Best METEOR Score: {max(meteor_scores):.4f}")
    if rougeL_scores:
        print(f"Final ROUGE-L Score: {rougeL_scores[-1]:.4f}")
        print(f"Best ROUGE-L Score: {max(rougeL_scores):.4f}")
