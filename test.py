"""
This script runs a full end-to-end diagnostic of the MultitaskModel and MultitaskDataset pipeline.

It performs the following:
- Checks GPU/CPU availability
- Creates a dummy dataset with synthetic features and text descriptions
- Initializes and tests the tokenizer
- Instantiates the dataset and verifies sample formatting
- Initializes the multitask model and runs forward passes for both classification and text generation
- Validates batching with a PyTorch DataLoader
- Executes a single training step to confirm backward pass and optimizer functionality

Intended Use:
    For debugging and validating model architecture and data handling before training on real data.

Note:
    This is not a formal unit test suite — it's a practical smoke test to confirm that the model,
    data pipeline, and training components integrate and run without errors on dummy data.

Requirements:
    - PyTorch
    - HuggingFace Transformers
    - pandas

To run:
    python test_model.py
"""

import torch
import pandas as pd
from transformers import T5Tokenizer
from model import MultitaskModel, MultitaskDataset  # Import your classes

def test_model():
    """
    Runs a comprehensive test suite for the MultitaskModel and MultitaskDataset classes.

    This function is designed for development-time sanity checks to ensure:
    - Device (CPU/GPU) availability
    - Dummy input data generation and formatting
    - Tokenizer and dataset class integration
    - Model initialization and forward pass (classification and generation)
    - Basic DataLoader batching
    - A single training step runs without errors

    It prints out key shape and loss information at each step, helping to verify that
    data flows correctly through the entire pipeline. This is useful before beginning
    training with real clinical data.

    Note:
        This function is meant for quick debugging and should not be used for benchmarking
        or validation. The dummy data is synthetic and not representative of real inputs.
    """
    print("Testing MultitaskModel...")
    
    # 1. Test GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 2. Create dummy data to test with
    print("\nCreating dummy data...")
    dummy_data = {
        'features': [[1.2, 2.3, 3.4] * 15 for _ in range(10)],  # 45 features per sample, 10 samples
        'class_label': [0, 1, 2, 3, 4, 5, 0, 1, 2, 3],  # Mix of classes
        'synthetic_description': [
            "No thermal changes detected.",
            "Slightly elevated temperature.",
            "Moderately elevated temperature (surface).",
            "Moderately elevated temperature (surface and depth).",
            "Increased temperature (surface and depth), partial asymmetry.",
            "Increased temperature (surface and depth), greater and clear asymmetry.",
            "No thermal changes detected.",
            "Slightly elevated temperature.",
            "Moderately elevated temperature (surface).",
            "Moderately elevated temperature (surface and depth)."
        ]
    }
    
    df = pd.DataFrame(dummy_data)
    print(f"Created dummy dataset with {len(df)} samples")
    
    # 3. Test tokenizer
    print("\nTesting tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    print("✓ Tokenizer loaded successfully")
    
    # 4. Test dataset class
    print("\nTesting dataset class...")
    dataset = MultitaskDataset(df, tokenizer, max_length=64)
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Test one sample
    features, label, input_ids, attention_mask, target_ids = dataset[0]
    print(f"✓ Sample shapes - Features: {features.shape}, Label: {label.shape}, Input IDs: {input_ids.shape}")
    
    # 5. Test model initialization
    print("\nTesting model...")
    num_features = len(dummy_data['features'][0])  # Should be 45
    model = MultitaskModel(num_features=num_features, num_classes=6)
    model = model.to(device)
    print(f"✓ Model created and moved to {device}")
    
    # 6. Test forward pass (classification only)
    print("\nTesting forward pass (classification only)...")
    model.eval()  # prevent BatchNorm from erroring
    with torch.no_grad():
        batch_features = features.unsqueeze(0).to(device)
        class_logits, gen_loss, gen_logits = model(batch_features)
        print(f"✓ Classification output shape: {class_logits.shape}")
        print(f"✓ Generation loss: {gen_loss} (should be None)")   
    
    # 7. Test forward pass (full multitask)
    print("\nTesting forward pass (full multitask)...")
    with torch.no_grad():
        batch_features = features.unsqueeze(0).to(device)
        batch_input_ids = input_ids.unsqueeze(0).to(device)
        batch_attention_mask = attention_mask.unsqueeze(0).to(device)
        batch_target_ids = target_ids.unsqueeze(0).to(device)
        
        class_logits, gen_loss, gen_logits = model(
            batch_features, batch_input_ids, batch_attention_mask, batch_target_ids
        )
        print(f"✓ Classification output shape: {class_logits.shape}")
        print(f"✓ Generation loss: {gen_loss.item():.4f}")
        print(f"✓ Generation logits shape: {gen_logits.shape}")
    
    # 8. Test DataLoader
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_idx, (features, labels, input_ids, attention_mask, target_ids) in enumerate(dataloader):
        print(f"✓ Batch {batch_idx}: Features {features.shape}, Labels {labels.shape}")
        if batch_idx == 0:  # Only test first batch
            break
    
    # 9. Test one training step
    print("\nTesting one training step...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    clf_loss_fn = torch.nn.CrossEntropyLoss()
    
    # Get one batch
    features, labels, input_ids, attention_mask, target_ids = next(iter(dataloader))
    features, labels = features.to(device), labels.to(device)
    input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(device)
    
    optimizer.zero_grad()
    class_logits, gen_loss, _ = model(features, input_ids, attention_mask, target_ids)
    clf_loss = clf_loss_fn(class_logits, labels)
    
    if gen_loss is not None:
        total_loss = clf_loss + gen_loss
    else:
        total_loss = clf_loss
    
    total_loss.backward()
    optimizer.step()
    
    print(f"✓ Training step completed - Loss: {total_loss.item():.4f}")
    
    print("\n🎉 All tests passed! Your model is working correctly.")
    print(f"Ready to train on {device}")

if __name__ == "__main__":
    test_model()