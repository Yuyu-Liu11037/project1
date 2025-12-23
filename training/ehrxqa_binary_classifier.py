"""
EHRXQA Binary Classification Training
Trains a binary classifier (yes/no) using patient vectors and question embeddings
"""
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score, f1_score
import argparse
from datetime import datetime

from extract_patient_vectors import (
    CLSExtractorWrapper,
    load_checkpoint,
    convert_json_entry_to_sample,
    prepare_input_from_sample
)
from extract_question_embeddings import (
    load_model_and_tokenizer,
    generate_sentence_embeddings
)


class EHRXQADataset(Dataset):
    """Dataset for EHRXQA binary classification"""
    
    def __init__(
        self,
        json_files: List[Path],
        patient_model,
        patient_vocabs,
        patient_config,
        question_model,
        question_tokenizer,
        device: str = 'cuda',
        cache_vectors: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            json_files: List of paths to _processed.json files
            patient_model: Loaded patient vector extraction model
            patient_vocabs: Vocabularies for patient model
            patient_config: Configuration for patient model
            question_model: Bio_ClinicalBERT model
            question_tokenizer: Bio_ClinicalBERT tokenizer
            device: Device to run inference on
            cache_vectors: Whether to cache extracted vectors
        """
        self.device = device
        self.patient_model = patient_model
        self.patient_vocabs = patient_vocabs
        self.patient_config = patient_config
        self.question_model = question_model
        self.question_tokenizer = question_tokenizer
        
        # Load all entries from JSON files
        self.entries = []
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.entries.extend(data)
                else:
                    print(f"Warning: {json_file} is not a list, skipping")
        
        print(f"Loaded {len(self.entries)} total entries")
        
        # Extract patient vectors and question embeddings
        self.patient_vectors_euclidean = []
        self.question_embeddings = []
        self.labels = []
        
        # Wrap patient model for CLS extraction
        self.patient_extractor = CLSExtractorWrapper(patient_model).to(device)
        self.patient_extractor.eval()
        
        # Freeze patient model parameters
        for param in self.patient_extractor.parameters():
            param.requires_grad = False
        
        # Freeze question model parameters
        for param in question_model.parameters():
            param.requires_grad = False
        
        # Get manifold from model
        # Try to access manifold_hidden from the wrapped model or directly
        if hasattr(patient_model, 'manifold_hidden'):
            self.manifold = patient_model.manifold_hidden
        elif hasattr(self.patient_extractor.model, 'manifold_hidden'):
            self.manifold = self.patient_extractor.model.manifold_hidden
        else:
            # Fallback to default Lorentz manifold
            from hypercore.manifolds import Lorentz
            self.manifold = Lorentz(1.0)
            print("Warning: Could not find manifold_hidden, using default Lorentz(1.0)")
        
        # Extract vectors
        self._extract_vectors(cache_vectors)
    
    def _extract_vectors(self, cache: bool):
        """Extract patient vectors and question embeddings for all entries"""
        print("Extracting patient vectors and question embeddings...")
        
        batch_size = 32
        questions = []
        
        for i, entry in enumerate(self.entries):
            # Collect question
            question = entry.get('question', '')
            if not question or not isinstance(question, str):
                question = ''
            questions.append(question)
            
            # Collect label
            answer = entry.get('answer', [0])
            label = int(answer[0]) if isinstance(answer, list) and len(answer) > 0 else 0
            self.labels.append(label)
        
        # Extract question embeddings in batches
        print("Extracting question embeddings...")
        question_embeddings_tensor = generate_sentence_embeddings(
            questions, self.question_model, self.question_tokenizer, self.device, batch_size
        )
        self.question_embeddings = question_embeddings_tensor.cpu().numpy()
        
        # Extract patient vectors in batches
        print("Extracting patient vectors...")
        patient_vectors_hyperbolic = []
        
        for i in range(0, len(self.entries), batch_size):
            batch_entries = self.entries[i:i+batch_size]
            batch_samples = [convert_json_entry_to_sample(entry) for entry in batch_entries]
            
            # Prepare batch inputs
            batch_x_diag = []
            batch_x_proc = []
            batch_x_drug = []
            batch_x_visit_ids = []
            
            for sample in batch_samples:
                x_diag, x_proc, x_drug, x_visit_ids = prepare_input_from_sample(
                    sample, self.patient_vocabs,
                    self.patient_config['max_diag_len'],
                    self.patient_config['max_proc_len'],
                    self.patient_config['max_drug_len']
                )
                batch_x_diag.append(x_diag)
                batch_x_proc.append(x_proc)
                batch_x_drug.append(x_drug)
                batch_x_visit_ids.append(x_visit_ids)
            
            # Stack into batch tensors
            batch_x_diag = torch.cat(batch_x_diag, dim=0).to(self.device)
            batch_x_proc = torch.cat(batch_x_proc, dim=0).to(self.device)
            batch_x_drug = torch.cat(batch_x_drug, dim=0).to(self.device)
            batch_x_visit_ids = torch.cat(batch_x_visit_ids, dim=0).to(self.device)
            
            # Extract CLS representations (in hyperbolic space)
            with torch.no_grad():
                patient_vectors_hyp = self.patient_extractor.extract_cls(
                    batch_x_diag, batch_x_proc, batch_x_drug, batch_x_visit_ids
                )
                # Map to Euclidean space using logmap0
                patient_vectors_euc = self.manifold.logmap0(patient_vectors_hyp)
                patient_vectors_hyperbolic.append(patient_vectors_euc.cpu())
            
            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(self.entries):
                print(f"  Processed {min(i + batch_size, len(self.entries))}/{len(self.entries)} entries")
        
        # Concatenate all patient vectors
        patient_vectors_tensor = torch.cat(patient_vectors_hyperbolic, dim=0)
        self.patient_vectors_euclidean = patient_vectors_tensor.numpy()
        
        print(f"Patient vector dimension: {self.patient_vectors_euclidean.shape[1]}")
        print(f"Question embedding dimension: {self.question_embeddings.shape[1]}")
        print(f"Labels distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.patient_vectors_euclidean[idx]),
            torch.FloatTensor(self.question_embeddings[idx]),
            torch.LongTensor([self.labels[idx]])[0]  # Return scalar, not tensor
        )


class BinaryClassifier(nn.Module):
    """Binary classifier for yes/no question answering"""
    
    def __init__(
        self,
        patient_vector_dim: int,
        question_embed_dim: int,
        hidden_dim: int = None
    ):
        """
        Initialize binary classifier
        
        Args:
            patient_vector_dim: Dimension of patient vector (after logmap0)
            question_embed_dim: Dimension of question embedding (Bio_ClinicalBERT)
            hidden_dim: Hidden dimension for MLP (default: 2 * patient_vector_dim)
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = 2 * patient_vector_dim
        
        # Trainable Linear layer to project question embedding to patient vector dimension
        self.question_proj = nn.Linear(question_embed_dim, patient_vector_dim)
        
        # Two-layer MLP
        # Input: concatenated vectors (patient_vector_dim + patient_vector_dim)
        # Hidden: hidden_dim
        # Output: 2 (yes/no logits)
        self.mlp = nn.Sequential(
            nn.Linear(2 * patient_vector_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, patient_vector, question_embedding):
        """
        Forward pass
        
        Args:
            patient_vector: (batch_size, patient_vector_dim)
            question_embedding: (batch_size, question_embed_dim)
        
        Returns:
            logits: (batch_size, 2) - yes/no logits
        """
        # Project question embedding to patient vector dimension
        question_proj = self.question_proj(question_embedding)  # (batch_size, patient_vector_dim)
        
        # Concatenate vectors
        combined = torch.cat([patient_vector, question_proj], dim=1)  # (batch_size, 2 * patient_vector_dim)
        
        # Pass through MLP
        logits = self.mlp(combined)  # (batch_size, 2)
        
        return logits


def evaluate(model, data_loader, device='cuda'):
    """Evaluate model and return Accuracy and F1 score"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for patient_vec, question_emb, labels in data_loader:
            patient_vec = patient_vec.to(device)
            question_emb = question_emb.to(device)
            labels = labels.to(device)
            
            logits = model(patient_vec, question_emb)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def train(
    train_dataset,
    val_dataset,
    test_dataset = None,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    epochs: int = 20,
    device: str = 'cuda',
    patience: int = 5,
    min_delta: float = 0.001
):
    """Train binary classifier"""
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None
    
    # Get dimensions from underlying dataset (in case of Subset)
    underlying_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
    patient_dim = underlying_dataset.patient_vectors_euclidean.shape[1]
    question_dim = underlying_dataset.question_embeddings.shape[1]
    
    # Create model
    model = BinaryClassifier(patient_dim, question_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nModel architecture:")
    print(f"  Patient vector dim: {patient_dim}")
    print(f"  Question embedding dim: {question_dim}")
    print(f"  MLP hidden dim: {2 * patient_dim}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Training loop
    best_val_f1 = -float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for patient_vec, question_emb, labels in train_loader:
            patient_vec = patient_vec.to(device)
            question_emb = question_emb.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(patient_vec, question_emb)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        current_f1 = val_metrics['f1']
        
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        # Early stopping
        if current_f1 > best_val_f1 + min_delta:
            best_val_f1 = current_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  → New best F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Restoring best model from epoch {epoch - patience_counter}")
            model.load_state_dict(best_model_state)
            break
    
    # Test evaluation
    if test_loader:
        test_metrics = evaluate(model, test_loader, device)
        print(f"\n[TEST] Accuracy: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")
        return model, test_metrics
    else:
        return model, val_metrics


def split_dataset(dataset: EHRXQADataset, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    """Split dataset into train/val/test"""
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    
    # Set random seed
    np.random.seed(seed)
    indices = np.random.permutation(total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_subset, val_subset, test_subset


def main():
    parser = argparse.ArgumentParser(description='Train EHRXQA binary classifier')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/ltransformer_encoder_next_20251220_103843.pth',
                        help='Path to patient vector model checkpoint')
    parser.add_argument('--dataset_dir', type=str,
                        default='/data/yuyu/data/EHRXQA/ehrxqa/dataset',
                        help='Directory containing _processed.json files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load patient vector model
    print("\nLoading patient vector model...")
    patient_model, patient_vocabs, patient_config = load_checkpoint(Path(args.checkpoint))
    patient_model = patient_model.to(device)
    patient_model.eval()
    
    # Load question embedding model
    print("\nLoading question embedding model...")
    question_tokenizer, question_model, _ = load_model_and_tokenizer(
        "emilyalsentzer/Bio_ClinicalBERT", device
    )
    question_model.eval()
    
    # Find all _processed.json files
    dataset_dir = Path(args.dataset_dir)
    json_files = list(dataset_dir.glob("*_processed.json"))
    
    if not json_files:
        print(f"Error: No _processed.json files found in {dataset_dir}")
        return
    
    print(f"\nFound {len(json_files)} JSON files:")
    for f in json_files:
        print(f"  - {f.name}")
    
    # Create dataset
    print("\nCreating dataset...")
    full_dataset = EHRXQADataset(
        json_files,
        patient_model,
        patient_vocabs,
        patient_config,
        question_model,
        question_tokenizer,
        device=device
    )
    
    # Split dataset
    print("\nSplitting dataset...")
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Train
    model, test_metrics = train(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        device=device,
        patience=args.patience
    )
    
    # Save model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"ehrxqa_binary_classifier_{timestamp}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'patient_vector_dim': full_dataset.patient_vectors_euclidean.shape[1],
        'question_embed_dim': full_dataset.question_embeddings.shape[1],
        'test_metrics': test_metrics,
        'args': vars(args)
    }, checkpoint_path)
    
    print(f"\nModel saved to: {checkpoint_path}")
    print(f"Final test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
