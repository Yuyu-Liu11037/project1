"""
Visualize diagnosis codes in hyperbolic space (2D projection)

This script loads a trained model and visualizes diagnosis codes
in the Poincaré disk (2D projection of hyperbolic space).
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
from sklearn.decomposition import PCA

from model.models import create_model
from util.code_trie import CodeTrie


def load_model_and_vocab(checkpoint_path, model_kwargs):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_state = checkpoint['model_state_dict']
    
    # Infer model parameters from checkpoint state_dict
    # emb_proc.weight shape: [P+1, embed_dim]
    # emb_third.weight shape: [T+1, embed_dim]
    # emb_diag.weight shape: [D+1, embed_dim]
    
    if 'emb_proc.weight' in model_state:
        proc_size = model_state['emb_proc.weight'].shape[0] - 1
    else:
        proc_size = model_kwargs.get('proc_size', 1000)
    
    if 'emb_third.weight' in model_state:
        # T = V - (D + P), so we need to infer from emb_third
        third_size = model_state['emb_third.weight'].shape[0] - 1
    else:
        third_size = None
    
    if 'emb_diag.weight' in model_state:
        diag_size = model_state['emb_diag.weight'].shape[0] - 1
    else:
        diag_size = model_kwargs.get('diag_size')
    
    # Infer x_vocab_size: V = D + P + T
    if third_size is not None and diag_size is not None:
        x_vocab_size = diag_size + proc_size + third_size
    else:
        x_vocab_size = model_kwargs.get('x_vocab_size', 29200)
    
    # Infer other parameters from checkpoint if available
    if 'output_projection.weight' in model_state:
        out_dim = model_state['output_projection.weight'].shape[0]
    else:
        out_dim = model_kwargs.get('out_dim', 272)
    
    if 'input_projection.weight' in model_state:
        hidden = model_state['input_projection.weight'].shape[0]
    else:
        hidden = model_kwargs.get('hidden', 512)
    
    # Infer hyp_dim from patient_hyp_head or diag_hyp_head
    if 'patient_hyp_head.proj.weight' in model_state:
        hyp_dim = model_state['patient_hyp_head.proj.weight'].shape[0]
    elif 'diag_hyp_head.proj.weight' in model_state:
        hyp_dim = model_state['diag_hyp_head.proj.weight'].shape[0]
    else:
        hyp_dim = model_kwargs.get('hyp_dim', 32)
    
    # Create model with inferred parameters
    model = create_model(
        model_type='transformer',
        x_vocab_size=x_vocab_size,
        hidden=hidden,
        out_dim=out_dim,
        diag_size=diag_size,
        proc_size=proc_size,
        hyp_dim=hyp_dim,
        **{k: v for k, v in model_kwargs.items() 
           if k not in ['x_vocab_size', 'hidden', 'out_dim', 'diag_size', 'proc_size', 'hyp_dim']}
    )
    
    model.load_state_dict(model_state)
    model.eval()
    
    return model


def get_code_embeddings(model, code_indices, diag_itos):
    """
    Get hyperbolic embeddings for given code indices.
    
    Args:
        model: Trained model
        code_indices: List of code indices (0-indexed, where 0 is padding)
        diag_itos: List mapping index to code string
    
    Returns:
        embeddings: torch.Tensor of shape (len(code_indices), hyp_dim)
        code_names: List of code strings
    """
    # Get all diagnosis code embeddings
    with torch.no_grad():
        Z_diag = model.get_diag_hyperbolic()  # (D+1, hyp_dim), where 0 is padding
    
    # Convert code strings to indices (accounting for padding at index 0)
    code_to_idx = {code: idx + 1 for idx, code in enumerate(diag_itos)}
    
    # Get embeddings for requested codes
    embeddings = []
    code_names = []
    valid_indices = []
    
    for code in code_indices:
        if code in code_to_idx:
            idx = code_to_idx[code]
            if idx < Z_diag.shape[0]:
                embeddings.append(Z_diag[idx].cpu().numpy())
                code_names.append(code)
                valid_indices.append(idx)
        else:
            print(f"Warning: Code '{code}' not found in vocabulary")
    
    if len(embeddings) == 0:
        raise ValueError("No valid codes found in the provided list")
    
    embeddings = np.array(embeddings)  # (n_codes, hyp_dim)
    
    return embeddings, code_names, valid_indices


def project_to_2d(embeddings, method='pca'):
    """
    Project high-dimensional hyperbolic embeddings to 2D.
    
    Args:
        embeddings: numpy array of shape (n_codes, hyp_dim)
        method: 'pca' for PCA projection, 'first2' for first two dimensions
    
    Returns:
        coords_2d: numpy array of shape (n_codes, 2)
    """
    if method == 'pca':
        if embeddings.shape[1] <= 2:
            return embeddings
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(embeddings)
        return coords_2d
    elif method == 'first2':
        return embeddings[:, :2]
    else:
        raise ValueError(f"Unknown projection method: {method}")


def visualize_poincare_disk(coords_2d, code_names, output_path=None, title="Diagnosis Codes in Hyperbolic Space"):
    """
    Visualize codes in the Poincaré disk (2D unit disk).
    
    Args:
        coords_2d: numpy array of shape (n_codes, 2) - coordinates in 2D
        code_names: List of code strings
        output_path: Path to save the figure (optional)
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw Poincaré disk (unit circle)
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    
    # Normalize coordinates to fit within unit disk if needed
    max_norm = np.max(np.linalg.norm(coords_2d, axis=1))
    if max_norm > 1.0:
        print(f"Warning: Some points are outside unit disk (max norm: {max_norm:.4f}). Scaling...")
        coords_2d = coords_2d / (max_norm * 1.1)  # Scale to fit with some margin
    
    # Plot points
    x_coords = coords_2d[:, 0]
    y_coords = coords_2d[:, 1]
    
    ax.scatter(x_coords, y_coords, s=100, alpha=0.6, c='blue', edgecolors='black', linewidths=1.5)
    
    # Add labels
    for i, code in enumerate(code_names):
        ax.annotate(code, (x_coords[i], y_coords[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    # Set limits and aspect ratio
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('X (2D Projection)', fontsize=12)
    ax.set_ylabel('Y (2D Projection)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_with_hierarchy(coords_2d, code_names, diag_trie, output_path=None, 
                            title="Diagnosis Codes in Hyperbolic Space (with Hierarchy)"):
    """
    Visualize codes with hierarchical relationships shown as edges.
    
    Args:
        coords_2d: numpy array of shape (n_codes, 2)
        code_names: List of code strings
        diag_trie: CodeTrie object for finding parent-child relationships
        output_path: Path to save the figure (optional)
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw Poincaré disk
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    
    # Normalize coordinates
    max_norm = np.max(np.linalg.norm(coords_2d, axis=1))
    if max_norm > 1.0:
        coords_2d = coords_2d / (max_norm * 1.1)
    
    # Create code to index mapping
    code_to_idx = {code: i for i, code in enumerate(code_names)}
    
    # Find parent-child relationships
    parent_child_pairs = []
    for code in code_names:
        all_sequences = diag_trie.get_all_sequences()
        sequences = [seq for seq in all_sequences if code in seq]
        
        for seq in sequences:
            if code in seq:
                code_idx_in_seq = seq.index(code)
                if code_idx_in_seq > 0:
                    parent_code = seq[code_idx_in_seq - 1]
                    if parent_code in code_to_idx:
                        parent_child_pairs.append((parent_code, code))
    
    # Draw edges for parent-child relationships
    for parent, child in parent_child_pairs:
        parent_idx = code_to_idx[parent]
        child_idx = code_to_idx[child]
        ax.plot([coords_2d[parent_idx, 0], coords_2d[child_idx, 0]],
                [coords_2d[parent_idx, 1], coords_2d[child_idx, 1]],
                'gray', alpha=0.3, linewidth=1, linestyle='-')
    
    # Plot points
    x_coords = coords_2d[:, 0]
    y_coords = coords_2d[:, 1]
    
    ax.scatter(x_coords, y_coords, s=100, alpha=0.6, c='blue', edgecolors='black', linewidths=1.5)
    
    # Add labels
    for i, code in enumerate(code_names):
        ax.annotate(code, (x_coords[i], y_coords[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    # Set limits and aspect ratio
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('X (2D Projection)', fontsize=12)
    ax.set_ylabel('Y (2D Projection)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_codes_from_checkpoint(checkpoint_path, code_list, diag_itos, 
                                    output_path='hyperbolic_visualization.png',
                                    method='pca', show_hierarchy=False,
                                    trie_file='cond_hist_codes.txt',
                                    model_kwargs=None):
    """
    Convenience function to visualize codes directly from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        code_list: List of code strings to visualize
        diag_itos: List mapping index to code string
        output_path: Output path for visualization
        method: 'pca' or 'first2' for projection
        show_hierarchy: Whether to show hierarchical relationships
        trie_file: Path to cond_hist_codes.txt
        model_kwargs: Dictionary of model parameters (optional)
    
    Example:
        diag_itos = ['A04', 'A047', 'I10', ...]  # Your vocabulary
        visualize_codes_from_checkpoint(
            'checkpoints/model_final.pt',
            ['A04', 'A047', 'A0471', 'I10', 'I100'],
            diag_itos,
            output_path='my_visualization.png',
            show_hierarchy=True
        )
    """
    if model_kwargs is None:
        model_kwargs = {
            'diag_size': len(diag_itos),
            'proc_size': 1000,
            'hyp_dim': 32,
        }
    
    # Load model
    model = load_model_and_vocab(checkpoint_path, model_kwargs)
    
    # Get embeddings
    embeddings, code_names, valid_indices = get_code_embeddings(model, code_list, diag_itos)
    
    # Project to 2D
    coords_2d = project_to_2d(embeddings, method=method)
    
    # Visualize
    if show_hierarchy:
        diag_trie = CodeTrie.from_file(trie_file)
        visualize_with_hierarchy(coords_2d, code_names, diag_trie, 
                               output_path=output_path,
                               title=f"Diagnosis Codes in Hyperbolic Space\n({len(code_names)} codes)")
    else:
        visualize_poincare_disk(coords_2d, code_names, 
                               output_path=output_path,
                               title=f"Diagnosis Codes in Hyperbolic Space\n({len(code_names)} codes)")


def main():
    parser = argparse.ArgumentParser(description='Visualize diagnosis codes in hyperbolic space')
    parser.add_argument('--checkpoint', type=str, required=True, default='checkpoints/model_final.pt',
                       help='Path to model checkpoint file')
    parser.add_argument('--codes', type=str, nargs='+', default=['A04','A047','A0471','A41','A418','A4181','C34','C340','C3400','C50','C500','C5001','C50011'],
                       help='List of diagnosis codes to visualize (e.g., A04 A047 I10)')
    parser.add_argument('--vocab', type=str, default='vocab.pkl',
                       help='Path to vocabulary file (pickle) containing diag_itos. If not provided, will try to infer from checkpoint.')
    parser.add_argument('--output', type=str, default='visualization.png',
                       help='Output path for the visualization')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'first2'],
                       help='Projection method: pca or first2')
    parser.add_argument('--show_hierarchy', action='store_true', default=True,
                       help='Show hierarchical relationships as edges')
    parser.add_argument('--trie_file', type=str, default='cond_hist_codes.txt',
                       help='Path to cond_hist_codes.txt for hierarchy visualization')
    
    args = parser.parse_args()
    
    # Load vocabulary if provided
    diag_itos = None
    if args.vocab:
        with open(args.vocab, 'rb') as f:
            vocab_data = pickle.load(f)
            if isinstance(vocab_data, dict):
                diag_itos = vocab_data.get('diag_itos')
            elif isinstance(vocab_data, tuple):
                diag_itos = vocab_data[0]  # Assuming (diag_stoi, diag_itos, ...)
    
    if diag_itos is None:
        print("Warning: Vocabulary not provided. Trying to load from checkpoint...")
        # You may need to adjust this based on how you save vocab in checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'vocab' in checkpoint:
            diag_itos = checkpoint['vocab'].get('diag_itos')
    
    if diag_itos is None:
        raise ValueError("Could not load vocabulary. Please provide --vocab argument.")
    
    # Prepare model_kwargs
    model_kwargs = {
        'diag_size': len(diag_itos),
        'proc_size': 1000,  # Adjust based on your data
        'hyp_dim': 32,  # Adjust based on your model
    }
    
    # Try to infer from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_kwargs' in checkpoint:
        model_kwargs.update(checkpoint['model_kwargs'])
    
    # Use the convenience function
    print(f"Visualizing {len(args.codes)} codes...")
    visualize_codes_from_checkpoint(
        checkpoint_path=args.checkpoint,
        code_list=args.codes,
        diag_itos=diag_itos,
        output_path=args.output,
        method=args.method,
        show_hierarchy=args.show_hierarchy,
        trie_file=args.trie_file,
        model_kwargs=model_kwargs
    )
    
    print("Visualization complete!")


if __name__ == '__main__':
    main()

