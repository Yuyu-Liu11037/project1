import math
import random
import pickle
import argparse
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Iterable, Set, Optional
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Poincaré ball geometry (curvature -1)
# ------------------------------

def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))

def poincare_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    """Hyperbolic distance on Poincaré ball with curvature -1."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx2 = np.dot(x, x)
    ny2 = np.dot(y, y)
    diff2 = np.dot(x - y, x - y)
    denom = max((1.0 - nx2) * (1.0 - ny2), eps)
    arg = 1.0 + 2.0 * diff2 / denom
    arg = max(arg, 1.0 + 1e-12)
    return math.acosh(arg)

def poincare_radius(x: np.ndarray) -> float:
    """Hyperbolic norm r_D = d(0, x) = 2 * atanh(||x||)"""
    r = _norm(x)
    r = min(max(r, 0.0), 1.0 - 1e-12)
    return 2.0 * math.atanh(r)

# ------------------------------
# Entailment cones (Ganea et al., 2018)
# ------------------------------

def xi_angle(x: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    """Ξ(x,y) = π - ∠Oxy"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = _norm(x)
    ny = _norm(y)
    if nx < eps:
        return math.pi
    xy = float(np.dot(x, y))
    x_minus_y = x - y
    nxy = _norm(x_minus_y)
    denom_inside = max(1.0 + nx*nx*ny*ny - 2.0*xy, eps)
    denom = max(nx * nxy * math.sqrt(denom_inside), eps)
    numer = xy * (1.0 + nx*nx) - (nx*nx) * (1.0 + ny*ny)
    cos_val = numer / denom
    cos_val = min(1.0, max(-1.0, cos_val))
    return math.acos(cos_val)

def psi_aperture(x: np.ndarray, epsilon: float = 0.1) -> float:
    """ψ(x) = arcsin( K * (1-||x||^2)/||x|| )"""
    nx = _norm(np.asarray(x, dtype=float))
    if nx <= epsilon:
        return 0.0
    K = epsilon / (1.0 - epsilon*epsilon)
    val = K * (1.0 - nx*nx) / max(nx, 1e-12)
    val = min(1.0, max(0.0, val))
    return math.asin(val)

def entails_by_cone(parent_vec: np.ndarray, child_vec: np.ndarray, epsilon: float = 0.1) -> Tuple[bool, float]:
    """Check if child lies within parent's entailment cone."""
    psi = psi_aperture(parent_vec, epsilon=epsilon)
    xi = xi_angle(parent_vec, child_vec)
    margin = psi - xi
    return (margin >= 0.0), margin

# ------------------------------
# Graph utilities
# ------------------------------

class ICDHierarchy:
    def __init__(self, edges: List[Tuple[str, str]]):
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.children: Dict[str, Set[str]] = defaultdict(set)
        self.nodes: Set[str] = set()
        for p, c in edges:
            self.parents[c].add(p)
            self.children[p].add(c)
            self.nodes.add(p); self.nodes.add(c)
        self.undirected: Dict[str, Set[str]] = defaultdict(set)
        for p, c in edges:
            self.undirected[p].add(c)
            self.undirected[c].add(p)

    def roots(self) -> List[str]:
        return [n for n in self.nodes if len(self.parents[n]) == 0]

    def depth(self) -> Dict[str, int]:
        depth = {n: math.inf for n in self.nodes}
        dq = deque()
        for r in self.roots():
            depth[r] = 0
            dq.append(r)
        while dq:
            u = dq.popleft()
            for v in self.children[u]:
                if depth[v] > depth[u] + 1:
                    depth[v] = depth[u] + 1
                    dq.append(v)
        maxd = max([d for d in depth.values() if d < math.inf] + [0])
        for n in list(depth.keys()):
            if not math.isfinite(depth[n]):
                depth[n] = maxd + 1
        return depth

    def is_ancestor(self, a: str, b: str, max_depth: Optional[int] = None) -> bool:
        seen = set()
        dq = deque([(b, 0)])
        while dq:
            node, d = dq.popleft()
            for p in self.parents[node]:
                if p == a:
                    return True
                if p not in seen and (max_depth is None or d+1 <= max_depth):
                    seen.add(p)
                    dq.append((p, d+1))
        return False

    def is_descendant(self, a: str, b: str, max_depth: Optional[int] = None) -> bool:
        seen = set()
        dq = deque([(b, 0)])
        while dq:
            node, d = dq.popleft()
            for ch in self.children[node]:
                if ch == a:
                    return True
                if ch not in seen and (max_depth is None or d+1 <= max_depth):
                    seen.add(ch)
                    dq.append((ch, d+1))
        return False

    def undirected_shortest_path_len(self, a: str, b: str, cutoff: Optional[int] = None) -> Optional[int]:
        if a == b:
            return 0
        seen = {a}
        dq = deque([(a, 0)])
        while dq:
            node, d = dq.popleft()
            if cutoff is not None and d >= cutoff:
                continue
            for nb in self.undirected[node]:
                if nb == b:
                    return d + 1
                if nb not in seen:
                    seen.add(nb)
                    dq.append((nb, d+1))
        return None

# ------------------------------
# Intrinsic evaluations
# ------------------------------

def edge_entailment_accuracy(edges, emb, epsilon=0.1):
    total = 0; correct = 0; margins = []
    for p, c in edges:
        if p not in emb or c not in emb: continue
        ok, m = entails_by_cone(emb[p], emb[c], epsilon)
        total += 1; correct += int(ok); margins.append(m)
    return {
        "n_edges_evaluated": total,
        "accuracy": correct / total if total > 0 else float('nan'),
        "mean_margin": float(np.mean(margins)) if margins else float('nan')
    }

def knn_purity(emb, graph, k=10, sample_nodes=500, seed=0):
    rng = random.Random(seed)
    nodes = list(graph.nodes & set(emb.keys()))
    if sample_nodes and sample_nodes < len(nodes):
        nodes = rng.sample(nodes, sample_nodes)
    purities = []
    for u in nodes:
        xu = emb[u]
        dists = {v: poincare_distance(xu, emb[v]) for v in nodes if v != u}
        nn = sorted(dists, key=dists.get)[:k]
        good = sum(1 for v in nn if graph.is_ancestor(v, u) or graph.is_descendant(v, u))
        purities.append(good / k if k > 0 else 0)
    return {"mean_purity": float(np.mean(purities))}

def sampled_graph_embed_corr(emb, graph, n_pairs=20000, seed=0):
    rng = random.Random(seed)
    nodes = list(graph.nodes & set(emb.keys()))
    pairs = [rng.sample(nodes, 2) for _ in range(min(n_pairs, len(nodes)))]
    gds, eds = [], []
    for a,b in pairs:
        gl = graph.undirected_shortest_path_len(a,b)
        if gl is None: continue
        gds.append(gl)
        eds.append(poincare_distance(emb[a], emb[b]))
    x = np.array(gds, dtype=float); y = np.array(eds, dtype=float)
    x -= x.mean(); y -= y.mean()
    denom = np.linalg.norm(x)*np.linalg.norm(y)
    r = float(np.dot(x, y)/denom) if denom>0 else float('nan')
    return {"pearson_r": r}

def radius_vs_depth_corr(emb, graph):
    depth = graph.depth()
    nodes = [n for n in graph.nodes if n in emb]
    radii = [poincare_radius(emb[n]) for n in nodes]
    depths = [depth[n] for n in nodes]
    ranks_r = np.argsort(np.argsort(radii))
    ranks_d = np.argsort(np.argsort(depths))
    rho = np.corrcoef(ranks_r, ranks_d)[0,1]
    return {"spearman_rho": float(rho)}

def evaluate_icd_embeddings(edges, embeddings, epsilon=0.1):
    graph = ICDHierarchy(edges)
    return {
        "edge_entailment": edge_entailment_accuracy(edges, embeddings, epsilon),
        "knn_purity": knn_purity(embeddings, graph),
        "graph_embed_corr": sampled_graph_embed_corr(embeddings, graph),
        "radius_vs_depth": radius_vs_depth_corr(embeddings, graph)
    }

# ------------------------------
# Loading and evaluation functions
# ------------------------------

def load_hyperbolic_cones_model(pkl_file: str):
    """
    Load hyperbolic entailment cones model from .pkl file.
    
    Args:
        pkl_file: Path to the .pkl file saved by hyperbolic_entailment_cones.py
        
    Returns:
        Dictionary containing model, embeddings, codes, id_map, and metadata
    """
    # Import here to avoid circular dependencies
    import sys
    import os
    # Add parent directory to path if needed
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import torch and model class so pickle can deserialize
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required to load the model. Please install it: pip install torch")
    
    # Import the model class and related classes
    try:
        import hyperbolic_entailment_cones
        HyperbolicEntailmentCones = getattr(hyperbolic_entailment_cones, 'HyperbolicEntailmentCones', None)
        PoincareOps = getattr(hyperbolic_entailment_cones, 'PoincareOps', None)
    except ImportError:
        # Try to import directly
        try:
            from hyperbolic_entailment_cones import HyperbolicEntailmentCones, PoincareOps
        except ImportError:
            HyperbolicEntailmentCones = None
            PoincareOps = None
    
    # Create a custom unpickler that can find classes even if saved from __main__
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Classes that might be saved from __main__ and need to be found in hyperbolic_entailment_cones
            main_module_classes = {
                'HyperbolicEntailmentCones': HyperbolicEntailmentCones,
                'PoincareOps': PoincareOps
            }
            
            # Handle the case where class was saved from __main__ module
            if module == '__main__' and name in main_module_classes:
                cls = main_module_classes[name]
                if cls is not None:
                    return cls
                # Try importing if not already done
                try:
                    import hyperbolic_entailment_cones
                    if hasattr(hyperbolic_entailment_cones, name):
                        return getattr(hyperbolic_entailment_cones, name)
                except ImportError:
                    pass
                # If still not found, raise informative error
                raise AttributeError(f"Can't get attribute '{name}' on <module '{module}'>. "
                                   f"Please ensure hyperbolic_entailment_cones module is importable.")
            
            # For other classes, try default behavior first
            try:
                return super().find_class(module, name)
            except AttributeError:
                # If it's a known class but from a different module path
                if name in main_module_classes:
                    cls = main_module_classes[name]
                    if cls is not None:
                        return cls
                    try:
                        import hyperbolic_entailment_cones
                        if hasattr(hyperbolic_entailment_cones, name):
                            return getattr(hyperbolic_entailment_cones, name)
                    except ImportError:
                        pass
                raise
    
    with open(pkl_file, 'rb') as f:
        unpickler = CustomUnpickler(f)
        save_data = unpickler.load()
    
    model = save_data['model']
    id_map = save_data['id_map']
    codes = save_data.get('codes', [])
    original_codes = save_data.get('original_codes', codes)
    eps = save_data.get('eps', 0.1)
    K = save_data.get('K', None)
    dim = save_data.get('dim', None)
    
    # Extract embeddings from model
    # model.emb is a nn.Parameter with shape (num_codes, dim)
    # Set model to eval mode to disable any dropout/batch norm if present
    model.eval()
    embeddings_tensor = model.emb.data.cpu().numpy()
    
    # Convert to dictionary: code -> embedding vector
    embeddings_dict = {}
    # Create reverse mapping: id -> code
    id_to_code = {v: k for k, v in id_map.items()}
    
    # Extract embeddings: index in embeddings_tensor corresponds to code_id
    # codes list should match the order of embeddings in the model
    num_embeddings = embeddings_tensor.shape[0]
    
    # Method 1: Use codes list if available (should match model.emb order)
    if len(codes) == num_embeddings:
        for i, code in enumerate(codes):
            embeddings_dict[code] = embeddings_tensor[i]
    
    # Method 2: Use id_map (might only include original codes)
    for code, code_id in id_map.items():
        if code_id < num_embeddings:
            embeddings_dict[code] = embeddings_tensor[code_id]
    
    # Method 3: Fill any gaps using codes list
    for i in range(min(num_embeddings, len(codes))):
        code = codes[i]
        if code not in embeddings_dict:
            embeddings_dict[code] = embeddings_tensor[i]
    
    print(f"Loaded model from: {pkl_file}")
    print(f"  Dimension: {dim}")
    print(f"  Number of codes: {len(codes)}")
    print(f"  Number of embeddings extracted: {len(embeddings_dict)}")
    print(f"  Epsilon: {eps}")
    if K is not None:
        print(f"  K: {K:.6f}")
    
    return {
        'model': model,
        'embeddings': embeddings_dict,
        'codes': codes,
        'original_codes': original_codes,
        'id_map': id_map,
        'eps': eps,
        'K': K,
        'dim': dim
    }

def build_edges_from_codes(codes: List[str]) -> List[Tuple[str, str]]:
    """
    Build parent-child edges from ICD-10 codes using hierarchy.
    This replicates the logic from hyperbolic_entailment_cones.py
    """
    def extract_icd_parent_child_pair(code: str) -> List[Tuple[str, str]]:
        """Extract parent-child pairs for an ICD-10 code."""
        if not code or len(code) < 3:
            return []
        
        pairs = []
        
        # Case 1: Code contains dots (e.g., "I11.0", "I11.01")
        if '.' in code:
            parts = code.split('.')
            prefix = parts[0]  # e.g., "I11"
            
            # Level 1: 3-char prefix (e.g., "I11" -> "I11.0")
            if len(prefix) >= 3 and len(code) > len(prefix):
                pairs.append((prefix, code))
            
            # Level 2+: For codes with suffix after dot
            if len(parts) >= 2 and len(parts[1]) > 0:
                suffix = parts[1]
                for i in range(1, len(suffix)):
                    parent_suffix = suffix[:i]
                    parent_code = f"{prefix}.{parent_suffix}"
                    if parent_code != code:
                        pairs.append((parent_code, code))
        
        # Case 2: Code without dots
        else:
            if len(code) > 3:
                parent = code[:-1]
                if len(parent) >= 3:
                    pairs.append((parent, code))
        
        return pairs
    
    pairs = []
    code_set = set(codes)
    all_codes_set = set(codes)
    
    # First pass: extract all parent-child pairs
    for code in codes:
        code_pairs = extract_icd_parent_child_pair(code)
        for parent, child in code_pairs:
            all_codes_set.add(parent)
            all_codes_set.add(child)
            if child in code_set:
                pairs.append((parent, child))
    
    # Second pass: recursively build hierarchy for intermediate parents
    changed = True
    while changed:
        changed = False
        current_all_codes = set(all_codes_set)
        for code in current_all_codes:
            if len(code) > 3 and '.' not in code:
                parent = code[:-1]
                if len(parent) >= 3:
                    if parent not in all_codes_set:
                        all_codes_set.add(parent)
                        changed = True
                    pairs.append((parent, code))
    
    # Remove duplicates
    final_pairs = []
    seen = set()
    for parent, child in pairs:
        pair = (parent, child)
        if pair not in seen:
            final_pairs.append(pair)
            seen.add(pair)
    
    return final_pairs

def evaluate_pkl_file(pkl_file: str, epsilon: Optional[float] = None, 
                     knn_k: int = 10, knn_sample_nodes: int = 500,
                     corr_n_pairs: int = 20000):
    """
    Evaluate hyperbolic embeddings from a .pkl file.
    
    Args:
        pkl_file: Path to the .pkl file
        epsilon: Epsilon parameter for entailment cones (defaults to saved value)
        knn_k: k for k-NN purity evaluation
        knn_sample_nodes: Number of nodes to sample for k-NN purity
        corr_n_pairs: Number of pairs for graph-embedding correlation
        
    Returns:
        Dictionary of evaluation results
    """
    # Load model and embeddings
    data = load_hyperbolic_cones_model(pkl_file)
    embeddings = data['embeddings']
    codes = data['codes']
    eps = epsilon if epsilon is not None else data['eps']
    
    # Build edges from codes
    print("\nBuilding parent-child edges from code hierarchy...")
    edges = build_edges_from_codes(codes)
    print(f"Built {len(edges)} parent-child edges")
    
    if len(edges) == 0:
        print("Warning: No edges found. Evaluation will be limited.")
        # Try to build edges from original_codes if available
        if 'original_codes' in data and len(data['original_codes']) > 0:
            print("Trying with original codes...")
            edges = build_edges_from_codes(data['original_codes'])
            print(f"Built {len(edges)} edges from original codes")
    
    if len(edges) == 0:
        print("Error: Could not build any edges. Cannot evaluate edge entailment.")
        return None
    
    # Filter edges to only include codes that have embeddings
    filtered_edges = [(p, c) for p, c in edges if p in embeddings and c in embeddings]
    print(f"Filtered to {len(filtered_edges)} edges with embeddings")
    
    if len(filtered_edges) == 0:
        print("Error: No edges have valid embeddings. Cannot evaluate.")
        return None
    
    # Evaluate
    print(f"\nEvaluating embeddings with epsilon={eps:.4f}...")
    print("=" * 60)
    
    results = evaluate_icd_embeddings(
        edges=filtered_edges,
        embeddings=embeddings,
        epsilon=eps
    )
    
    # Override k-NN parameters if needed
    if knn_k != 10 or knn_sample_nodes != 500:
        graph = ICDHierarchy(filtered_edges)
        results['knn_purity'] = knn_purity(
            embeddings, graph, k=knn_k, sample_nodes=knn_sample_nodes
        )
    
    # Override correlation parameters if needed
    if corr_n_pairs != 20000:
        graph = ICDHierarchy(filtered_edges)
        results['graph_embed_corr'] = sampled_graph_embed_corr(
            embeddings, graph, n_pairs=corr_n_pairs
        )
    
    return results

def print_evaluation_results(results: Dict):
    """Print evaluation results in a readable format."""
    if results is None:
        print("No results to print.")
        return
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Edge entailment accuracy
    if 'edge_entailment' in results:
        ee = results['edge_entailment']
        print(f"\n1. Edge Entailment Accuracy:")
        print(f"   Number of edges evaluated: {ee['n_edges_evaluated']}")
        print(f"   Accuracy: {ee['accuracy']:.4f} ({ee['accuracy']*100:.2f}%)")
        print(f"   Mean margin: {ee['mean_margin']:.6f}")
    
    # K-NN purity
    if 'knn_purity' in results:
        kp = results['knn_purity']
        print(f"\n2. K-NN Purity:")
        print(f"   Mean purity: {kp['mean_purity']:.4f} ({kp['mean_purity']*100:.2f}%)")
    
    # Graph-embedding correlation
    if 'graph_embed_corr' in results:
        ge = results['graph_embed_corr']
        print(f"\n3. Graph-Embedding Correlation:")
        print(f"   Pearson r: {ge['pearson_r']:.6f}")
        if ge['pearson_r'] > 0:
            print(f"   (Positive correlation: longer graph distance -> larger embedding distance)")
        elif ge['pearson_r'] < 0:
            print(f"   (Negative correlation: longer graph distance -> smaller embedding distance)")
        else:
            print(f"   (No correlation)")
    
    # Radius vs depth correlation
    if 'radius_vs_depth' in results:
        rd = results['radius_vs_depth']
        print(f"\n4. Radius vs Depth Correlation:")
        print(f"   Spearman rho: {rd['spearman_rho']:.6f}")
        if rd['spearman_rho'] > 0:
            print(f"   (Positive correlation: deeper nodes -> larger radius from origin)")
        elif rd['spearman_rho'] < 0:
            print(f"   (Negative correlation: deeper nodes -> smaller radius from origin)")
        else:
            print(f"   (No correlation)")
    
    print("\n" + "=" * 60)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate hyperbolic entailment cones embeddings from .pkl file'
    )
    
    parser.add_argument(
        '--pkl_file',
        type=str,
        required=True,
        help='Path to the .pkl file generated by hyperbolic_entailment_cones.py'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=None,
        help='Epsilon parameter for entailment cones (defaults to saved value in .pkl file)'
    )
    
    parser.add_argument(
        '--knn_k',
        type=int,
        default=10,
        help='k for k-NN purity evaluation (default: 10)'
    )
    
    parser.add_argument(
        '--knn_sample_nodes',
        type=int,
        default=500,
        help='Number of nodes to sample for k-NN purity (default: 500)'
    )
    
    parser.add_argument(
        '--corr_n_pairs',
        type=int,
        default=20000,
        help='Number of pairs for graph-embedding correlation (default: 20000)'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Optional: Save results to JSON file'
    )
    
    return parser.parse_args()

# ------------------------------
# Main execution
# ------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    # Evaluate
    results = evaluate_pkl_file(
        pkl_file=args.pkl_file,
        epsilon=args.epsilon,
        knn_k=args.knn_k,
        knn_sample_nodes=args.knn_sample_nodes,
        corr_n_pairs=args.corr_n_pairs
    )
    
    # Print results
    print_evaluation_results(results)
    
    # Save to file if requested
    if args.output_file and results is not None:
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            return obj
        
        json_results = convert_to_json_serializable(results)
        
        with open(args.output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {args.output_file}")
    
    if results is not None:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed. Please check the error messages above.")
        exit(1)
