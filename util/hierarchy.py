"""
Hierarchy preprocessing module for ICD-10-CM codes
Extracts ancestor paths from code prefixes to build hierarchical structure
"""
from collections import defaultdict


def extract_ancestor_path(code):
    """
    Extract ancestor path from ICD-10-CM code based on prefixes.
    For code "I10011", extract prefixes with length >= 3 excluding length 2:
    ["I", "I10", "I100", "I1001", "I10011"]
    
    Args:
        code: ICD-10-CM code string (e.g., "I10011")
    
    Returns:
        List of ancestor codes from root to leaf (inclusive)
    """
    if not code or len(code) < 1:
        return []
    
    path = []
    # Always include the root (first character)
    path.append(code[0])
    
    # Extract prefixes of length 3, 4, 5, ... (skip length 2)
    for i in range(3, len(code) + 1):
        prefix = code[:i]
        path.append(prefix)
    
    return path


def build_ancestor_paths(vocab):
    """
    Build ancestor path dictionary for all codes in vocabulary.
    
    Args:
        vocab: Dictionary mapping code strings to indices (stoi) or list of code strings
    
    Returns:
        Dictionary mapping code -> list of ancestor codes [root, ..., code]
    """
    # Handle both dict (stoi) and list inputs
    if isinstance(vocab, dict):
        codes = list(vocab.keys())
    else:
        codes = vocab
    
    ancestors_dict = {}
    for code in codes:
        ancestors_dict[code] = extract_ancestor_path(code)
    
    return ancestors_dict


def get_max_depth(vocab):
    """
    Get maximum depth needed for embedding tables based on longest code.
    
    Args:
        vocab: Dictionary mapping code strings to indices (stoi) or list of code strings
    
    Returns:
        Maximum depth (number of levels) needed
    """
    if isinstance(vocab, dict):
        codes = list(vocab.keys())
    else:
        codes = vocab
    
    if not codes:
        return 1
    
    max_code_len = max(len(code) for code in codes)
    # Depth = 1 (root) + number of additional levels (length 3, 4, ..., max_len)
    # For length n: depth = 1 + max(0, n - 2)  (since we skip length 2)
    max_depth = 1 + max(0, max_code_len - 2)
    
    return max_depth


def build_level_vocabularies(ancestors_dict):
    """
    Build separate vocabularies for each level in the hierarchy.
    
    Args:
        ancestors_dict: Dictionary mapping code -> list of ancestor codes
    
    Returns:
        Dictionary mapping level_index -> set of codes at that level
    """
    level_vocabs = defaultdict(set)
    
    for code, ancestors in ancestors_dict.items():
        for level_idx, ancestor_code in enumerate(ancestors):
            level_vocabs[level_idx].add(ancestor_code)
    
    # Convert sets to sorted lists for consistent ordering
    level_vocabs = {level: sorted(list(codes)) for level, codes in level_vocabs.items()}
    
    return level_vocabs

