from typing import List, Set, Optional, Tuple


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import pickle

# --------- 双曲空间基础运算：Poincaré 盘模型 ---------

def mobius_add(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Möbius addition on the 2D Poincaré disk.
    u, v: shape (2,)
    """
    u2 = np.dot(u, u)
    v2 = np.dot(v, v)
    uv = np.dot(u, v)
    denom = 1.0 + 2.0 * uv + u2 * v2
    # 防止数值爆炸
    if denom == 0:
        denom = 1e-8

    num = (1.0 + 2.0 * uv + v2) * u + (1.0 - u2) * v
    return num / denom


def reflect_to_origin(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    把点 a 映射到原点的等距变换：x ↦ (-a) ⊕ x
    """
    return mobius_add(-a, x)


def reflect_from_origin(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    反变换：x ↦ a ⊕ x
    """
    return mobius_add(a, x)


def proj_inside_unit_disk(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    确保点在单位圆盘内部，避免数值精度导致 ||x|| >= 1.
    """
    norm = np.linalg.norm(x)
    if norm >= 1.0:
        x = x / (norm + eps) * (1.0 - eps)
    return x


# --------- Sarkar 嵌入在 Trie 上的实现 ---------

def sarkar_embed_trie(trie: "CodeTrie",
                      tau: float = 1.0) -> Dict["TrieNode", np.ndarray]:
    embeddings: Dict["TrieNode", np.ndarray] = {}

    root = trie.root
    embeddings[root] = np.zeros(2, dtype=float)

    r = np.tanh(tau / 2.0)

    def dfs(node: "TrieNode",
            parent: Optional["TrieNode"] = None):
        p = embeddings[node]
        children = list(node.children.values())
        if not children:
            return

        deg = len(children)

        # 计算参考方向 base_angle
        if parent is None:
            # 根节点：随便选一个参考方向，比如 0
            base_angle = 0.0
        else:
            # 在以当前节点为中心的坐标系下，看父节点的方向
            parent_pos = embeddings[parent]
            # 把当前节点映射到原点：node -> 0
            parent_in_local = reflect_to_origin(p, parent_pos)
            # 计算父节点在本地坐标系中的极角
            base_angle = np.angle(parent_in_local[0] + 1j * parent_in_local[1])

        # 为了让孩子远离父节点来的方向，+π
        start_angle = base_angle + np.pi

        for i, child in enumerate(children):
            theta = start_angle + 2.0 * np.pi * (i / deg)
            # 在“node 为原点”的局部坐标系中构造子节点
            y_local = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=float)
            # 再变回全局坐标：node_pos ⊕ y_local
            child_pos = reflect_from_origin(p, y_local)
            child_pos = proj_inside_unit_disk(child_pos)

            embeddings[child] = child_pos
            dfs(child, node)

    dfs(root, parent=None)
    return embeddings


class TrieNode:
    """Node in the Trie data structure."""
    
    def __init__(self, code: Optional[str] = None):
        self.code = code  # The code at this node (e.g., "A04")
        self.children = {}  # Dictionary mapping code -> TrieNode
        self.is_end = False  # True if this node represents the end of a valid code sequence
        self.full_sequence = None  # Store the full code sequence ending at this node


class CodeTrie:
    """
    Trie data structure for hierarchical medical codes.
    
    Each code sequence is stored as a path in the trie. For example,
    "A04 A047 A0471" creates nodes: A04 -> A047 -> A0471.
    """
    
    def __init__(self):
        """Initialize an empty Trie."""
        self.root = TrieNode()
        self._size = 0  # Number of code sequences stored
    
    def insert(self, code_sequence: List[str]) -> None:
        """
        Insert a code sequence into the trie.
        
        Args:
            code_sequence: List of codes in hierarchical order, e.g., ["A04", "A047", "A0471"]
        """
        node = self.root
        
        for code in code_sequence:
            if code not in node.children:
                node.children[code] = TrieNode(code)
            node = node.children[code]
        
        # Mark the end of the sequence and store the full sequence
        if not node.is_end:
            node.is_end = True
            node.full_sequence = code_sequence.copy()
            self._size += 1
    
    def search(self, code_sequence: List[str]) -> bool:
        """
        Check if a code sequence exists in the trie.
        
        Args:
            code_sequence: List of codes to search for
            
        Returns:
            True if the exact sequence exists, False otherwise
        """
        node = self.root
        
        for code in code_sequence:
            if code not in node.children:
                return False
            node = node.children[code]
        
        return node.is_end
    
    def search_prefix(self, prefix: List[str]) -> List[List[str]]:
        """
        Find all code sequences that start with the given prefix.
        
        Args:
            prefix: List of codes representing the prefix, e.g., ["A04", "A047"]
            
        Returns:
            List of all full code sequences that start with the prefix
        """
        node = self.root
        
        # Navigate to the prefix node
        for code in prefix:
            if code not in node.children:
                return []  # Prefix doesn't exist
            node = node.children[code]
        
        # Collect all sequences starting from this node
        results = []
        self._collect_sequences(node, results)
        return results
    
    def _collect_sequences(self, node: TrieNode, results: List[List[str]]) -> None:
        """Helper method to recursively collect all sequences from a node."""
        if node.is_end and node.full_sequence:
            results.append(node.full_sequence)
        
        for child in node.children.values():
            self._collect_sequences(child, results)
    
    def find_matching_codes(self, code_list: List[str]) -> List[List[str]]:
        """
        Find all code sequences in the trie that match any code in the given list.
        
        This is useful for checking if any code from a patient's diagnosis list
        matches any sequence in the trie.
        
        Args:
            code_list: List of codes to check, e.g., ["A04", "A0471", "B12"]
            
        Returns:
            List of matching full code sequences
        """
        matches = set()
        
        # Check each code in the list
        for code in code_list:
            # Try to find sequences starting with this code
            if code in self.root.children:
                sequences = self.search_prefix([code])
                for seq in sequences:
                    # Check if all codes in the sequence are in the input list
                    if all(c in code_list for c in seq):
                        matches.add(tuple(seq))
        
        return [list(seq) for seq in matches]
    
    def get_all_sequences(self) -> List[List[str]]:
        """
        Get all code sequences stored in the trie.
        
        Returns:
            List of all code sequences
        """
        results = []
        self._collect_sequences(self.root, results)
        return results
    
    def size(self) -> int:
        """Return the number of code sequences in the trie."""
        return self._size
    
    @classmethod
    def from_file(cls, filepath: str) -> 'CodeTrie':
        """
        Create a CodeTrie from a text file.
        
        The file should have one code sequence per line, with codes separated by spaces.
        Example line: "A04 A047 A0471"
        
        Args:
            filepath: Path to the text file
            
        Returns:
            A CodeTrie instance loaded with data from the file
        """
        trie = cls()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Split the line into individual codes
                code_sequence = line.split()
                if code_sequence:
                    trie.insert(code_sequence)
        
        return trie
    
    def __len__(self) -> int:
        """Return the number of code sequences in the trie."""
        return self._size
    
    def __contains__(self, code_sequence: List[str]) -> bool:
        """Check if a code sequence is in the trie."""
        return self.search(code_sequence)


def main():
    print("Loading code trie from cond_hist_codes.txt...")
    trie = CodeTrie.from_file("cond_hist_codes.txt")
    print(f"Loaded {len(trie)} code sequences\n")
    
    embeddings = sarkar_embed_trie(trie, tau=1.5)
    
    # Convert TrieNode -> np.ndarray to code -> [x, y]
    # Only save code -> embedding mapping for individual codes
    code_embeddings = {}  # code -> embedding
    
    def collect_code_embeddings(node: TrieNode):
        """Recursively collect embeddings for all code nodes"""
        if node not in embeddings:
            return
        
        # If this node has a code, save its embedding
        # Use the first occurrence of each code (don't overwrite)
        if node.code and node.code not in code_embeddings:
            coords = embeddings[node].tolist()
            code_embeddings[node.code] = coords
        
        # Recursively process children
        for child_node in node.children.values():
            collect_code_embeddings(child_node)
    
    collect_code_embeddings(trie.root)
    
    # Save to pickle file
    output_path = "sarkar_embeddings.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(code_embeddings, f)


if __name__ == "__main__":
    main()
