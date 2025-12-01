from typing import List, Set, Optional, Tuple


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import pickle


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


if __name__ == "__main__":
    main()
