import pickle
import sys
import numpy as np


def load_pkl_file(pkl_file: str):
    """
    加载.pkl文件（支持自定义类的导入）
    
    Returns:
        加载的数据（可能是字典或对象）
    """
    # 先尝试导入hyperbolic_entailment_cones模块（如果需要）
    # 这样可以确保pickle能够找到HyperbolicEntailmentCones类
    try:
        import hyperbolic_entailment_cones
        if 'hyperbolic_entailment_cones' not in sys.modules:
            sys.modules['hyperbolic_entailment_cones'] = hyperbolic_entailment_cones
    except ImportError:
        # 如果直接导入失败，尝试使用importlib
        try:
            spec = importlib.util.find_spec("hyperbolic_entailment_cones")
            if spec is not None:
                hyperbolic_module = importlib.util.module_from_spec(spec)
                sys.modules['hyperbolic_entailment_cones'] = hyperbolic_module
                spec.loader.exec_module(hyperbolic_module)
        except Exception as e:
            # 如果导入失败，继续尝试加载（可能会失败，但会给出更明确的错误）
            pass
    
    # 创建自定义Unpickler，可以处理HyperbolicEntailmentCones类
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # 处理HyperbolicEntailmentCones类
            if name == 'HyperbolicEntailmentCones':
                # 首先尝试从hyperbolic_entailment_cones模块获取
                if 'hyperbolic_entailment_cones' in sys.modules:
                    mod = sys.modules['hyperbolic_entailment_cones']
                    if hasattr(mod, 'HyperbolicEntailmentCones'):
                        return mod.HyperbolicEntailmentCones
                # 如果还没有导入，尝试导入
                try:
                    import hyperbolic_entailment_cones
                    if hasattr(hyperbolic_entailment_cones, 'HyperbolicEntailmentCones'):
                        return hyperbolic_entailment_cones.HyperbolicEntailmentCones
                except ImportError:
                    pass
                # 如果模块名是__main__，尝试从hyperbolic_entailment_cones模块获取
                if module == '__main__':
                    if 'hyperbolic_entailment_cones' in sys.modules:
                        mod = sys.modules['hyperbolic_entailment_cones']
                        if hasattr(mod, 'HyperbolicEntailmentCones'):
                            return mod.HyperbolicEntailmentCones
                    try:
                        import hyperbolic_entailment_cones
                        if hasattr(hyperbolic_entailment_cones, 'HyperbolicEntailmentCones'):
                            return hyperbolic_entailment_cones.HyperbolicEntailmentCones
                    except ImportError:
                        pass
            
            # 对于其他类，先尝试默认行为
            try:
                return super().find_class(module, name)
            except AttributeError:
                # 如果是__main__模块中的HyperbolicEntailmentCones，尝试从hyperbolic_entailment_cones模块获取
                if module == '__main__' and name == 'HyperbolicEntailmentCones':
                    if 'hyperbolic_entailment_cones' in sys.modules:
                        mod = sys.modules['hyperbolic_entailment_cones']
                        if hasattr(mod, 'HyperbolicEntailmentCones'):
                            return mod.HyperbolicEntailmentCones
                    try:
                        import hyperbolic_entailment_cones
                        if hasattr(hyperbolic_entailment_cones, 'HyperbolicEntailmentCones'):
                            return hyperbolic_entailment_cones.HyperbolicEntailmentCones
                    except ImportError:
                        pass
                raise
    
    # 使用CustomUnpickler加载文件
    with open(pkl_file, 'rb') as f:
        unpickler = CustomUnpickler(f)
        data = unpickler.load()
    
    return data


def load_icd10_codes(icd10_file_path: str):
    """
    Load all ICD-10 condition codes from the MIMIC-IV file
    
    Args:
        icd10_file_path: Path to the ICD-10 codes file
        
    Returns:
        List of ICD-10 condition codes
    """
    all_conditions = []
    with open(icd10_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract the code (first part before tab/space)
                code = line.split()[0]
                all_conditions.append(code)
    
    return all_conditions


def extract_icd_parent_child_pair(code: str):
    """
    Extract parent-child pairs for an ICD-10 code based on hierarchy.
    Supports both formats:
    - With dots: "I11.0" -> [("I11", "I11.0")]
                 "I11.01" -> [("I11", "I11.0"), ("I11.0", "I11.01")]
    - Without dots: "I110" -> [("I11", "I110")]
                   "I1101" -> [("I11", "I110"), ("I110", "I1101")]
    
    ICD-10 hierarchy: 
    - 3 chars: "I11" (category)
    - With dots: 5 chars "I11.0" (subcategory), 6+ chars "I11.01" (more specific)
    - Without dots: 4 chars "I110" (subcategory), 5+ chars "I1101" (more specific)
    
    This function generates ALL possible parent-child relationships based on the code structure,
    even if the parent codes are not in the original code list.
    
    Args:
        code: ICD-10 code string (may or may not contain dots)
        
    Returns:
        List of (parent, child) tuples
    """
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
        
        # Level 2+: For codes with suffix after dot (e.g., "I11.01")
        if len(parts) >= 2 and len(parts[1]) > 0:
            suffix = parts[1]
            # Generate parents by shortening the suffix
            for i in range(1, len(suffix)):
                parent_suffix = suffix[:i]
                parent_code = f"{prefix}.{parent_suffix}"
                if parent_code != code:
                    pairs.append((parent_code, code))
    
    # Case 2: Code without dots (e.g., "I110", "I1101")
    else:
        # Build hierarchy by incrementally increasing length
        # Only generate DIRECT parent-child relationships (adjacent levels)
        # e.g., "I1101" should generate:
        #   - ("I110", "I1101") - direct parent
        # But we also need to track all intermediate levels for the hierarchy
        # The direct parent is always the code with one less character
        if len(code) > 3:
            # Find the direct parent (one character shorter)
            # For ICD-10 codes, the parent is typically the code minus the last character
            # But we need to be careful: "A0100" -> parent is "A010" (not "A010" -> "A01" directly for "A0100")
            parent = code[:-1]  # Remove last character to get direct parent
            if len(parent) >= 3:  # Ensure parent is at least 3 characters (ICD-10 minimum)
                pairs.append((parent, code))
    
    return pairs


def build_parent_child_edges_from_codes(codes):
    """
    Build parent-child edges for all ICD-10 codes based on their hierarchy.
    Includes parent codes even if they're not in the original code set.
    
    Args:
        codes: List of ICD codes
        
    Returns:
        Tuple of (edges, all_codes):
        - edges: List of (parent, child) tuples representing hierarchy relationships
        - all_codes: List of all codes including original codes and generated parent codes
    """
    pairs = []
    code_set = set(codes)
    all_codes_set = set(codes)  # Track all codes including parents
    
    # First pass: extract all parent-child pairs and collect all parent codes
    # We need to iteratively build the hierarchy because intermediate parents might be missing
    for code in codes:
        code_pairs = extract_icd_parent_child_pair(code)
        for parent, child in code_pairs:
            # Add both parent and child to all_codes_set
            all_codes_set.add(parent)
            all_codes_set.add(child)
            # Only keep pairs where child is in the original code set
            # This ensures we build hierarchy even if parents are not explicitly in the list
            if child in code_set:
                pairs.append((parent, child))
    
    # Second pass: recursively build hierarchy for intermediate parents
    # This ensures we have complete chains: A01 -> A010 -> A0100
    # We need to process all_codes_set recursively to build the full hierarchy
    changed = True
    while changed:
        changed = False
        current_all_codes = set(all_codes_set)
        for code in current_all_codes:
            if len(code) > 3 and '.' not in code:
                # Check if this code itself has a parent
                parent = code[:-1]
                if len(parent) >= 3:
                    if parent not in all_codes_set:
                        # Add the intermediate parent
                        all_codes_set.add(parent)
                        changed = True
                    # Add the parent-child relationship (parent -> code)
                    # We add this edge if code is in all_codes_set (either original or generated)
                    pairs.append((parent, code))
    
    # Remove duplicates
    final_pairs = []
    seen = set()
    for parent, child in pairs:
        pair = (parent, child)
        if pair not in seen:
            final_pairs.append(pair)
            seen.add(pair)
    
    # Return all codes (original + generated parents) sorted for consistent ordering
    all_codes_list = sorted(list(all_codes_set))
    
    return final_pairs, all_codes_list


# ---------- geometry: Xi and psi (Poincaré) ----------
def xi_poincare(p, c):
    '''双曲锥的开口角'''
    # p, c: shape (d,)
    # 与训练代码中的 angle_Xi 保持一致
    p2 = np.dot(p, p)
    c2 = np.dot(c, c)
    pc = np.dot(p, c)
    
    num = pc * (1 + p2) - p2 * (1 + c2)
    
    # 使用 clamp 保护数值稳定性，与训练代码一致
    p_minus_c = p - c
    p_minus_c_norm = np.linalg.norm(p_minus_c)
    p_minus_c_norm = max(p_minus_c_norm, 1e-15)  # 对应训练代码中的 clamp(min=1e-15)
    
    p_norm = np.sqrt(p2)
    p_norm = max(p_norm, 1e-15)  # 对应训练代码中的 clamp(min=1e-15)
    
    inside = 1 + p2 * c2 - 2 * pc
    inside = max(inside, 1e-15)  # 对应训练代码中的 clamp(min=1e-15)
    
    den = p_norm * p_minus_c_norm * np.sqrt(inside)
    
    # 与训练代码保持一致：clamp 到 [-1.0 + 1e-7, 1.0 - 1e-7]
    val = num / (den + 1e-15)
    val = np.clip(val, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(val)  # Xi(p,c)


def psi_from_norm_poincare(norm_p, K, eps):
    '''以 p 为顶点的双曲锥体的半角'''
    # psi(p) = arcsin( K * (1 - ||p||^2) / ||p|| )
    # 与训练代码中的 psi 保持一致：使用 eps 来 clamp 范数的最小值
    norm_p = max(norm_p, eps)  # 对应训练代码中的 clamp(min=eps)
    arg = K * (1.0 - norm_p * norm_p) / (norm_p + 1e-15)
    # 与训练代码保持一致：clamp 到 [-1.0 + 1e-7, 1.0 - 1e-7]
    arg = np.clip(arg, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arcsin(arg)


def in_cone(p, c, K, eps):
    return xi_poincare(p, c) <= psi_from_norm_poincare(np.linalg.norm(p), K, eps) + 1e-12


if __name__ == "__main__":
    save_data = load_pkl_file("hyperbolic_cones_embeddings.pkl")
    id_map = save_data['id_map']
    
    # 从保存的数据中获取参数
    K = save_data['K']
    eps = save_data.get('eps', 0.1)  # 如果不存在则使用默认值 0.1
    
    print(f"Using parameters: K={K:.6f}, eps={eps:.6f}")
    print()
    
    code_pairs = [("E1131", "E11319"), ("E113", "E1131"), ("E11", "E113"), ("E11", "E11319"), ("E113", "E11319")]
    for code1, code2 in code_pairs:
        # 将 torch tensor 转换为 numpy array
        embedding1 = save_data['model'].emb.data[id_map[code1]].cpu().clone().numpy()
        embedding2 = save_data['model'].emb.data[id_map[code2]].cpu().clone().numpy()
        
        # 传递 eps 参数
        result = in_cone(embedding1, embedding2, K, eps)
        print(f"{code2} in {code1} cone: {result}")
