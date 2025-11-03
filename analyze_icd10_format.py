"""
分析ICD-10-CM代码格式的脚本
统计代码中包含的字符类型、长度分布等信息
"""
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Set
import re


def load_icd10_codes(icd10_file_path: str) -> List[str]:
    """
    加载所有ICD-10代码
    
    Args:
        icd10_file_path: ICD-10代码文件路径
        
    Returns:
        代码列表
    """
    all_codes = []
    with open(icd10_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # 提取代码（第一个空格或制表符之前的部分）
                code = line.split()[0]
                all_codes.append(code)
    
    return all_codes


def analyze_code_format(codes: List[str]) -> Dict:
    """
    分析ICD-10代码格式特征
    
    Args:
        codes: 代码列表
        
    Returns:
        包含各种统计信息的字典
    """
    stats = {
        'total_codes': len(codes),
        'contains_dot': 0,
        'contains_dash': 0,
        'contains_slash': 0,
        'contains_parentheses': 0,
        'contains_other_punctuation': 0,
        'no_punctuation': 0,
        'code_lengths': Counter(),
        'prefix_patterns': Counter(),  # 前3个字符的模式
        'dot_positions': Counter(),  # 小数点位置
        'dash_patterns': Counter(),  # 破折号模式
        'sample_codes_by_type': defaultdict(list)
    }
    
    punctuation_chars = set()
    
    for code in codes:
        # 统计长度
        stats['code_lengths'][len(code)] += 1
        
        # 提取前缀（前3个字符，如果长度>=3）
        if len(code) >= 3:
            prefix = code[:3]
            stats['prefix_patterns'][prefix] += 1
        
        # 检查是否包含各种标点符号
        has_dot = '.' in code
        has_dash = '-' in code
        has_slash = '/' in code
        has_parentheses = '(' in code or ')' in code
        
        if has_dot:
            stats['contains_dot'] += 1
            # 记录小数点位置（小数点后有多少位）
            dot_idx = code.find('.')
            if dot_idx != -1:
                suffix_len = len(code) - dot_idx - 1
                stats['dot_positions'][suffix_len] += 1
            if len(stats['sample_codes_by_type']['with_dot']) < 10:
                stats['sample_codes_by_type']['with_dot'].append(code)
        
        if has_dash:
            stats['contains_dash'] += 1
            # 记录破折号模式（位置或周围字符）
            dash_idx = code.find('-')
            if dash_idx != -1:
                if dash_idx == len(code) - 1:
                    pattern = 'dash_at_end'
                elif dash_idx == 0:
                    pattern = 'dash_at_start'
                else:
                    # 记录破折号前后的字符数
                    before = dash_idx
                    after = len(code) - dash_idx - 1
                    pattern = f'before{dash_idx}_after{after}'
                stats['dash_patterns'][pattern] += 1
            if len(stats['sample_codes_by_type']['with_dash']) < 10:
                stats['sample_codes_by_type']['with_dash'].append(code)
        
        if has_slash:
            stats['contains_slash'] += 1
            if len(stats['sample_codes_by_type']['with_slash']) < 10:
                stats['sample_codes_by_type']['with_slash'].append(code)
        
        if has_parentheses:
            stats['contains_parentheses'] += 1
            if len(stats['sample_codes_by_type']['with_parentheses']) < 10:
                stats['sample_codes_by_type']['with_parentheses'].append(code)
        
        # 收集所有标点符号
        for char in code:
            if not (char.isalnum() or char == '.' or char == '-' or char == '/' or char == '(' or char == ')'):
                punctuation_chars.add(char)
        
        # 检查是否完全没有标点符号
        if not (has_dot or has_dash or has_slash or has_parentheses):
            stats['no_punctuation'] += 1
            if len(stats['sample_codes_by_type']['no_punctuation']) < 10:
                stats['sample_codes_by_type']['no_punctuation'].append(code)
    
    stats['other_punctuation_chars'] = sorted(list(punctuation_chars))
    stats['contains_other_punctuation'] = len(stats['other_punctuation_chars'])
    
    return stats


def print_analysis_results(stats: Dict):
    """
    打印分析结果
    
    Args:
        stats: 统计信息字典
    """
    print("=" * 80)
    print("ICD-10-CM 代码格式分析结果")
    print("=" * 80)
    
    print(f"\n总代码数: {stats['total_codes']:,}")
    
    print("\n" + "-" * 80)
    print("标点符号统计:")
    print("-" * 80)
    print(f"  包含小数点 (.): {stats['contains_dot']:,} ({stats['contains_dot']/stats['total_codes']*100:.2f}%)")
    print(f"  包含破折号 (-): {stats['contains_dash']:,} ({stats['contains_dash']/stats['total_codes']*100:.2f}%)")
    print(f"  包含斜杠 (/): {stats['contains_slash']:,} ({stats['contains_slash']/stats['total_codes']*100:.2f}%)")
    print(f"  包含括号 (): {stats['contains_parentheses']:,} ({stats['contains_parentheses']/stats['total_codes']*100:.2f}%)")
    print(f"  完全不包含标点: {stats['no_punctuation']:,} ({stats['no_punctuation']/stats['total_codes']*100:.2f}%)")
    
    if stats['contains_other_punctuation'] > 0:
        print(f"  其他标点符号: {', '.join(stats['other_punctuation_chars'])}")
    
    print("\n" + "-" * 80)
    print("代码长度分布 (Top 20):")
    print("-" * 80)
    for length, count in stats['code_lengths'].most_common(20):
        print(f"  长度 {length:2d}: {count:8,} ({count/stats['total_codes']*100:6.2f}%)")
    
    if stats['contains_dot'] > 0:
        print("\n" + "-" * 80)
        print("小数点位置统计 (小数点后的位数):")
        print("-" * 80)
        for suffix_len, count in sorted(stats['dot_positions'].most_common()):
            print(f"  小数点后 {suffix_len} 位: {count:8,} ({count/stats['contains_dot']*100:6.2f}%)")
    
    if stats['contains_dash'] > 0:
        print("\n" + "-" * 80)
        print("破折号模式统计 (Top 10):")
        print("-" * 80)
        for pattern, count in stats['dash_patterns'].most_common(10):
            print(f"  {pattern:30s}: {count:8,} ({count/stats['contains_dash']*100:6.2f}%)")
    
    print("\n" + "-" * 80)
    print("代码前缀统计 (Top 20):")
    print("-" * 80)
    print(f"  长度为3的前缀种类数: {len(stats['prefix_patterns']):,}")
    for prefix, count in stats['prefix_patterns'].most_common(20):
        print(f"  {prefix:10s}: {count:8,} ({count/stats['total_codes']*100:6.2f}%)")
    
    print("\n" + "-" * 80)
    print("各类型代码示例:")
    print("-" * 80)
    for code_type, samples in stats['sample_codes_by_type'].items():
        if samples:
            print(f"\n{code_type}:")
            for sample in samples[:10]:
                print(f"  {sample}")
    
    # 统计组合类型
    print("\n" + "-" * 80)
    print("代码格式组合统计:")
    print("-" * 80)
    codes_with_dot_only = sum(1 for code in stats['sample_codes_by_type']['with_dot'] 
                              if '.' in code and '-' not in code and '/' not in code)
    codes_with_dash_only = sum(1 for code in stats['sample_codes_by_type']['with_dash'] 
                              if '-' in code and '.' not in code and '/' not in code)
    codes_with_both = sum(1 for code in stats['sample_codes_by_type']['with_dot'] 
                         if '.' in code and '-' in code)
    
    print(f"  仅包含小数点: ~{codes_with_dot_only} (示例)")
    print(f"  仅包含破折号: ~{codes_with_dash_only} (示例)")
    print(f"  同时包含小数点和破折号: ~{codes_with_both} (示例)")


def analyze_code_structure(codes: List[str]):
    """
    更详细的结构分析
    
    Args:
        codes: 代码列表
    """
    print("\n" + "=" * 80)
    print("详细结构分析")
    print("=" * 80)
    
    # 分析代码的模式（字母+数字组合）
    patterns = Counter()
    
    for code in codes[:1000]:  # 只分析前1000个以减少计算量
        # 将代码转换为模式（L=字母, D=数字, P=标点）
        pattern = []
        for char in code:
            if char.isalpha():
                pattern.append('L')
            elif char.isdigit():
                pattern.append('D')
            else:
                pattern.append(char)  # 保留标点符号
        pattern_str = ''.join(pattern)
        patterns[pattern_str] += 1
    
    print("\n代码模式统计 (Top 20):")
    print("-" * 80)
    print("  L=字母, D=数字, 标点符号原样显示")
    for pattern, count in patterns.most_common(20):
        print(f"  {pattern:30s}: {count:4d}")


def main():
    parser = argparse.ArgumentParser(description='分析ICD-10-CM代码格式')
    parser.add_argument('--icd10_file', type=str,
                       default="/data/yuyu/data/MIMIC_IV/icd10cm-codes-April-2024.txt",
                       help='ICD-10代码文件路径')
    parser.add_argument('--detailed', action='store_true',
                       help='显示详细的结构分析')
    
    args = parser.parse_args()
    
    print(f"正在加载ICD-10代码文件: {args.icd10_file}")
    codes = load_icd10_codes(args.icd10_file)
    print(f"成功加载 {len(codes):,} 个代码")
    
    print("\n正在分析代码格式...")
    stats = analyze_code_format(codes)
    
    print_analysis_results(stats)
    
    if args.detailed:
        analyze_code_structure(codes)
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()

