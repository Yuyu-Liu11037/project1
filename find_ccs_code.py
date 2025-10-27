#!/usr/bin/env python3
"""
查找 ICD-10-CM 代码对应的 CCS code
Usage:
    python find_ccs_code.py I25110
    python find_ccs_code.py  # 交互模式
"""
import sys
from pyhealth.medcode import CrossMap

def find_ccs_code(icd10_code):
    """查找单个 ICD-10-CM 代码对应的 CCS code"""
    mapping = CrossMap("ICD10CM", "CCSCM")
    ccs_result = mapping.map(icd10_code)
    
    print(f"ICD-10-CM 代码: {icd10_code}")
    if ccs_result:
        print(f"对应的 CCS code: {ccs_result}")
        print(f"主要 CCS code: {ccs_result[0]}")
    else:
        print(f"未找到对应的 CCS code")
    
    return ccs_result

def find_ccs_codes_batch(icd10_codes):
    """批量查找多个 ICD-10-CM 代码对应的 CCS code"""
    mapping = CrossMap("ICD10CM", "CCSCM")
    
    results = []
    for icd10_code in icd10_codes:
        ccs_result = mapping.map(icd10_code)
        if ccs_result:
            results.append((icd10_code, ccs_result[0], ccs_result))
        else:
            results.append((icd10_code, None, None))
    
    return results

def print_batch_results(results):
    """打印批量查找结果"""
    print("\n" + "="*80)
    print(f"{'ICD-10-CM':<20} {'CCS (主要)':<15} {'全部 CCS codes'}")
    print("="*80)
    for icd10, ccs_primary, ccs_all in results:
        if ccs_primary:
            print(f"{icd10:<20} {ccs_primary:<15} {ccs_all}")
        else:
            print(f"{icd10:<20} {'未找到':<15} {ccs_all}")
    print("="*80 + "\n")

def interactive_mode():
    """交互模式"""
    print("\n进入交互模式，输入 ICD-10-CM 代码查找对应的 CCS code")
    print("输入 'q' 或 'quit' 退出")
    print("输入 'batch' 进入批量模式\n")
    
    while True:
        try:
            user_input = input("请输入 ICD-10-CM 代码: ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("退出程序")
                break
            elif user_input.lower() == 'batch':
                # 批量模式
                print("\n批量模式：输入多个 ICD-10-CM 代码（用空格或逗号分隔）")
                batch_input = input("批量输入: ").strip()
                if not batch_input:
                    continue
                
                # 处理空格或逗号分隔
                icd10_codes = batch_input.replace(',', ' ').split()
                results = find_ccs_codes_batch(icd10_codes)
                print_batch_results(results)
            elif user_input:
                # 单个查找
                print()
                find_ccs_code(user_input)
                print()
        except KeyboardInterrupt:
            print("\n\n退出程序")
            break
        except Exception as e:
            print(f"错误: {e}\n")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 命令行模式
        if len(sys.argv) == 2:
            # 单个 ICD 代码
            y_codes = find_ccs_code(sys.argv[1])
        else:
            # 批量 ICD 代码
            icd10_codes = sys.argv[1:]
            results = find_ccs_codes_batch(icd10_codes)
            print_batch_results(results)
    else:
        # 交互模式
        interactive_mode()

if __name__ == "__main__":
    main()

