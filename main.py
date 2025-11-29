# 主入口文件
"""
主程序入口
一键执行完整的跨域故障诊断流程
"""

import os
import sys
from datetime import datetime

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cross_domain_test_framework import CrossDomainTestFramework, config_template

def main():
    """
    主函数：执行完整实验流程
    """
    print("=" * 70)
    print("PID-IMSDE跨域故障诊断系统")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. 检查数据目录
    if not os.path.exists('./data/CWRU'):
        print("错误: 请创建数据目录 ./data/CWRU 和 ./data/SEU")
        print("并将CWRU和SEU数据集放入对应目录")
        return
    
    # 2. 初始化配置
    config = config_template.copy()
    config['pid_params'] = {'Kp': 0.15, 'Ki': 0.02, 'Kd': 0.08}
    config['elm_params'] = {'n_hidden': 100, 'activation': 'sigmoid'}
    config['use_pid_optimization'] = True  # 开启PID优化
    
    # 3. 创建测试框架
    framework = CrossDomainTestFramework(config)
    
    try:
        # 4. 运行CWRU跨域测试
        print("\n【阶段1】运行CWRU跨域测试...")
        cwru_results = framework.run_cwru_cross_domain_test()
        
        # 5. 运行SEU验证
        print("\n【阶段2】运行SEU验证测试...")
        seu_results = framework.run_seu_validation()
        
        # 6. 可视化结果
        print("\n【阶段3】生成可视化图表...")
        framework.visualize_results()
        
        # 7. 生成报告
        print("\n【阶段4】生成测试报告...")
        report = framework.generate_report()
        print(report)
        
        # 8. 保存结果
        import pickle
        with open('experiment_results.pkl', 'wb') as f:
            pickle.dump(framework.results, f)
        print("\n实验结果已保存到 experiment_results.pkl")
        
    except Exception as e:
        print(f"\n实验过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"实验完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
