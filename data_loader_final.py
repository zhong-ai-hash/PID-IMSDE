"""
数据加载模块（最终确定版）
反向查找策略：自动扫描实际文件夹，不再猜测命名规则
"""

import scipy.io as sio
import numpy as np
import os

class DataLoader:
    def __init__(self, cwru_path: str, seu_path: str):
        self.cwru_path = cwru_path
        self.seu_path = seu_path
        self.sample_rate_cwru = 48000
        
    def load_cwru_data(self, fault_type: str, fault_size: str, 
                      load_hp: int = 0, data_num: int = 1) -> tuple:
        """
        加载CWRU轴承数据（自动适配实际文件夹命名）
        
        Args:
            fault_type: 故障类型 ['ball', 'inner', 'outer', 'normal']
            fault_size: 故障尺寸 ['007', '014', '021']
            load_hp: 负载马力（仅用于记录，48kHz数据不包含此信息）
            data_num: 数据文件编号 [1, 2, 3...]
            
        Returns:
            vibration_signal: 振动信号数组
            rpm: 转速
        """
        # ==================== 关键修复：反向查找实际文件夹 ====================
        fault_folder_map = {
            'normal': 'Normal Baseline Data',
            'inner': 'Inner Race', 
            'outer': 'Outer Race',
            'ball': 'Ball'
        }
        
        rate_folder = '48k Drive End Bearing Fault Data'
        
        # 构建到故障类型层级的路径（不包含尺寸文件夹）
        base_path = os.path.join(self.cwru_path, rate_folder, fault_folder_map[fault_type])
        
        if not os.path.exists(base_path):
            print(f"\n❌ 基础路径不存在: {base_path}")
            return None, None
        
        # 关键步骤：扫描该目录下所有子文件夹，自动匹配包含fault_size的文件夹
        if fault_type != 'normal':
            subfolders = [f for f in os.listdir(base_path) 
                         if os.path.isdir(os.path.join(base_path, f))]
            
            # 找出包含故障尺寸的文件夹（例如"0007"包含"007"）
            matching_folders = [f for f in subfolders if fault_size in f]
            
            if not matching_folders:
                print(f"\n⚠️ 在 {base_path} 下未找到包含 '{fault_size}' 的文件夹")
                print(f"   实际子文件夹: {subfolders[:5]}...")
                return None, None
            
            size_folder = matching_folders[0]  # 使用第一个匹配的文件夹
            print(f"   自动匹配文件夹: {size_folder}")
        else:
            # 正常数据无尺寸子文件夹
            size_folder = ""
        
        # 构建完整文件夹路径
        if fault_type == 'normal':
            folder_path = base_path
        else:
            folder_path = os.path.join(base_path, size_folder)
        
        # 文件命名规则
        if fault_type == 'normal':
            filename = f"normal_{data_num}.mat"  # 需根据实际调整
        else:
            prefix_map = {'ball': 'B', 'inner': 'IR', 'outer': 'OR'}
            filename = f"{prefix_map[fault_type]}{fault_size}_{data_num}.mat"
        
        filepath = os.path.join(folder_path, filename)
        
        # 再次检查文件是否存在
        if not os.path.exists(filepath):
            print(f"\n❌ 文件不存在: {filepath}")
            # 列出该目录实际文件帮助调试
            if os.path.exists(folder_path):
                actual_files = os.listdir(folder_path)
                print(f"   目录实际文件: {actual_files[:5]}...")
            return None, None
        
        # 加载数据（简化但鲁棒的核心逻辑）
        try:
            mat_data = sio.loadmat(filepath)
            
            # 自动寻找最大的变量作为振动信号
            max_var = None
            max_size = 0
            
            for key, value in mat_data.items():
                if key.startswith('__'):
                    continue
                if isinstance(value, np.ndarray) and value.size > max_size:
                    max_size = value.size
                    max_var = key
            
            if max_var is None:
                raise ValueError("未找到有效的数据变量")
                
            vibration_signal = mat_data[max_var].flatten()
            
            # 读取转速
            rpm = 1772
            if 'X123RPM' in mat_data:
                rpm = int(mat_data['X123RPM'].flatten()[0])
            
            print(f"\n✅ 加载成功: {filename}")
            print(f"   变量名: {max_var}, 信号长度: {len(vibration_signal)}")
            print(f"   转速: {rpm} RPM")
            
            return vibration_signal, rpm
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return None, None

# 测试代码（确保在paper_A目录下运行）
if __name__ == "__main__":
    import os
    
    # 获取当前脚本所在目录（paper_A）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    loader = DataLoader(
        cwru_path=os.path.join(base_dir, "data", "CWRU", "cwru_raw"),
        seu_path=os.path.join(base_dir, "data", "SEU")
    )
    
    # 测试B007_1.mat
    print("="*60)
    print("测试: 加载 B007_1.mat")
    print("="*60)
    
    signal, rpm = loader.load_cwru_data('ball', '007', 0, 1)
    
    if signal is not None:
        print(f"\n🎉 测试成功！")
        print(f"信号长度: {len(signal)}")
        print(f"转速: {rpm} RPM")
        
        # 可视化前1000点
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(signal[:1000])
        plt.title(f"B007_1.mat - 驱动端振动信号 ({rpm} RPM)")
        plt.xlabel("采样点")
        plt.ylabel("幅值")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("B007_1_signal.png", dpi=300)
        plt.show()
    else:
        print("\n❌ 测试失败")
