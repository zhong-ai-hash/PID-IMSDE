# PID优化器模块
"""
PID参数优化器模块
用PID控制理论优化IMSDE的尺度因子和嵌入维度参数
"""

import numpy as np
from typing import Tuple, Callable
from sklearn.metrics import silhouette_score
from collections import deque

class PIDOptimizer:
    """
    PID参数优化器
    优化目标：最大化特征空间的类间可分性
    """
    
    def __init__(self, Kp: float = 0.1, Ki: float = 0.01, Kd: float = 0.05, 
                 scale_range: Tuple[int, int] = (5, 30)):
        """
        初始化PID优化器
        
        Args:
            Kp: 比例增益
            Ki: 积分增益
            Kd: 微分增益
            scale_range: 尺度因子搜索范围 (min, max)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.scale_range = scale_range
        
        # PID状态变量
        self.integral_error = 0
        self.prev_error = 0
        self.error_window = deque(maxlen=10)  # 用于平滑误差
        
        # 优化历史
        self.optimization_history = []
        
    def optimize(self, signal: np.ndarray, target_separability: float, 
                 imsde_func: Callable, max_iter: int = 100, 
                 convergence_thresh: float = 0.01) -> Tuple[float, int, dict]:
        """
        主优化函数：用PID控制优化IMSDE的尺度因子
        
        Args:
            signal: 输入振动信号 (N,)
            target_separability: 目标可分性（如0.95）
            imsde_func: IMSDE特征提取函数
            max_iter: 最大迭代次数
            convergence_thresh: 收敛阈值
            
        Returns:
            optimal_scale: 最优尺度因子
            optimal_m: 最优嵌入维度
            history: 优化历史记录
        """
        # 初始化参数
        current_scale = np.mean(self.scale_range)
        current_m = 3  # 嵌入维度固定为3
        
        # 重置PID状态
        self.integral_error = 0
        self.prev_error = 0
        
        for iteration in range(max_iter):
            # 1. 计算当前参数下的特征可分性
            features = imsde_func(signal, scale=current_scale, m=current_m)
            current_separability = self._calculate_separability(features)
            
            # 2. 计算误差（目标-当前）
            error = target_separability - current_separability
            
            # 3. 误差平滑处理
            self.error_window.append(error)
            smooth_error = np.mean(self.error_window)
            
            # 4. PID控制律
            self.integral_error += smooth_error
            derivative = smooth_error - self.prev_error
            
            delta_scale = (self.Kp * smooth_error + 
                          self.Ki * self.integral_error + 
                          self.Kd * derivative)
            
            # 5. 更新参数
            current_scale += delta_scale
            current_scale = np.clip(current_scale, 
                                   self.scale_range[0], 
                                   self.scale_range[1])
            
            # 6. 记录历史
            self.optimization_history.append({
                'iteration': iteration,
                'scale': current_scale,
                'separability': current_separability,
                'error': error
            })
            
            # 7. 收敛判断
            if abs(error) < convergence_thresh:
                print(f"PID优化收敛于第{iteration}次迭代")
                break
                
            self.prev_error = smooth_error
        
        return current_scale, current_m, self.optimization_history
    
    def _calculate_separability(self, features: np.ndarray) -> float:
        """
        计算特征可分性（内部函数）
        使用轮廓系数（Silhouette Score）作为可分性指标
        
        Args:
            features: 特征矩阵 (nsamples, nfeatures)
            
        Returns:
            separability_score: 可分性分数 [0, 1]
        """
        # 如果特征是时间序列，需要分段并打标签
        # 这里简化处理：假设每100个点是一个样本
        n_samples = len(features) // 100
        segmented_features = []
        labels = []
        
        for i in range(n_samples):
            start_idx = i * 100
            end_idx = start_idx + 100
            
            # 简单的标签分配（实际需要基于故障类型）
            label = i % 4  # 假设4类状态
            
            segmented_features.append(features[start_idx:end_idx])
            labels.append(label)
        
        segmented_features = np.array(segmented_features)
        labels = np.array(labels)
        
        # 计算轮廓系数
        if len(np.unique(labels)) > 1:
            score = silhouette_score(segmented_features, labels)
            # 归一化到[0, 1]
            return (score + 1) / 2
        else:
            return 0.0
    
    def get_optimization_curve(self) -> dict:
        """返回优化历史曲线"""
        return self.optimization_history

# 使用示例
if __name__ == "__main__":
    # 创建PID优化器实例
    pid_opt = PIDOptimizer(Kp=0.15, Ki=0.02, Kd=0.08)
    
    # 示例振动信号
    t = np.linspace(0, 1, 12000)
    vibration_signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(12000)
    
    # 优化目标：可分性达到0.9
    optimal_scale, optimal_m, history = pid_opt.optimize(
        signal=vibration_signal,
        target_separability=0.9,
        imsde_func=None,  # 实际使用时传入IMSDE函数
        max_iter=50
    )
    
    print(f"最优尺度因子: {optimal_scale:.2f}")
    print(f"最优嵌入维度: {optimal_m}")
    print(f"优化历史记录数: {len(history)}")