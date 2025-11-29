# ELM分类器模块
"""
极端学习机（ELM）分类器
轻量级、训练极快的单隐层神经网络
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional

class ELMClassifier:
    """
    ELM分类器：基于Moore-Penrose伪逆的快速训练
    """
    
    def __init__(self, n_hidden: int = 100, activation: str = 'sigmoid', 
                 random_state: int = 42):
        """
        初始化ELM分类器
        
        Args:
            n_hidden: 隐层神经元数量
            activation: 激活函数 ['sigmoid', 'relu', 'tanh']
            random_state: 随机种子
        """
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state
        
        # 模型参数（训练后赋值）
        self.input_weights = None
        self.bias = None
        self.output_weights = None
        self.label_encoder = LabelEncoder()
        
        # 训练状态
        self.is_fitted = False
        
    def _activation_function(self, X: np.ndarray) -> np.ndarray:
        """应用激活函数"""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(X, -500, 500)))
        elif self.activation == 'relu':
            return np.maximum(0, X)
        elif self.activation == 'tanh':
            return np.tanh(X)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        训练ELM分类器（核心函数）
        
        Args:
            X_train: 训练特征 (n_samples, n_features)
            y_train: 训练标签 (n_samples,)
        """
        n_samples, n_features = X_train.shape
        
        # 1. 标签编码
        y_encoded = self.label_encoder.fit_transform(y_train)
        n_classes = len(self.label_encoder.classes_)
        
        # 2. 随机初始化输入权重和偏置
        np.random.seed(self.random_state)
        self.input_weights = np.random.uniform(-1, 1, (n_features, self.n_hidden))
        self.bias = np.random.uniform(-1, 1, (1, self.n_hidden))
        
        # 3. 计算隐层输出矩阵 H
        H = np.dot(X_train, self.input_weights) + self.bias
        H_activated = self._activation_function(H)
        
        # 4. 添加偏置项到隐层输出
        H_activated = np.hstack([H_activated, np.ones((n_samples, 1))])
        
        # 5. 标签转换为one-hot编码
        T = np.eye(n_classes)[y_encoded]
        
        # 6. 计算输出权重（Moore-Penrose伪逆）
        # β = H⁺ * T
        self.output_weights = np.linalg.pinv(H_activated) @ T
        
        self.is_fitted = True
        print(f"ELM训练完成: 隐层节点={self.n_hidden}, 激活函数={self.activation}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X_test: 测试特征 (n_samples, n_features)
            
        Returns:
            predictions: 预测标签 (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        # 1. 计算隐层输出
        H = np.dot(X_test, self.input_weights) + self.bias
        H_activated = self._activation_function(H)
        
        # 2. 添加偏置项
        H_activated = np.hstack([H_activated, np.ones((X_test.shape[0], 1))])
        
        # 3. 计算输出
        output = H_activated @ self.output_weights
        
        # 4. 取最大值索引作为预测类别
        pred_encoded = np.argmax(output, axis=1)
        
        # 5. 转换回原始标签
        return self.label_encoder.inverse_transform(pred_encoded)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        预测概率（用于评估）
        
        Args:
            X_test: 测试特征 (n_samples, n_features)
            
        Returns:
            probabilities: 类别概率 (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        H = np.dot(X_test, self.input_weights) + self.bias
        H_activated = self._activation_function(H)
        H_activated = np.hstack([H_activated, np.ones((X_test.shape[0], 1))])
        
        output = H_activated @ self.output_weights
        
        # Softmax转换为概率
        exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
        return exp_output / np.sum(exp_output, axis=1, keepdims=True)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_test: 真实标签
            
        Returns:
            metrics: 评估指标字典
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

# 使用示例
if __name__ == "__main__":
    # 示例数据
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 4, 100)
    
    X_test = np.random.randn(50, 10)
    y_test = np.random.randint(0, 4, 50)
    
    # 训练
    elm = ELMClassifier(n_hidden=50, activation='sigmoid')
    elm.fit(X_train, y_train)
    
    # 评估
    metrics = elm.evaluate(X_test, y_test)
    print(f"测试集性能: {metrics}")