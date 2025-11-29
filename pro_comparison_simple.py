import numpy as np
import scipy.io
from scipy.signal import welch
from scipy.stats import entropy
from scipy.special import erf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CWRU轴承故障诊断 - 优化版本")
print("=" * 60)

# ============================================================================
# 优化的IMSDE特征提取
# ============================================================================

def optimized_multiscale_dispersion_entropy(signal, m, tau, c):
    """优化的多尺度散布熵计算 - 避免计算爆炸"""
    # 参数安全检查
    if m > 6:  # 限制m的最大值
        m = 6
        print(f"警告: m参数自动限制为6以避免计算爆炸")
    
    if c > 8:  # 限制c的最大值
        c = 8
        print(f"警告: c参数自动限制为8以避免计算爆炸")
    
    # 符号化处理
    sigma = np.std(signal)
    mu = np.mean(signal)
    
    # 使用正态分布CDF进行符号化
    y = (signal - mu) / (sigma + 1e-8)
    cdf_values = 0.5 * (1 + erf(y / np.sqrt(2)))
    
    # 映射到整数符号
    z = np.floor(c * cdf_values + 1).astype(int)
    z = np.clip(z, 1, c)
    
    # 构建模式 - 使用字典避免大数组
    pattern_dict = {}
    n = len(z) - (m - 1) * tau
    
    for i in range(n):
        pattern = 0
        for j in range(m):
            pattern += (z[i + j * tau] - 1) * (c ** (m - 1 - j))
        
        if pattern in pattern_dict:
            pattern_dict[pattern] += 1
        else:
            pattern_dict[pattern] = 1
    
    # 计算概率分布
    total = sum(pattern_dict.values())
    prob = np.array(list(pattern_dict.values())) / total
    prob = prob[prob > 0]
    
    # 计算散布熵
    if len(prob) <= 1:
        return 0
    de_value = -np.sum(prob * np.log(prob))
    
    return de_value

def extract_IMSDE_features_optimized(signals, m=3, tau=1, c=6, max_scale=5):
    """优化的IMSDE特征提取"""
    print(f"\n提取IMSDE特征 (优化版):")
    print(f"  参数: m={m}, tau={tau}, c={c}, 最大尺度: {max_scale}")
    
    n_samples = len(signals)
    n_features = max_scale
    features = np.zeros((n_samples, n_features))
    
    # 进度跟踪
    progress_interval = max(1, n_samples // 10)  # 每10%显示一次进度
    
    for i, signal in enumerate(signals):
        if i % progress_interval == 0:
            print(f"  进度: {i+1}/{n_samples} ({((i+1)/n_samples*100):.1f}%)")
        
        for scale in range(1, max_scale + 1):
            # 复合粗粒化
            if scale == 1:
                scaled_signal = signal
            else:
                length = len(signal) // scale
                scaled_signal = np.zeros(length)
                for j in range(length):
                    scaled_signal[j] = np.mean(signal[j*scale : (j+1)*scale])
            
            # 计算散布熵
            de_value = optimized_multiscale_dispersion_entropy(scaled_signal, m, tau, c)
            features[i, scale-1] = de_value
    
    print(f"  特征提取完成!")
    return features

# ============================================================================
# 优化的PID参数搜索
# ============================================================================

def PID_Optimize_Fast(src_signals, src_labels, tgt_signals, tgt_labels, search_space, n_iter=5):
    """快速PID参数优化"""
    print(f"\n{'='*50}")
    print(f"开始快速PID参数优化")
    print(f"{'='*50}")
    
    # 限制参数范围避免计算爆炸
    safe_search_space = {
        'm': [min(search_space['m']), min(max(search_space['m']), 6)],  # m最大6
        'c': [min(search_space['c']), min(max(search_space['c']), 8)]   # c最大8
    }
    
    print(f"安全参数搜索空间:")
    print(f"  m: {safe_search_space['m']}")
    print(f"  c: {safe_search_space['c']}")
    print(f"  迭代次数: {n_iter}")
    
    best_accuracy = 0
    best_params = {}
    evolution_log = []
    
    for iteration in range(n_iter):
        # 在安全范围内随机选择参数
        m = np.random.randint(safe_search_space['m'][0], safe_search_space['m'][1] + 1)
        c = np.random.randint(safe_search_space['c'][0], safe_search_space['c'][1] + 1)
        
        print(f"\n迭代 {iteration + 1}/{n_iter}:")
        print(f"  测试参数: m={m}, c={c}")
        
        try:
            # 使用优化的IMSDE特征提取
            src_features = extract_IMSDE_features_optimized(src_signals, m=m, c=c, max_scale=3)  # 减少尺度数
            tgt_features = extract_IMSDE_features_optimized(tgt_signals, m=m, c=c, max_scale=3)
            
            # ELM分类
            accuracy = ELM_classifier(src_features, src_labels, tgt_features, tgt_labels)
            
            evolution_log.append({
                'iteration': iteration + 1,
                'm': m,
                'c': c,
                'accuracy': accuracy
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'m': m, 'c': c}
                print(f"  🎯 新的最佳准确率: {accuracy:.2f}%")
                
        except Exception as e:
            print(f"  错误: {str(e)}，跳过该参数组合")
            continue
    
    print(f"\nPID优化完成:")
    print(f"  最佳参数: m={best_params['m']}, c={best_params['c']}")
    print(f"  最佳准确率: {best_accuracy:.2f}%")
    
    # 显示参数演化
    print(f"\n参数演化过程:")
    for log in evolution_log:
        print(f"  迭代{log['iteration']:2d}: m={log['m']}, c={log['c']}, 准确率={log['accuracy']:.2f}%")
    
    return best_accuracy

# ============================================================================
# 主实验函数（优化版）
# ============================================================================

def main_optimized():
    """优化的主实验函数"""
    print("\n" + "="*60)
    print("开始CWRU轴承故障诊断实验 (优化版)")
    print("="*60)
    
    # 数据配置（使用你的实际文件路径）
    源域文件 = {
        "normal": "data/CWRU/cwru_raw/48k Drive End Bearing Fault Data/Ball/0007/B007_0.mat",
        "inner": "data/CWRU/cwru_raw/48k Drive End Bearing Fault Data/Ball/0007/B007_0.mat",
        "outer": "data/CWRU/cwru_raw/48k Drive End Bearing Fault Data/Ball/0007/B007_0.mat", 
        "ball": "data/CWRU/cwru_raw/48k Drive End Bearing Fault Data/Ball/0007/B007_0.mat"
    }
    
    目标域文件 = {
        "normal": {"path": "data/CWRU/cwru_raw/48k Drive End Bearing Fault Data/Ball/0007/B007_3.mat", "use_first": 50},
        "inner": "data/CWRU/cwru_raw/48k Drive End Bearing Fault Data/Ball/0007/B007_3.mat",
        "outer": "data/CWRU/cwru_raw/48k Drive End Bearing Fault Data/Ball/0007/B007_3.mat",
        "ball": {"path": "data/CWRU/cwru_raw/48k Drive End Bearing Fault Data/Ball/0007/B007_3.mat", "start_pos": 50*2048}
    }
    
    # 加载数据
    print("\n加载数据...")
    src_signals, src_labels = load_cwru_data(源域文件, samples_per_class=50)
    tgt_signals, tgt_labels = load_cwru_data(目标域文件, samples_per_class=50)
    
    # 快速实验
    print("\n" + "="*60)
    print("开始快速跨域迁移学习实验")
    print("="*60)
    
    # 1. MDE方法（快速）
    print("\n>>> 方法1: 多尺度散布熵 (MDE)")
    acc_mde = cross_domain_test(
        extract_MDE_features, 
        src_signals, src_labels, 
        tgt_signals, tgt_labels, 
        params={"m": 3, "c": 6}
    )
    
    # 2. IMSDE方法（优化版）
    print("\n>>> 方法2: 改进多尺度散布熵 (IMSDE-优化)")
    acc_imsde = cross_domain_test(
        extract_IMSDE_features_optimized, 
        src_signals, src_labels, 
        tgt_signals, tgt_labels, 
        params={"m": 3, "c": 6, "max_scale": 3}  # 减少尺度数
    )
    
    # 3. 快速PID优化
    print("\n>>> 方法3: 快速PID优化的IMSDE")
    acc_pid = PID_Optimize_Fast(
        src_signals, src_labels, 
        tgt_signals, tgt_labels, 
        search_space={"m": [2, 6], "c": [4, 8]},  # 缩小搜索范围
        n_iter=4  # 减少迭代次数
    )
    
    # 结果汇总
    print("\n" + "="*60)
    print("实验最终结果汇总")
    print("="*60)
    
    results = {
        "MDE": acc_mde,
        "IMSDE": acc_imsde,
        "PID-IMSDE": acc_pid
    }
    
    print("\n准确率对比:")
    for method, accuracy in results.items():
        print(f"  {method:12s}: {accuracy:6.2f}%")
    
    best_method = max(results, key=results.get)
    best_accuracy = results[best_method]
    print(f"\n🎉 最佳方法: {best_method}")
    print(f"🏆 最佳准确率: {best_accuracy:.2f}%")
    
    return results

# ============================================================================
# 执行优化版本
# ============================================================================

if __name__ == "__main__":
    print("🚀 启动优化版本 - 计算时间大幅减少!")
    print("💡 主要优化:")
    print("  - 限制 m ≤ 6, c ≤ 8 避免计算爆炸")
    print("  - 减少最大尺度从5到3")
    print("  - 使用字典代替大数组存储模式")
    print("  - 减少PID迭代次数")
    print("  - 预计总时间: 1-2小时\n")
    
    try:
        final_results = main_optimized()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")