#!/usr/bin/env python3
"""
CHB-MIT数据集预处理脚本 - MPS兼容版本
运行此脚本将生成训练所需的X_train.npy, Y_train.npy, X_test.npy, Y_test.npy文件

MPS兼容性说明:
- 所有浮点数据将保存为float32格式（MPS不支持float64）
- 标签数据保存为int64格式以确保兼容性
- 预处理后的数据可直接用于MPS设备训练
"""

import os
import numpy as np
from pathlib import Path
from processData import process_all_files
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("警告: scikit-learn未安装，使用简单的随机划分")
    # 简单的随机划分实现
    def train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
        np.random.seed(random_state)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        indices = np.random.permutation(n_samples)
        
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def prepare_chbmit_data(data_dir="./database/physionet.org/files/chbmit/1.0.0/", 
                       test_size=0.2, random_state=42):
    """
    准备CHB-MIT数据集用于训练
    
    Args:
        data_dir: CHB-MIT数据集根目录
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        成功返回True，失败返回False
    """
    print("=== CHB-MIT 数据集预处理 ===")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        print("请确保CHB-MIT数据集已下载并放置在正确位置")
        return False
    
    # 创建输出目录
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    
    all_windows = []
    all_labels = []
    
    # 处理每个病人的数据
    patient_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir() and d.name.startswith('chb')]
    patient_dirs.sort()
    
    print(f"找到 {len(patient_dirs)} 个病人目录")
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        summary_file = patient_dir / f"{patient_id}-summary.txt"
        
        if not summary_file.exists():
            print(f"警告: {patient_id} 的summary文件不存在，跳过")
            continue
            
        print(f"\n处理病人 {patient_id}...")
        
        try:
            # 处理该病人的所有EDF文件
            windows, labels = process_all_files(
                str(patient_dir), 
                str(summary_file),
                window_size=2.0,  # 2秒窗口
                l_freq=0.5,       # 低频滤波 0.5Hz
                h_freq=40.0       # 高频滤波 40Hz
            )
            
            all_windows.extend(windows)
            all_labels.extend(labels)
            
        except Exception as e:
            print(f"处理病人 {patient_id} 时出错: {e}")
            continue
    
    if not all_windows:
        print("错误: 没有成功处理任何数据")
        return False
    
    # 转换为numpy数组，确保MPS兼容性
    # 使用float32而不是默认的float64，以支持MPS设备
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)  # 标签使用int64
    
    print(f"\n总样本数: {len(X)}")
    print(f"数据形状: {X.shape}")
    print(f"发作期样本: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"非发作期样本: {len(y) - sum(y)} ({(len(y) - sum(y))/len(y)*100:.1f}%)")
    
    # 划分训练集和测试集
    print("\n划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # MPS兼容性检查和数据类型验证
    print("\n验证数据类型兼容性...")
    print(f"X_train数据类型: {X_train.dtype} (MPS兼容: {'✓' if X_train.dtype == np.float32 else '✗'})")
    print(f"y_train数据类型: {y_train.dtype}")
    print(f"X_test数据类型: {X_test.dtype} (MPS兼容: {'✓' if X_test.dtype == np.float32 else '✗'})")
    print(f"y_test数据类型: {y_test.dtype}")
    
    # 确保数据类型正确
    if X_train.dtype != np.float32:
        print("警告: 转换X_train到float32以确保MPS兼容性")
        X_train = X_train.astype(np.float32)
    if X_test.dtype != np.float32:
        print("警告: 转换X_test到float32以确保MPS兼容性")
        X_test = X_test.astype(np.float32)
    
    # 保存数据
    print("\n保存数据...")
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "Y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "Y_test.npy"), y_test)
    
    print(f"数据已保存到 {output_dir} 目录:")
    print("- X_train.npy (float32, MPS兼容)")
    print("- Y_train.npy")
    print("- X_test.npy (float32, MPS兼容)")
    print("- Y_test.npy")
    
    return True

def main():
    """主函数"""
    print("开始预处理CHB-MIT数据集...")
    
    # 运行数据预处理
    success = prepare_chbmit_data()
    
    if success:
        print("\n✅ 数据预处理完成！")
        print("现在可以运行 train.py 开始训练模型了")
    else:
        print("\n❌ 数据预处理失败")
        print("请检查数据路径和文件完整性")

if __name__ == "__main__":
    main()