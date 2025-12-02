#!/usr/bin/env python3
"""
CHB-MIT癫痫预测项目使用指南

这个脚本提供了完整的使用流程，帮助您快速开始训练模型
"""

import os
import subprocess
import sys

def check_dependencies():
    """检查必要的依赖包"""
    print("=== 检查依赖包 ===")
    
    required_packages = [
        'torch', 'numpy', 'sklearn', 'mne',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    # 特殊处理scikit-learn
    try:
        import sklearn
        print(f"✅ scikit-learn (sklearn) 已安装")
    except ImportError:
        if 'sklearn' not in missing_packages:
            missing_packages.append('sklearn')
        print(f"❌ scikit-learn 未安装")
    
    if missing_packages:
        print(f"\n需要安装以下包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_data_files():
    """检查数据文件是否存在"""
    print("\n=== 检查数据文件 ===")
    
    data_files = [
        'database/physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf',
        'database/physionet.org/files/chbmit/1.0.0/chb01/chb01-summary.txt'
    ]
    
    missing_files = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 存在")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} 不存在")
    
    if missing_files:
        print(f"\n缺少以下文件: {', '.join(missing_files)}")
        print("请确保CHB-MIT数据集已正确下载并放置")
        return False
    
    return True

def run_data_preprocessing():
    """运行数据预处理"""
    print("\n=== 运行数据预处理 ===")
    
    # 检查是否已有预处理数据
    if os.path.exists('data/X_train.npy'):
        print("预处理数据已存在，跳过此步骤")
        return True
    
    print("正在运行数据预处理...")
    try:
        result = subprocess.run([sys.executable, 'prepare_data.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 数据预处理完成")
            return True
        else:
            print(f"❌ 数据预处理失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 运行数据预处理时出错: {e}")
        return False

def run_training():
    """运行模型训练"""
    print("\n=== 运行模型训练 ===")
    
    print("正在运行模型训练...")
    try:
        result = subprocess.run([sys.executable, 'train.py'], 
                              capture_output=True, text=True)
        
        # 实时输出训练过程
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ 模型训练完成")
            return True
        else:
            print(f"❌ 模型训练失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 运行模型训练时出错: {e}")
        return False

def main():
    """主函数：完整的使用流程"""
    print("🧠 CHB-MIT 癫痫预测项目使用指南")
    print("=" * 40)
    
    # 步骤1: 检查依赖
    if not check_dependencies():
        print("\n请先安装缺失的依赖包")
        return
    
    # 步骤2: 检查数据文件
    if not check_data_files():
        print("\n请确保数据文件存在")
        return
    
    # 步骤3: 数据预处理（可选）
    if not run_data_preprocessing():
        print("\n数据预处理失败，请检查错误信息")
        return
    
    # 步骤4: 运行训练
    if not run_training():
        print("\n模型训练失败，请检查错误信息")
        return
    
    print("\n🎉 所有步骤完成！")
    print("\n您也可以单独运行各个脚本:")
    print("  python prepare_data.py  # 数据预处理")
    print("  python train.py         # 模型训练")

if __name__ == "__main__":
    main()