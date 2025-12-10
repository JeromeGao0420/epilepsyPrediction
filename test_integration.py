"""
整合测试脚本
验证多尺度注意力BiLSTM模型与现有项目的兼容性

测试内容:
1. 模型导入和初始化
2. 数据格式兼容性
3. 前向传播测试
4. 注意力权重提取
5. 模型参数统计
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append('.')

def test_model_imports():
    """测试模型导入"""
    print("🔍 测试模型导入...")
    
    try:
        # 测试原有模型导入
        from ablation_models.BaseEEGNet import EEGNet as BaseEEGNet
        print("✅ BaseEEGNet 导入成功")
        
        # 测试新的注意力模型导入
        from advanced_models.AttentionBiLSTM import AttentionBiLSTM
        print("✅ AttentionBiLSTM 导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 模型导入失败: {e}")
        return False

def test_model_initialization():
    """测试模型初始化"""
    print("\n🔍 测试模型初始化...")
    
    try:
        # 导入模型
        from ablation_models.BaseEEGNet import EEGNet as BaseEEGNet
        from advanced_models.AttentionBiLSTM import AttentionBiLSTM
        
        # 测试BaseEEGNet初始化
        base_model = BaseEEGNet(
            nb_classes=2,
            Chans=23,
            Samples=512,
            dropoutRate=0.5
        )
        print("✅ BaseEEGNet 初始化成功")
        
        # 测试AttentionBiLSTM初始化
        attention_model = AttentionBiLSTM(
            nb_classes=2,
            Chans=23,
            Samples=512,
            hidden_dim=128,
            num_layers=2,
            dropout=0.15,
            use_attention=True,
            attention_heads=4
        )
        print("✅ AttentionBiLSTM 初始化成功")
        
        return base_model, attention_model
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return None, None

def test_data_compatibility():
    """测试数据格式兼容性"""
    print("\n🔍 测试数据格式兼容性...")
    
    try:
        # 创建模拟数据 (与CHB-MIT格式一致)
        batch_size = 4
        channels = 23
        samples = 512
        
        # 模拟EEG数据: [B, C, T]
        X = torch.randn(batch_size, channels, samples)
        y = torch.randint(0, 2, (batch_size,))
        
        print(f"✅ 测试数据创建成功: X{X.shape}, y{y.shape}")
        
        return X, y
    except Exception as e:
        print(f"❌ 数据创建失败: {e}")
        return None, None

def test_forward_pass(models, test_data):
    """测试前向传播"""
    print("\n🔍 测试前向传播...")
    
    if models[0] is None or models[1] is None or test_data[0] is None:
        print("❌ 跳过前向传播测试 (模型或数据初始化失败)")
        return False
    
    base_model, attention_model = models
    X, y = test_data
    
    try:
        # 测试BaseEEGNet前向传播
        base_model.eval()
        with torch.no_grad():
            base_output = base_model(X)
            print(f"✅ BaseEEGNet 前向传播成功: 输入{X.shape} -> 输出{base_output.shape}")
        
        # 测试AttentionBiLSTM前向传播
        attention_model.eval()
        with torch.no_grad():
            attention_output = attention_model(X)
            print(f"✅ AttentionBiLSTM 前向传播成功: 输入{X.shape} -> 输出{attention_output.shape}")
        
        # 验证输出格式
        expected_output_shape = (X.size(0), 2)  # [batch_size, num_classes]
        
        if base_output.shape == expected_output_shape:
            print("✅ BaseEEGNet 输出格式正确")
        else:
            print(f"⚠️ BaseEEGNet 输出格式异常: 期望{expected_output_shape}, 实际{base_output.shape}")
        
        if attention_output.shape == expected_output_shape:
            print("✅ AttentionBiLSTM 输出格式正确")
        else:
            print(f"⚠️ AttentionBiLSTM 输出格式异常: 期望{expected_output_shape}, 实际{attention_output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False

def test_attention_weights(attention_model, test_data):
    """测试注意力权重提取"""
    print("\n🔍 测试注意力权重提取...")
    
    if attention_model is None or test_data[0] is None:
        print("❌ 跳过注意力权重测试 (模型或数据初始化失败)")
        return False
    
    X, _ = test_data
    
    try:
        attention_model.eval()
        with torch.no_grad():
            # 测试注意力权重提取
            if hasattr(attention_model, 'get_attention_weights'):
                attention_weights = attention_model.get_attention_weights(X[:2])  # 取2个样本
                
                if attention_weights:
                    print("✅ 注意力权重提取成功:")
                    for key, value in attention_weights.items():
                        print(f"   - {key}: {value.shape}")
                else:
                    print("⚠️ 注意力权重为空")
                
                return True
            else:
                print("⚠️ 模型不支持注意力权重提取")
                return False
    except Exception as e:
        print(f"❌ 注意力权重提取失败: {e}")
        return False

def test_model_statistics(models):
    """测试模型参数统计"""
    print("\n🔍 测试模型参数统计...")
    
    if models[0] is None or models[1] is None:
        print("❌ 跳过模型统计测试 (模型初始化失败)")
        return False
    
    base_model, attention_model = models
    
    try:
        # BaseEEGNet参数统计
        base_params = sum(p.numel() for p in base_model.parameters())
        base_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        
        print(f"📊 BaseEEGNet 统计:")
        print(f"   - 总参数数: {base_params:,}")
        print(f"   - 可训练参数: {base_trainable:,}")
        print(f"   - 模型大小: {base_params * 4 / (1024 * 1024):.2f} MB")
        
        # AttentionBiLSTM参数统计
        if hasattr(attention_model, 'get_model_size'):
            attention_stats = attention_model.get_model_size()
            print(f"📊 AttentionBiLSTM 统计:")
            print(f"   - 总参数数: {attention_stats['total_params']:,}")
            print(f"   - 可训练参数: {attention_stats['trainable_params']:,}")
            print(f"   - 模型大小: {attention_stats['size_mb']:.2f} MB")
        else:
            attention_params = sum(p.numel() for p in attention_model.parameters())
            attention_trainable = sum(p.numel() for p in attention_model.parameters() if p.requires_grad)
            print(f"📊 AttentionBiLSTM 统计:")
            print(f"   - 总参数数: {attention_params:,}")
            print(f"   - 可训练参数: {attention_trainable:,}")
            print(f"   - 模型大小: {attention_params * 4 / (1024 * 1024):.2f} MB")
        
        return True
    except Exception as e:
        print(f"❌ 模型统计失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    
    try:
        # 检查数据文件是否存在
        data_files = [
            'data/X_train_balanced.npy',
            'data/y_train_balanced.npy',
            'data/X_test_balanced.npy',
            'data/y_test_balanced.npy'
        ]
        
        missing_files = []
        for file_path in data_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print("⚠️ 以下数据文件不存在:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            print("💡 请先运行 prepare_data_balanced_memory.py 生成数据")
            return False
        else:
            # 尝试加载数据
            X_train = np.load('data/X_train_balanced.npy')
            y_train = np.load('data/y_train_balanced.npy')
            X_test = np.load('data/X_test_balanced.npy')
            y_test = np.load('data/y_test_balanced.npy')
            
            print("✅ 数据文件加载成功:")
            print(f"   - 训练集: {X_train.shape}, 标签: {y_train.shape}")
            print(f"   - 测试集: {X_test.shape}, 标签: {y_test.shape}")
            print(f"   - 训练集标签分布: {np.bincount(y_train)}")
            print(f"   - 测试集标签分布: {np.bincount(y_test)}")
            
            return True
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def test_training_script():
    """测试训练脚本"""
    print("\n🔍 测试训练脚本兼容性...")
    
    try:
        # 检查训练脚本是否存在
        if not Path('train_enhanced.py').exists():
            print("❌ train_enhanced.py 不存在")
            return False
        
        # 检查配置文件是否存在
        if not Path('config_enhanced.yaml').exists():
            print("❌ config_enhanced.yaml 不存在")
            return False
        
        print("✅ 训练脚本和配置文件存在")
        print("💡 可以使用以下命令开始训练:")
        print("   python train_enhanced.py --model AttentionBiLSTM")
        print("   python train_enhanced.py --model BaseEEGNet")
        print("   python train_enhanced.py --model AttentionBiLSTM --save_attention")
        
        return True
    except Exception as e:
        print(f"❌ 训练脚本检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始整合测试...")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        ("模型导入", test_model_imports),
        ("数据加载", test_data_loading),
        ("训练脚本", test_training_script),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 如果基础测试通过，进行模型测试
    if results[0][1]:  # 模型导入成功
        print("\n" + "=" * 50)
        print("🔬 进行详细模型测试...")
        
        # 初始化模型
        models = test_model_initialization()
        
        # 创建测试数据
        test_data = test_data_compatibility()
        
        # 前向传播测试
        forward_result = test_forward_pass(models, test_data)
        results.append(("前向传播", forward_result))
        
        # 注意力权重测试
        attention_result = test_attention_weights(models[1], test_data)
        results.append(("注意力权重", attention_result))
        
        # 模型统计测试
        stats_result = test_model_statistics(models)
        results.append(("模型统计", stats_result))
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("📋 测试结果汇总:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 测试通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！项目整合成功！")
        print("\n💡 下一步操作:")
        print("1. 安装依赖: pip install -r requirements_enhanced.txt")
        print("2. 准备数据: python prepare_data_balanced_memory.py")
        print("3. 开始训练: python train_enhanced.py --model AttentionBiLSTM")
    else:
        print("⚠️ 部分测试失败，请检查相关问题")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)