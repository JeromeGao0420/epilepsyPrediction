"""
测试对比模型的兼容性和功能
验证DeepConvNet和ShallowConvNet是否能正确集成到训练系统中
"""

import torch
import numpy as np
import sys
import os

# 添加模块路径
sys.path.append('ablation_models')

from ablation_models.DeepConvNet import DeepConvNet
from ablation_models.ShallowConvNet import ShallowConvNet
from ablation_models.TCFormer import SimplifiedTCFormer
from ablation_models.BaseEEGNet import EEGNet as BaseEEGNet
from ablation_models.AttentionEEGNet import EEGNet as AttentionEEGNet
from ablation_models.AttentionBiLSTM import AttentionBiLSTM

def test_model_compatibility():
    """测试所有模型的兼容性"""
    print("=== 测试模型兼容性 ===\n")
    
    # 测试参数
    batch_size = 4
    chans = 23
    samples = 512
    num_classes = 2
    
    # 创建测试数据
    test_input = torch.randn(batch_size, chans, samples)
    print(f"测试输入形状: {test_input.shape}")
    
    # 定义所有模型
    models = {
        'BaseEEGNet': BaseEEGNet(
            nb_classes=num_classes,
            Chans=chans,
            Samples=samples,
            dropoutRate=0.15
        ),
        'AttentionEEGNet': AttentionEEGNet(
            nb_classes=num_classes,
            Chans=chans,
            Samples=samples,
            dropoutRate=0.15
        ),
        'AttentionBiLSTM': AttentionBiLSTM(
            nb_classes=num_classes,
            Chans=chans,
            Samples=samples,
            hidden_dim=128,
            num_layers=2,
            dropout=0.15,
            use_attention=True,
            attention_heads=4
        ),
        'DeepConvNet': DeepConvNet(
            nb_classes=num_classes,
            Chans=chans,
            Samples=samples,
            dropoutRate=0.5
        ),
        'ShallowConvNet': ShallowConvNet(
            nb_classes=num_classes,
            Chans=chans,
            Samples=samples,
            dropoutRate=0.5
        ),
        'TCFormer': SimplifiedTCFormer(
            nb_classes=num_classes,
            Chans=chans,
            Samples=samples,
            temp_kernels=(16, 32, 64),
            F1=16,
            D=2,
            d_model=64,
            num_heads=8,
            num_layers=4,
            tcn_channels=32,
            tcn_layers=2,
            dropout=0.3
        )
    }
    
    # 测试每个模型
    results = {}
    
    for model_name, model in models.items():
        print(f"--- 测试 {model_name} ---")
        
        try:
            # 设置为评估模式
            model.eval()
            
            # 前向传播测试
            with torch.no_grad():
                output = model(test_input)
            
            # 检查输出形状
            expected_shape = (batch_size, num_classes)
            if output.shape == expected_shape:
                print(f"✅ 输出形状正确: {output.shape}")
            else:
                print(f"❌ 输出形状错误: {output.shape}, 期望: {expected_shape}")
                results[model_name] = False
                continue
            
            # 检查输出值范围
            if torch.isnan(output).any():
                print(f"❌ 输出包含NaN值")
                results[model_name] = False
                continue
            
            if torch.isinf(output).any():
                print(f"❌ 输出包含无穷值")
                results[model_name] = False
                continue
            
            # 计算参数数量
            param_count = sum(p.numel() for p in model.parameters())
            print(f"📊 参数数量: {param_count:,}")
            
            # 测试特征图提取（如果支持）
            if hasattr(model, 'get_feature_maps'):
                try:
                    features = model.get_feature_maps(test_input)
                    print(f"🔍 特征图数量: {len(features)}")
                    for feat_name, feat_tensor in features.items():
                        print(f"   {feat_name}: {feat_tensor.shape}")
                except Exception as e:
                    print(f"⚠️  特征图提取失败: {e}")
            
            # 测试梯度计算
            model.train()
            output = model(test_input)
            loss = torch.nn.functional.cross_entropy(
                output, 
                torch.randint(0, num_classes, (batch_size,))
            )
            loss.backward()
            
            # 检查梯度
            grad_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_count += 1
            
            print(f"🔄 梯度计算正常: {grad_count} 个参数有梯度")
            
            results[model_name] = True
            print(f"✅ {model_name} 测试通过\n")
            
        except Exception as e:
            print(f"❌ {model_name} 测试失败: {e}\n")
            results[model_name] = False
    
    return results

def test_training_integration():
    """测试训练脚本集成"""
    print("=== 测试训练脚本集成 ===\n")
    
    # 模拟训练脚本的模型创建函数
    def create_model(model_name, args):
        """模拟训练脚本中的模型创建函数"""
        if model_name == 'BaseEEGNet':
            return BaseEEGNet(
                nb_classes=args['num_classes'],
                Chans=args['chans'],
                Samples=args['samples'],
                dropoutRate=args['dropout']
            )
        elif model_name == 'AttentionEEGNet':
            return AttentionEEGNet(
                nb_classes=args['num_classes'],
                Chans=args['chans'],
                Samples=args['samples'],
                dropoutRate=args['dropout']
            )
        elif model_name == 'AttentionBiLSTM':
            return AttentionBiLSTM(
                nb_classes=args['num_classes'],
                Chans=args['chans'],
                Samples=args['samples'],
                hidden_dim=args['hidden_dim'],
                num_layers=args['num_layers'],
                dropout=args['dropout'],
                use_attention=True,
                attention_heads=args['attention_heads']
            )
        elif model_name == 'DeepConvNet':
            return DeepConvNet(
                nb_classes=args['num_classes'],
                Chans=args['chans'],
                Samples=args['samples'],
                dropoutRate=args['dropout']
            )
        elif model_name == 'ShallowConvNet':
            return ShallowConvNet(
                nb_classes=args['num_classes'],
                Chans=args['chans'],
                Samples=args['samples'],
                dropoutRate=args['dropout']
            )
        else:
            raise ValueError(f"未知的模型类型: {model_name}")
    
    # 测试参数
    args = {
        'num_classes': 2,
        'chans': 23,
        'samples': 512,
        'dropout': 0.15,
        'hidden_dim': 128,
        'num_layers': 2,
        'attention_heads': 4
    }
    
    # 测试所有模型创建
    model_names = ['BaseEEGNet', 'AttentionEEGNet', 'AttentionBiLSTM', 'DeepConvNet', 'ShallowConvNet']
    
    for model_name in model_names:
        try:
            model = create_model(model_name, args)
            print(f"✅ {model_name} 创建成功")
        except Exception as e:
            print(f"❌ {model_name} 创建失败: {e}")
    
    print()

def test_visualization_integration():
    """测试可视化脚本集成"""
    print("=== 测试可视化脚本集成 ===\n")
    
    # 模拟可视化脚本的模型创建函数
    def create_model_for_viz(model_name):
        """模拟可视化脚本中的模型创建函数"""
        CHANS = 23
        SAMPLES = 512
        
        if model_name == 'BaseEEGNet':
            from ablation_models.BaseEEGNet import EEGNet
            return EEGNet(
                nb_classes=2,
                Chans=CHANS,
                Samples=SAMPLES,
                dropoutRate=0.15
            )
        elif model_name == 'AttentionEEGNet':
            from ablation_models.AttentionEEGNet import EEGNet
            return EEGNet(
                nb_classes=2,
                Chans=CHANS,
                Samples=SAMPLES,
                dropoutRate=0.15
            )
        elif model_name == 'AttentionBiLSTM':
            from ablation_models.AttentionBiLSTM import AttentionBiLSTM
            return AttentionBiLSTM(
                nb_classes=2,
                Chans=CHANS,
                Samples=SAMPLES,
                hidden_dim=128,
                num_layers=2,
                dropout=0.15,
                use_attention=True,
                attention_heads=4
            )
        elif model_name == 'DeepConvNet':
            from ablation_models.DeepConvNet import DeepConvNet
            return DeepConvNet(
                nb_classes=2,
                Chans=CHANS,
                Samples=SAMPLES,
                dropoutRate=0.5
            )
        elif model_name == 'ShallowConvNet':
            from ablation_models.ShallowConvNet import ShallowConvNet
            return ShallowConvNet(
                nb_classes=2,
                Chans=CHANS,
                Samples=SAMPLES,
                dropoutRate=0.5
            )
        else:
            raise ValueError(f"不支持的模型: {model_name}")
    
    # 测试所有模型创建
    model_names = ['BaseEEGNet', 'AttentionEEGNet', 'AttentionBiLSTM', 'DeepConvNet', 'ShallowConvNet']
    
    for model_name in model_names:
        try:
            model = create_model_for_viz(model_name)
            print(f"✅ {model_name} 可视化创建成功")
        except Exception as e:
            print(f"❌ {model_name} 可视化创建失败: {e}")
    
    print()

def main():
    """主测试函数"""
    print("🧪 开始测试对比模型集成\n")
    
    # 测试模型兼容性
    compatibility_results = test_model_compatibility()
    
    # 测试训练集成
    test_training_integration()
    
    # 测试可视化集成
    test_visualization_integration()
    
    # 总结结果
    print("=== 测试总结 ===")
    passed = sum(compatibility_results.values())
    total = len(compatibility_results)
    
    print(f"模型兼容性测试: {passed}/{total} 通过")
    
    for model_name, result in compatibility_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {model_name}: {status}")
    
    if passed == total:
        print("\n🎉 所有模型集成测试通过！")
        return True
    else:
        print(f"\n⚠️  {total - passed} 个模型测试失败，需要修复")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)