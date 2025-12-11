"""
ShallowConvNet模型实现 - PyTorch版本
基于Schirrmeister et al. (2017)的Deep learning with convolutional neural networks for EEG decoding and visualization

参考文献:
Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., 
Eggensperger, K., Tangermann, M., ... & Ball, T. (2017). 
Deep learning with convolutional neural networks for EEG decoding and visualization. 
Human brain mapping, 38(11), 5391-5420.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    """
    ShallowConvNet模型 - 适配CHB-MIT癫痫检测数据格式
    
    特点:
    - 浅层网络结构，参数较少
    - 大卷积核捕获长时程依赖
    - 适合小数据集训练
    
    输入格式: [B, C, T] (批次, 通道, 时间点) - 与现有EEGNet兼容
    输出格式: [B, nb_classes] - 与现有训练脚本兼容
    """
    
    def __init__(self, nb_classes=2, Chans=23, Samples=512, dropoutRate=0.5):
        """
        参数:
        - nb_classes: 分类数量 (默认2: 正常/癫痫)
        - Chans: EEG通道数 (CHB-MIT为23)
        - Samples: 时间点数 (如2秒*256Hz=512)
        - dropoutRate: Dropout比例
        """
        super(ShallowConvNet, self).__init__()
        
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        
        # Block 1: 时间卷积 - 使用大卷积核捕获长时程特征
        self.temporal_conv = nn.Conv2d(1, 40, (1, 25), bias=False)
        
        # Block 2: 空间卷积 - 提取通道间关系
        self.spatial_conv = nn.Conv2d(40, 40, (Chans, 1), bias=False)
        
        # 批归一化
        self.batch_norm = nn.BatchNorm2d(40)
        
        # 激活函数 - 使用平方激活函数 (论文中的特色)
        # 这里用ELU代替，效果相近且更稳定
        self.activation = nn.ELU()
        
        # 平均池化 - 大池化窗口
        self.avg_pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        
        # Dropout
        self.dropout = nn.Dropout(dropoutRate)
        
        # 计算全连接层输入维度
        self._calculate_fc_input_size()
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, nb_classes)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _calculate_fc_input_size(self):
        """计算全连接层输入维度"""
        # 创建一个虚拟输入来计算维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.Chans, self.Samples)
            
            # 模拟前向传播
            x = self.temporal_conv(dummy_input)
            x = self.spatial_conv(x)
            x = self.batch_norm(x)
            x = self.activation(x)
            x = self.avg_pool(x)
            x = self.dropout(x)
            
            self.fc_input_size = x.numel()  # 总元素数量
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        输入: x [B, C, T] - (批次大小, 通道数, 时间点)
        输出: [B, nb_classes] - 分类logits
        """
        # 输入形状: [B, C, T] -> [B, 1, C, T] (添加一个维度用于2D卷积)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, C, T]
        
        # 时间卷积 - 大卷积核捕获长时程特征
        x = self.temporal_conv(x)  # [B, 40, C, T']
        
        # 空间卷积 - 提取通道间关系
        x = self.spatial_conv(x)  # [B, 40, 1, T']
        
        # 批归一化
        x = self.batch_norm(x)  # [B, 40, 1, T']
        
        # 激活函数
        x = self.activation(x)  # [B, 40, 1, T']
        
        # 平均池化 - 大幅降维
        x = self.avg_pool(x)  # [B, 40, 1, T'']
        
        # Dropout
        x = self.dropout(x)  # [B, 40, 1, T'']
        
        # 分类
        output = self.classifier(x)  # [B, nb_classes]
        
        return output
    
    def get_feature_maps(self, x):
        """
        获取中间特征图用于可视化
        
        返回: 包含各层特征图的字典
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        features = {}
        
        # 时间卷积
        x = self.temporal_conv(x)
        features['temporal_conv'] = x.clone()
        
        # 空间卷积
        x = self.spatial_conv(x)
        features['spatial_conv'] = x.clone()
        
        # 批归一化 + 激活
        x = self.batch_norm(x)
        x = self.activation(x)
        features['after_activation'] = x.clone()
        
        # 池化
        x = self.avg_pool(x)
        features['after_pooling'] = x.clone()
        
        return features


class ShallowConvNetSquare(nn.Module):
    """
    ShallowConvNet模型 - 使用平方激活函数的原始版本
    
    更接近原论文的实现，使用平方激活函数和对数激活
    """
    
    def __init__(self, nb_classes=2, Chans=23, Samples=512, dropoutRate=0.5):
        """
        参数:
        - nb_classes: 分类数量 (默认2: 正常/癫痫)
        - Chans: EEG通道数 (CHB-MIT为23)
        - Samples: 时间点数 (如2秒*256Hz=512)
        - dropoutRate: Dropout比例
        """
        super(ShallowConvNetSquare, self).__init__()
        
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        
        # Block 1: 时间卷积
        self.temporal_conv = nn.Conv2d(1, 40, (1, 25), bias=False)
        
        # Block 2: 空间卷积
        self.spatial_conv = nn.Conv2d(40, 40, (Chans, 1), bias=False)
        
        # 批归一化
        self.batch_norm = nn.BatchNorm2d(40)
        
        # 平均池化
        self.avg_pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        
        # Dropout
        self.dropout = nn.Dropout(dropoutRate)
        
        # 计算全连接层输入维度
        self._calculate_fc_input_size()
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, nb_classes)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _calculate_fc_input_size(self):
        """计算全连接层输入维度"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.Chans, self.Samples)
            
            x = self.temporal_conv(dummy_input)
            x = self.spatial_conv(x)
            x = self.batch_norm(x)
            x = self._square_activation(x)
            x = self.avg_pool(x)
            x = self._log_activation(x)
            x = self.dropout(x)
            
            self.fc_input_size = x.numel()
    
    def _square_activation(self, x):
        """平方激活函数"""
        return torch.square(x)
    
    def _log_activation(self, x):
        """对数激活函数"""
        return torch.log(torch.clamp(x, min=1e-6))
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # 时间卷积
        x = self.temporal_conv(x)
        
        # 空间卷积
        x = self.spatial_conv(x)
        
        # 批归一化
        x = self.batch_norm(x)
        
        # 平方激活
        x = self._square_activation(x)
        
        # 平均池化
        x = self.avg_pool(x)
        
        # 对数激活
        x = self._log_activation(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 分类
        output = self.classifier(x)
        
        return output
    
    def get_feature_maps(self, x):
        """获取中间特征图用于可视化"""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        features = {}
        
        x = self.temporal_conv(x)
        features['temporal_conv'] = x.clone()
        
        x = self.spatial_conv(x)
        features['spatial_conv'] = x.clone()
        
        x = self.batch_norm(x)
        x = self._square_activation(x)
        features['after_square'] = x.clone()
        
        x = self.avg_pool(x)
        x = self._log_activation(x)
        features['after_log'] = x.clone()
        
        return features


# 测试函数
def test_shallowconvnet():
    """测试ShallowConvNet模型"""
    print("测试ShallowConvNet模型...")
    
    # 测试ELU版本
    print("\n=== 测试ELU版本 ===")
    model_elu = ShallowConvNet(nb_classes=2, Chans=23, Samples=512, dropoutRate=0.5)
    
    batch_size = 4
    test_input = torch.randn(batch_size, 23, 512)
    
    with torch.no_grad():
        output_elu = model_elu(test_input)
        features_elu = model_elu.get_feature_maps(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output_elu.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model_elu.parameters()):,}")
    
    print("ELU版本特征图形状:")
    for name, feature in features_elu.items():
        print(f"  {name}: {feature.shape}")
    
    # 测试平方激活版本
    print("\n=== 测试平方激活版本 ===")
    model_square = ShallowConvNetSquare(nb_classes=2, Chans=23, Samples=512, dropoutRate=0.5)
    
    with torch.no_grad():
        output_square = model_square(test_input)
        features_square = model_square.get_feature_maps(test_input)
    
    print(f"平方版本输出形状: {output_square.shape}")
    print(f"平方版本参数数量: {sum(p.numel() for p in model_square.parameters()):,}")
    
    print("平方版本特征图形状:")
    for name, feature in features_square.items():
        print(f"  {name}: {feature.shape}")
    
    print("\nShallowConvNet模型测试完成!")
    return True


if __name__ == "__main__":
    test_shallowconvnet()