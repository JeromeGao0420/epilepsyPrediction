"""
带注意力机制的EEGNet模型
在基础EEGNet基础上添加注意力机制

特性:
- 基于EEGNet的卷积结构
- 添加通道注意力机制
- 增强特征提取能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseEEGNet import EEGNet as BaseEEGNet


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out) * x


class EEGNet(nn.Module):
    """
    带注意力机制的EEGNet模型
    
    参数:
    - nb_classes: 分类数量
    - Chans: EEG通道数
    - Samples: 时间点数
    - dropoutRate: Dropout比例
    - kernLength: 第一层卷积核长度
    - F1: 第一层滤波器数量
    - D: 深度乘数
    - F2: 第二层滤波器数量
    - norm_rate: 约束参数
    - dropoutType: Dropout类型
    - attention_dim: 注意力机制维度
    """
    
    def __init__(self, nb_classes, Chans=22, Samples=512, 
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, 
                 norm_rate=0.25, dropoutType='Dropout', attention_dim=64):
        super(EEGNet, self).__init__()
        
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate
        self.attention_dim = attention_dim
        
        # Block 1: 时间卷积
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # 添加注意力机制到Block1后
        self.attention1 = ChannelAttention(F1, reduction=4)
        
        # Block 2: 深度卷积
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        # 添加注意力机制到Block2后
        self.attention2 = ChannelAttention(F1 * D, reduction=4)
        
        # Block 3: 可分离卷积
        self.block3 = nn.Sequential(
            # 深度卷积
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            # 点卷积
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        # 添加注意力机制到Block3后
        self.attention3 = ChannelAttention(F2, reduction=4)
        
        # 计算全连接层输入维度
        self.feature_dim = F2 * (Samples // 32)
        
        # 增强的分类器，包含注意力权重
        self.feature_attention = nn.Sequential(
            nn.Linear(self.feature_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(dropoutRate),
            nn.Linear(attention_dim, self.feature_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, nb_classes)
        )
        
        # 权重初始化
        self._init_weights()
    
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
        # 输入形状: [B, C, T] -> [B, 1, C, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Block 1 + Attention
        x = self.block1(x)
        x = self.attention1(x)
        
        # Block 2 + Attention
        x = self.block2(x)
        x = self.attention2(x)
        
        # Block 3 + Attention
        x = self.block3(x)
        x = self.attention3(x)
        
        # 展平特征
        x_flat = x.view(x.size(0), -1)
        
        # 特征级注意力
        attention_weights = self.feature_attention(x_flat)
        x_attended = x_flat * attention_weights
        
        # 分类
        output = self.classifier(x_attended)
        
        return output
    
    def get_attention_weights(self, x):
        """
        获取注意力权重用于可视化
        
        返回: 包含各层注意力权重的字典
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        attention_weights = {}
        
        # Block 1
        x = self.block1(x)
        x = self.attention1(x)
        
        # Block 2
        x = self.block2(x)
        x = self.attention2(x)
        
        # Block 3
        x = self.block3(x)
        x = self.attention3(x)
        
        # 特征级注意力权重
        x_flat = x.view(x.size(0), -1)
        feature_attention = self.feature_attention(x_flat)
        
        attention_weights['feature_attention'] = feature_attention.detach().cpu().numpy()
        
        return attention_weights
    
    def get_feature_maps(self, x):
        """
        获取中间特征图用于可视化
        
        返回: 包含各层特征图的字典
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        features = {}
        
        # Block 1
        x = self.block1(x)
        x = self.attention1(x)
        features['block1_attention'] = x.clone()
        
        # Block 2
        x = self.block2(x)
        x = self.attention2(x)
        features['block2_attention'] = x.clone()
        
        # Block 3
        x = self.block3(x)
        x = self.attention3(x)
        features['block3_attention'] = x.clone()
        
        return features
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'AttentionEEGNet',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'input_shape': (self.Chans, self.Samples),
            'output_classes': self.nb_classes,
            'feature_dim': self.feature_dim,
            'attention_dim': self.attention_dim
        }


# 为了兼容性，创建一个别名
AttentionEEGNet = EEGNet


if __name__ == "__main__":
    # 测试模型
    model = EEGNet(
        nb_classes=2,
        Chans=23,
        Samples=512,
        dropoutRate=0.5,
        attention_dim=64
    )
    
    # 打印模型信息
    print("AttentionEEGNet模型信息:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    x = torch.randn(4, 23, 512)  # [batch_size, channels, samples]
    print(f"\n输入形状: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
        
        # 测试注意力权重提取
        attention_weights = model.get_attention_weights(x)
        print("\n注意力权重形状:")
        for name, weights in attention_weights.items():
            print(f"  {name}: {weights.shape}")
        
        # 测试特征图提取
        features = model.get_feature_maps(x)
        print("\n特征图形状:")
        for name, feature in features.items():
            print(f"  {name}: {feature.shape}")