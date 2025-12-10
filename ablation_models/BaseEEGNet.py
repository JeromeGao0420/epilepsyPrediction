"""
基础EEGNet模型实现
基于Lawhern et al. (2018)的EEGNet架构

参考文献:
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).
EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces.
Journal of neural engineering, 15(5), 056013.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet模型实现
    
    参数:
    - nb_classes: 分类数量 (癫痫检测通常是2: 发作/不发作)
    - Chans: EEG通道数 (CHB-MIT数据集通常是22或23)
    - Samples: 时间点数 (如果切片是2秒，采样率256Hz，这里就是512)
    - dropoutRate: Dropout比例
    - kernLength: 第一层卷积核长度
    - F1: 第一层滤波器数量
    - D: 深度乘数
    - F2: 第二层滤波器数量
    - norm_rate: 约束参数
    - dropoutType: Dropout类型
    """
    
    def __init__(self, nb_classes, Chans=22, Samples=512, 
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, 
                 norm_rate=0.25, dropoutType='Dropout'):
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
        
        # Block 1: 时间卷积 (Temporal Convolution)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # Block 2: 深度卷积 (Depthwise Convolution) - 空间滤波，提取不同通道的关系
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        # Block 3: 可分离卷积 (Separable Convolution)
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
        
        # 计算全连接层输入维度
        # 经过两次池化: 4 * 8 = 32倍下采样
        self.feature_dim = F2 * (Samples // 32)
        
        # 分类层
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
        # 输入形状: [B, C, T] -> [B, 1, C, T] (添加一个维度用于2D卷积)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, C, T]
        
        # Block 1: 时间卷积
        x = self.block1(x)  # [B, F1, C, T]
        
        # Block 2: 深度卷积 + 池化
        x = self.block2(x)  # [B, F1*D, 1, T//4]
        
        # Block 3: 可分离卷积 + 池化
        x = self.block3(x)  # [B, F2, 1, T//32]
        
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
        
        # Block 1
        x = self.block1(x)
        features['block1'] = x.clone()
        
        # Block 2
        x = self.block2(x)
        features['block2'] = x.clone()
        
        # Block 3
        x = self.block3(x)
        features['block3'] = x.clone()
        
        return features
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'EEGNet',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'input_shape': (self.Chans, self.Samples),
            'output_classes': self.nb_classes,
            'feature_dim': self.feature_dim
        }


# 为了兼容性，创建一个别名
BaseEEGNet = EEGNet


if __name__ == "__main__":
    # 测试模型
    model = EEGNet(
        nb_classes=2,
        Chans=23,
        Samples=512,
        dropoutRate=0.5
    )
    
    # 打印模型信息
    print("EEGNet模型信息:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    x = torch.randn(4, 23, 512)  # [batch_size, channels, samples]
    print(f"\n输入形状: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
        
        # 测试特征图提取
        features = model.get_feature_maps(x)
        print("\n特征图形状:")
        for name, feature in features.items():
            print(f"  {name}: {feature.shape}")