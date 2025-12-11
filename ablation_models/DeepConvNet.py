"""
DeepConvNet模型实现 - PyTorch版本
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


class DeepConvNet(nn.Module):
    """
    DeepConvNet模型 - 适配CHB-MIT癫痫检测数据格式
    
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
        super(DeepConvNet, self).__init__()
        
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        
        # Block 1: 时间卷积 + 空间卷积
        self.block1 = nn.Sequential(
            # 时间卷积 - 提取时域特征
            nn.Conv2d(1, 25, (1, 10), bias=False),
            # 空间卷积 - 提取通道间关系 
            nn.Conv2d(25, 25, (Chans, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropoutRate)
        )
        
        # Block 2: 深度卷积
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 10), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropoutRate)
        )
        
        # Block 3: 深度卷积
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 10), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropoutRate)
        )
        
        # Block 4: 深度卷积
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 10), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropoutRate)
        )
        
        # 计算全连接层输入维度
        # 通过多次池化: 3^4 = 81倍下采样 (近似)
        # 实际计算考虑卷积核大小的影响
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
            
            x = self.block1(dummy_input)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            
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
        
        # Block 1: 时间卷积 + 空间卷积
        x = self.block1(x)  # [B, 25, 1, T']
        
        # Block 2-4: 深度卷积层
        x = self.block2(x)  # [B, 50, 1, T'']
        x = self.block3(x)  # [B, 100, 1, T''']
        x = self.block4(x)  # [B, 200, 1, T'''']
        
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
        
        # Block 4
        x = self.block4(x)
        features['block4'] = x.clone()
        
        return features


# 测试函数
def test_deepconvnet():
    """测试DeepConvNet模型"""
    print("测试DeepConvNet模型...")
    
    # 创建模型
    model = DeepConvNet(nb_classes=2, Chans=23, Samples=512, dropoutRate=0.5)
    
    # 创建测试数据
    batch_size = 4
    test_input = torch.randn(batch_size, 23, 512)
    
    # 前向传播
    with torch.no_grad():
        output = model(test_input)
        features = model.get_feature_maps(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("特征图形状:")
    for name, feature in features.items():
        print(f"  {name}: {feature.shape}")
    
    print("DeepConvNet模型测试完成!")
    return True


if __name__ == "__main__":
    test_deepconvnet()