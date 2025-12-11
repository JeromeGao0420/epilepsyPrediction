# 🧠 癫痫预测 - 多模型对比研究

**基于深度学习的EEG癫痫检测与分型框架**

本项目实现了多种先进的深度学习模型用于EEG癫痫检测，使用公开的CHB-MIT数据集优化，提供完整的训练、测试和可视化分析流程。

## 🎯 项目特色

### 🔥 **核心亮点**
- **6种深度学习模型**: 从基础CNN到先进的注意力机制和Transformer架构
- **多尺度注意力机制**: 通道注意力 + 时序注意力双重机制
- **端到端流程**: 数据预处理 → 模型训练 → 可视化分析 → 性能评估
- **内存优化**: 支持8GB GPU运行，智能内存管理和批处理
- **脑电地形图可视化**: 提供临床可解释的诊断依据
- **完整测试套件**: 模型兼容性和集成测试

### 📊 **训练的模型架构**

| 模型 | 类型 | 模型架构 | 
|------|------|----------|
| **BaseEEGNet** | 基础CNN | 轻量级卷积网络 |
| **AttentionEEGNet** | CNN+注意力 | 通道+空间注意力 
| **AttentionBiLSTM** | LSTM+注意力 | 多尺度注意力+时序建模
| **DeepConvNet** | 深度CNN | 多层卷积特征提取 
| **ShallowConvNet** | 浅层CNN | 快速训练收敛 
| **TCFormer** | Transformer | 多核卷积+自注意力

## 🚀 快速开始

### 1️⃣ **环境准备**

```bash
# 创建虚拟环境
conda create -n epilepsy_detection python=3.11
conda activate epilepsy_detection

# 安装依赖
pip install -r requirements_enhanced.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2️⃣ **数据准备**


需要从 [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/ "公开数据集CHB-MIT下载链接") 下载公开数据集 CHB-MIT<span style="color: #888888;">（由于数据集太大我们只选取了前5个病人的数据进行研究）</span>

```bash
# 运行内存优化的数据预处理脚本
python prepare_data_balanced_memory.py
```

**数据处理详细流程**:

#### 📊 **CHB-MIT数据集处理**
- **数据来源**: CHB-MIT Scalp EEG Database (PhysioNet)
- **患者数量**: 24名儿童癫痫患者 (本研究使用前5名患者数据)
- **通道配置**: 23通道头皮EEG记录
- **采样频率**: 256 Hz
- **数据格式**: EDF (European Data Format)

#### 🔧 **预处理流程**
1. **信号滤波**:
   - 带通滤波: 0.5-40Hz
   - 去除工频干扰和高频噪声
   - 保留癫痫相关的关键频段

2. **时间窗口分割**:
   - 窗口大小: 2秒 (512个时间点 @ 256Hz)
   - 发作期重叠采样: 0.5秒步长 (提高发作期样本数)
   - 非发作期采样: 2秒步长 (减少冗余)

3. **标签定义**:
   - 发作期 (Ictal): 标签=1
   - 非发作期 (Interictal): 标签=0
   - 基于summary文件精确标注发作时间段

#### ⚖️ **数据平衡策略**
- **原始分布**: 严重不平衡 (发作期 << 非发作期)
- **平衡方法**:
  - 发作期样本: 重复采样增强
  - 非发作期样本: 随机下采样
  - 目标比例: 30% 发作期, 70% 非发作期
- **最终数据集**:
  - 训练集: 5,000样本 (1,500发作期 + 3,500非发作期)
  - 测试集: 1,966样本 (590发作期 + 1,376非发作期)

#### 💾 **内存优化技术**
- **生成器模式**: 避免一次性加载所有数据
- **分批处理**: 1,000样本为一批，减少内存峰值
- **临时文件**: 中间结果保存，支持大规模数据
- **垃圾回收**: 及时释放内存，支持8GB GPU训练

### 3️⃣ **测试验证**

```bash
# 运行模型兼容性测试
python test_comparison_models.py
```

### 4️⃣ **模型训练**

```bash
# 训练多尺度注意力BiLSTM 
python train_enhanced.py --model AttentionBiLSTM 

# 训练Transformer模型
python train_enhanced.py --model TCFormer 

# 训练基础EEGNet
python train_enhanced.py --model BaseEEGNet 

# 训练深度卷积网络
python train_enhanced.py --model DeepConvNet 

# 训练浅层卷积网络
python train_enhanced.py --model ShallowConvNet

# 训练注意力EEGNet
python train_enhanced.py --model AttentionEEGNet 
```

### 5️⃣ **可视化分析**

```bash
# 生成脑电地形图可视化
python visualize_topomap.py --model AttentionBiLSTM
python visualize_topomap.py --model TCFormer
python visualize_topomap.py --model BaseEEGNet
python visualize_topomap.py --model DeepConvNet
python visualize_topomap.py --model ShallowConvNet
python visualize_topomap.py --model AttentionEEGNet
```



## 🏗️ 项目架构

```
epilepsyPrediction/
├── 📁 ablation_models/          # 🧠 深度学习模型库(消融实验)
│   ├── AttentionBiLSTM.py      # 多尺度注意力BiLSTM
│   ├── AttentionEEGNet.py      # 注意力增强EEGNet
│   ├── BaseEEGNet.py           # 基础EEGNet
│   ├── DeepConvNet.py          # 深度卷积网络
│   ├── ShallowConvNet.py       # 浅层卷积网络
│   └── TCFormer.py             # Transformer架构
├── 📁 data/                     # 📊 处理后数据
│   ├── X_train_balanced.npy    # 训练集特征
│   ├── y_train_balanced.npy    # 训练集标签
│   ├── X_test_balanced.npy     # 测试集特征
│   └── y_test_balanced.npy     # 测试集标签
├── 📁 database/                 # 🗄️ 原始CHB-MIT数据
│   └── physionet.org/files/
├── 📁 outputs/                  # 📈 训练输出和可视化
│   ├── logs/                   # 训练日志
│   ├── models/                 # 保存的模型
│   └── *.png                   # 可视化图表
├── 📁 pictures/                 # 🖼️ 示例图片
├── 🐍 train_enhanced.py         # 🎯 统一训练脚本
├── 🐍 visualize_topomap.py      # 🎨 脑电地形图可视化
├── 🐍 test_comparison_models.py # ✅ 模型测试脚本
├── 🐍 prepare_data_balanced_memory.py # 📊 数据预处理
├── 📄 requirements_enhanced.txt # 📦 项目依赖
└── 📖 README.md        # 📚 项目文档
```

## 🧠 模型架构详解

### 🎯 **核心创新: AttentionEEGNet**

**本文提出的AttentionEEGNet是在BaseEEGNet基础上的创新改进，构成消融实验对比**:

#### 🔥 **多层注意力机制设计**
1. **通道注意力模块 (Channel Attention)**
   - SE-Net变体设计，结合平均池化和最大池化
   - 自适应学习重要EEG通道权重
   - 突出病理相关的脑区活动模式

2. **空间注意力模块 (Spatial Attention)**
   - 基于通道维度统计的空间特征增强
   - 7×7卷积核进行空间注意力计算
   - 聚焦于空间上的关键区域

3. **特征级注意力 (Feature-level Attention)**
   - 分类前的特征级权重调整机制
   - 全连接层生成注意力权重
   - 进一步优化最终特征表示

#### 📊 **消融实验设计**
- **BaseEEGNet** (基线模型): 标准EEGNet架构，无注意力机制
- **AttentionEEGNet** (本文模型): BaseEEGNet + 多层注意力机制

### 🔍 **对比实验模型**

#### **AttentionBiLSTM - 多尺度注意力架构**
- 多头自注意力机制 (4个注意力头)
- 2层双向LSTM (隐藏层128维)
- 时序建模与注意力机制结合

#### **TCFormer - Transformer混合架构**
- 多核卷积块: 16、32、64不同尺度时域卷积
- 8个注意力头的自注意力机制
- 时域卷积网络保持因果关系

#### **传统CNN对比模型**
- **DeepConvNet**: 深层卷积特征提取网络
- **ShallowConvNet**: 浅层网络快速收敛架构

## 📋 使用指南

### 🎛️ **训练参数配置**

```bash
# 完整参数示例
python train_enhanced.py \
    --model AttentionBiLSTM \
    --batch_size 32 \
    --learning_rate 0.001 \
    --epochs 30 \
    --hidden_dim 128 \
    --num_layers 2 \
    --attention_heads 4 \
    --dropout 0.15 \
    --save_attention \
    --device auto
```

### 📊 **数据格式规范**

- **输入**: `[B, C, T]` - (批次, 通道, 时间点)
  - B: 批次大小 (默认32)
  - C: 23个EEG通道 (CHB-MIT标准)
  - T: 512个时间点 (2秒 × 256Hz采样率)

- **输出**: `[B, 2]` - (批次, 分类数)
  - 0: 正常状态 (非癫痫)
  - 1: 癫痫发作

### 🎨 **可视化功能**

**脑电地形图分析**:
```bash
# 生成所有模型的地形图对比
for model in AttentionBiLSTM TCFormer AttentionEEGNet DeepConvNet ShallowConvNet BaseEEGNet; do
    python visualize_topomap.py --model $model
done
```

**可视化输出**:
- 训练历史曲线 (`*_training_history.png`)
- 混淆矩阵 (`*_confusion_matrix.png`)
- 脑电地形图分析 (`*_topomap_analysis.png`)

## 📈 性能基准

### 🏆 **模型性能对比** (基于CHB-MIT测试集)

#### **消融实验结果** (验证注意力机制有效性)
| 模型 | 准确率 | F1分数 | 精确率 | 召回率 |
|------|--------|--------|--------|--------|
| **AttentionEEGNet** | **79.15%** | **80.00%** | **85.25%** | **79.15%** |
| BaseEEGNet | 61.29% | 61.76% | 80.90% | 61.29% |

#### **对比实验结果** (与其他先进方法对比)
| 模型 | 准确率 | F1分数 | 精确率 | 召回率 | 参数量 |
|------|--------|--------|--------|--------|--------|
| **AttentionEEGNet** | **79.15%** | **80.00%** | **85.25%** | **79.15%** | **35,362** |
| DeepConvNet | 75.79% | 76.72% | 80.33% | 75.79% | 278,277 |
| AttentionBiLSTM | 69.99% | 57.63% | 48.99% | 69.99% | 1,128,194 |
| BaseEEGNet | 61.29% | 61.76% | 80.90% | 61.29% | 1,986 |
| ShallowConvNet | 60.17% | 61.57% | 72.23% | 60.17% | 40,122 |
| TCFormer | 44.66% | 40.66% | 76.17% | 44.66% | 227,858 |

**🎯 AttentionEEGNet核心优势**:
- ✅ **显著性能提升**: 相比BaseEEGNet准确率提升17.86个百分点 (79.15% vs 61.29%)
- ✅ **F1分数改善**: 从61.76%提升到80.00%，提升18.24个百分点
- ✅ **参数效率**: 仅增加33K参数实现大幅性能提升
- ✅ **计算效率**: 训练时间合理，适合实际应用部署

### 📊 **数据集统计**

- **训练集**: 平衡后约5,000样本
- **测试集**: 约2,000样本
- **数据平衡**: SMOTE过采样 + 随机欠采样
- **通道配置**: 23个标准10-20系统EEG通道
- **采样率**: 256Hz
- **窗口长度**: 2秒 (512个时间点)

## 🔧 高级功能

### 🎯 **超参数优化**

```bash
# 网格搜索示例
for lr in 0.001 0.0005 0.0001; do
    for batch in 16 32 64; do
        python train_enhanced.py --model AttentionBiLSTM \
            --learning_rate $lr --batch_size $batch
    done
done
```

### 🧪 **消融研究**

```bash
# 测试不同注意力头数
for heads in 2 4 8 16; do
    python train_enhanced.py --model AttentionBiLSTM \
        --attention_heads $heads --save_attention
done

# 测试不同隐藏层维度
for dim in 64 128 256 512; do
    python train_enhanced.py --model AttentionBiLSTM \
        --hidden_dim $dim
done
```

### 📊 **批量实验**

```bash
# 运行所有模型对比实验
models=("BaseEEGNet" "AttentionEEGNet" "AttentionBiLSTM" "DeepConvNet" "ShallowConvNet" "TCFormer")
for model in "${models[@]}"; do
    echo "Training $model..."
    python train_enhanced.py --model $model --epochs 30
    python visualize_topomap.py --model $model
done
```

## 🛠️ 开发指南

### 🔍 **添加新模型**

1. 在 `ablation_models/` 中创建新模型文件
2. 确保模型接口兼容: `__init__(nb_classes, Chans, Samples, ...)`
3. 实现 `forward()` 方法，输入输出格式: `[B,C,T] → [B,num_classes]`
4. 在 [`train_enhanced.py`](train_enhanced.py:49) 中添加模型选项
5. 运行 [`test_comparison_models.py`](test_comparison_models.py) 验证兼容性

### 🧪 **测试流程**

```bash
# 完整测试套件
python test_comparison_models.py

# 单独测试模型加载
python -c "
from ablation_models.AttentionBiLSTM import AttentionBiLSTM
model = AttentionBiLSTM()
print('AttentionBiLSTM模型测试通过!')
"

# 测试数据处理流程
python -c "
import numpy as np
X_train = np.load('data/X_train_balanced.npy')
print(f'训练数据形状: {X_train.shape}')
"
```

### 📁 **文件组织规范**

```python
# 模型文件结构
class YourModel(nn.Module):
    def __init__(self, nb_classes=2, Chans=23, Samples=512, **kwargs):
        super().__init__()
        # 模型定义
    
    def forward(self, x):
        # x: [batch_size, channels, time_points]
        # return: [batch_size, nb_classes]
        pass
    
    def get_feature_maps(self, x):  # 可选
        # 返回中间特征图用于可视化
        pass
```

##  技术参考

### 📖 **相关论文**

1. **EEGNet**: Lawhern et al. (2018) - "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces"
2. **注意力机制**: Vaswani et al. (2017) - "Attention Is All You Need"
3. **BiLSTM**: Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
4. **CHB-MIT数据集**: Shoeb (2009) - "Application of machine learning to epileptic seizure onset detection and treatment"
5. **DeepConvNet**: Schirrmeister et al. (2017) - "Deep learning with convolutional neural networks for EEG decoding and visualization"

### 🔗 **相关资源**

- [CHB-MIT数据集](https://physionet.org/content/chbmit/1.0.0/) - 儿童医院波士顿癫痫数据
- [PyTorch官方文档](https://pytorch.org/docs/) - 深度学习框架
- [MNE-Python](https://mne.tools/) - 神经生理学数据处理
- [scikit-learn](https://scikit-learn.org/) - 机器学习工具包

### 🏥 **临床应用**

- **实时监测**: 模型可部署用于实时EEG监测
- **辅助诊断**: 为临床医生提供客观的癫痫检测支持
- **研究工具**: 用于癫痫发作机制研究和新药评估

## 🤝 贡献指南

### 🐛 **问题报告**

如遇到问题，请提供：
1. 完整的错误信息和堆栈跟踪
2. 运行环境信息 (Python版本、PyTorch版本、GPU型号等)
3. 详细的复现步骤和数据信息
4. 期望的行为描述

### 💡 **功能建议**

欢迎提出：
- 新的深度学习架构 (如Graph Neural Networks)
- 改进的注意力机制设计
- 数据增强和预处理技术
- 模型解释性和可视化改进
- 性能优化方案

### 🔧 **代码贡献**

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- **数据提供**: CHB-MIT数据集团队和PhysioNet平台
- **开源社区**: PyTorch、MNE-Python、scikit-learn等开源项目
- **学术贡献**: EEGNet、Transformer等经典架构的原始作者
- **临床合作**: 为项目提供医学指导的临床专家

---

## 🚨 重要声明

### ⚠️ **医疗免责声明**
本项目仅用于**研究和教育目的**，不能替代专业医疗诊断。任何临床应用都需要经过严格的医疗验证和监管部门批准。

### 🔒 **数据隐私**
请确保在使用真实患者数据时遵守相关的数据保护法规和伦理准则。

### 📊 **性能说明**
模型性能可能因数据集、硬件配置和超参数设置而异。建议在自己的数据集上进行验证。

---

## 📞 **联系方式**

- 📧 **技术问题**: 请通过GitHub Issues提交
- 🐛 **Bug报告**: 使用Issue模板详细描述问题
- 💡 **功能请求**: 欢迎在Discussions中讨论新想法
- 🤝 **合作机会**: 欢迎学术和工业界合作

---

**🎉 祝您使用愉快！让我们一起推动EEG癫痫检测技术的发展！**

*最后更新: 2025年12月*