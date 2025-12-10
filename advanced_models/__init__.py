"""
高级EEG癫痫检测模型集合

包含从EEG-Epilepsy-Prediction项目整合的先进模型:
- AttentionBiLSTM: 多尺度注意力BiLSTM模型
- 特征提取和数据处理工具
"""

from .AttentionBiLSTM import AttentionBiLSTM, MultiScaleAttentionBiLSTM

__all__ = [
    'AttentionBiLSTM',
    'MultiScaleAttentionBiLSTM',
]

__version__ = '1.0.0'