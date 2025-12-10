#!/usr/bin/env python3
"""
CHB-MIT数据集平衡预处理脚本 - 内存优化版本
解决类别不平衡问题，支持大文件处理

内存优化策略：
1. 使用生成器模式，避免一次性加载所有数据到内存
2. 分批处理和保存数据，减少内存峰值使用
3. 及时释放不再使用的变量，使用垃圾回收
4. 使用中间文件处理大型数据集
"""

import mne
import os
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Generator
from collections import Counter
import logging
import gc
import pickle
from sklearn.utils import resample

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """获取当前内存使用情况"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        return 0.0

def parse_summary_file(summary_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """解析summary文件，提取每个文件的发作期时间段"""
    seizure_info = {}
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取文件信息
    file_pattern = r'File Name: (chb\d+_\d+\.edf)'
    seizure_pattern = r'Seizure Start Time: (\d+(?:\.\d+)?) seconds\s+Seizure End Time: (\d+(?:\.\d+)?) seconds'
    
    files = re.finditer(file_pattern, content)
    for file_match in files:
        filename = file_match.group(1)
        file_start_pos = file_match.end()
        
        # 查找下一个文件的位置
        next_file = re.search(file_pattern, content[file_start_pos:])
        if next_file:
            file_section = content[file_start_pos:file_start_pos + next_file.start()]
        else:
            file_section = content[file_start_pos:]
        
        # 提取该文件的发作期信息
        seizures = []
        for seizure_match in re.finditer(seizure_pattern, file_section):
            start_time = float(seizure_match.group(1))
            end_time = float(seizure_match.group(2))
            seizures.append((start_time, end_time))
        
        seizure_info[filename] = seizures
    
    return seizure_info

def apply_bandpass_filter(raw, l_freq=0.5, h_freq=40.0):
    """对EEG数据进行带通滤波"""
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', skip_by_annotation='edge')
    return raw_filtered

def is_seizure_period(t_start: float, t_end: float, seizure_periods: List[Tuple[float, float]]) -> bool:
    """判断时间段是否与发作期重叠"""
    for seizure_start, seizure_end in seizure_periods:
        if not (t_end <= seizure_start or t_start >= seizure_end):
            return True
    return False

def window_seizure_data_memory_efficient(raw, seizure_periods: List[Tuple[float, float]], 
                                        window_size: float, step_size: float = 0.5):
    """内存高效的发作期数据窗口切片"""
    windows = []
    sfreq = raw.info['sfreq']
    window_samples = int(window_size * sfreq)
    
    for seizure_start, seizure_end in seizure_periods:
        # 确保不超出数据范围
        seizure_start = max(0, seizure_start)
        seizure_end = min(raw.times[-1], seizure_end)
        
        # 重叠采样
        current_start = seizure_start
        while current_start + window_size <= seizure_end:
            # 提取窗口数据
            start_idx = int(current_start * sfreq)
            end_idx = start_idx + window_samples
            
            if end_idx <= raw.n_times:
                window_data = raw[:, start_idx:end_idx][0]  # 获取数据数组
                windows.append(window_data)
            
            current_start += step_size
    
    return windows

def window_seizure_data_generator(raw, seizure_periods: List[Tuple[float, float]], 
                                 window_size: float, step_size: float = 0.5):
    """生成器版本的发作期数据窗口切片，避免内存累积"""
    sfreq = raw.info['sfreq']
    window_samples = int(window_size * sfreq)
    
    logger.info(f"正在处理发作期数据...")
    seizure_windows = 0
    
    for seizure_start, seizure_end in seizure_periods:
        # 确保不超出数据范围
        seizure_start = max(0, seizure_start)
        seizure_end = min(raw.times[-1], seizure_end)
        
        # 重叠采样
        current_start = seizure_start
        while current_start + window_size <= seizure_end:
            start_idx = int(current_start * sfreq)
            end_idx = start_idx + window_samples
            
            if end_idx <= raw.n_times:
                window_data = raw[:, start_idx:end_idx][0]  # 获取数据数组
                yield window_data.astype(np.float32), 1
                seizure_windows += 1
            
            current_start += step_size
    
    logger.info(f"发作期窗口生成完成: {seizure_windows} 个")

def window_non_seizure_data_memory_efficient(raw, seizure_periods: List[Tuple[float, float]], 
                                            window_size: float, step_size: float = 2.0):
    """内存高效的非发作期数据窗口切片"""
    windows = []
    sfreq = raw.info['sfreq']
    window_samples = int(window_size * sfreq)
    total_duration = raw.times[-1]
    
    # 将发作期时间段合并并排序
    seizure_ranges = sorted(seizure_periods, key=lambda x: x[0])
    
    # 找出所有非发作期时间段
    non_seizure_ranges = []
    current_start = 0.0
    
    for seizure_start, seizure_end in seizure_ranges:
        if current_start < seizure_start:
            non_seizure_ranges.append((current_start, seizure_start))
        current_start = max(current_start, seizure_end)
    
    # 处理最后一个时间段
    if current_start < total_duration:
        non_seizure_ranges.append((current_start, total_duration))
    
    # 对每个非发作期时间段进行窗口切片
    for range_start, range_end in non_seizure_ranges:
        current_start = range_start
        while current_start + window_size <= range_end:
            start_idx = int(current_start * sfreq)
            end_idx = start_idx + window_samples
            
            if end_idx <= raw.n_times:
                window_data = raw[:, start_idx:end_idx][0]  # 获取数据数组
                windows.append(window_data)
            
            current_start += step_size
    
    return windows

def window_non_seizure_data_generator(raw, seizure_periods: List[Tuple[float, float]], 
                                     window_size: float, step_size: float = 2.0):
    """生成器版本的非发作期数据窗口切片，避免内存累积"""
    sfreq = raw.info['sfreq']
    window_samples = int(window_size * sfreq)
    total_duration = raw.times[-1]
    
    logger.info(f"正在处理非发作期数据...")
    non_seizure_windows = 0
    
    # 将发作期时间段合并并排序
    seizure_ranges = sorted(seizure_periods, key=lambda x: x[0])
    
    # 找出所有非发作期时间段
    non_seizure_ranges = []
    current_start = 0.0
    
    for seizure_start, seizure_end in seizure_ranges:
        if current_start < seizure_start:
            non_seizure_ranges.append((current_start, seizure_start))
        current_start = max(current_start, seizure_end)
    
    # 处理最后一个时间段
    if current_start < total_duration:
        non_seizure_ranges.append((current_start, total_duration))
    
    # 对每个非发作期时间段进行非重叠窗口切片
    for range_start, range_end in non_seizure_ranges:
        current_start = range_start
        while current_start + window_size <= range_end:
            start_idx = int(current_start * sfreq)
            end_idx = start_idx + window_samples
            
            if end_idx <= raw.n_times:
                window_data = raw[:, start_idx:end_idx][0]  # 获取数据数组
                yield window_data.astype(np.float32), 0
                non_seizure_windows += 1
            
            current_start += step_size  # 非重叠
    
    logger.info(f"非发作期窗口生成完成: {non_seizure_windows} 个")
    
    # 强制垃圾回收
    gc.collect()

def save_data_in_batches(data_generator, batch_size: int = 1000, output_prefix: str = "batch"):
    """
    分批保存数据到临时文件，避免内存累积
    
    Args:
        data_generator: 数据生成器
        batch_size: 每批数据大小
        output_prefix: 输出文件前缀
        
    Returns:
        保存的文件路径列表
    """
    temp_files = []
    current_batch_data = []
    current_batch_labels = []
    batch_count = 0
    
    # 创建temp目录
    temp_dir = "temp"
    try:
        os.makedirs(temp_dir, exist_ok=True)
        # 确保目录有写入权限
        if not os.access(temp_dir, os.W_OK):
            logger.error(f"没有权限写入目录 {temp_dir}")
            return []
    except Exception as e:
        logger.error(f"创建目录 {temp_dir} 时出错: {e}")
        return []
    
    logger.info(f"开始分批保存数据，批大小: {batch_size}")
    
    for window_data, label in data_generator:
        current_batch_data.append(window_data)
        current_batch_labels.append(label)
        
        # 当达到批大小时，保存当前批次
        if len(current_batch_data) >= batch_size:
            batch_count += 1
            temp_file = os.path.join(temp_dir, f"{output_prefix}_{batch_count:04d}.npz")
            
            try:
                # 保存当前批次
                np.savez_compressed(temp_file,
                                  data=np.array(current_batch_data, dtype=np.float32),
                                  labels=np.array(current_batch_labels, dtype=np.int64))
                
                temp_files.append(temp_file)
                logger.info(f"保存批次 {batch_count}: {len(current_batch_data)} 样本 -> {temp_file}")
            except Exception as e:
                logger.error(f"保存批次 {batch_count} 时出错: {e}")
                # 继续处理，不中断整个流程
                continue
            
            # 清空当前批次数据，释放内存
            current_batch_data.clear()
            current_batch_labels.clear()
            gc.collect()
    
    # 保存最后一批（如果有剩余）
    if current_batch_data:
        batch_count += 1
        temp_file = os.path.join(temp_dir, f"{output_prefix}_{batch_count:04d}.npz")
        
        try:
            np.savez_compressed(temp_file,
                              data=np.array(current_batch_data, dtype=np.float32),
                              labels=np.array(current_batch_labels, dtype=np.int64))
            
            temp_files.append(temp_file)
            logger.info(f"保存最后批次 {batch_count}: {len(current_batch_data)} 样本 -> {temp_file}")
        except Exception as e:
            logger.error(f"保存最后批次时出错: {e}")
    
    logger.info(f"分批保存完成，共 {len(temp_files)} 个文件")
    return temp_files

def merge_batch_files(temp_files: List[str], final_output_path: str):
    """
    合并分批保存的文件为最终数据文件
    
    Args:
        temp_files: 临时文件列表
        final_output_path: 最终输出文件路径
    """
    logger.info(f"开始合并 {len(temp_files)} 个批次文件...")
    
    if not temp_files:
        logger.warning("没有临时文件需要合并")
        return False
    
    all_data = []
    all_labels = []
    expected_shape = None
    
    for temp_file in temp_files:
        try:
            data = np.load(temp_file)
            current_data = data['data']
            current_labels = data['labels']
            
            # 检查数据形状的一致性
            if expected_shape is None:
                expected_shape = current_data.shape[1:]  # 记录第一个文件的形状（排除样本数维度）
                logger.info(f"期望的数据形状: {expected_shape}")
            
            # 验证当前文件的数据形状
            if current_data.shape[1:] != expected_shape:
                logger.warning(f"跳过文件 {temp_file}: 形状不匹配 {current_data.shape[1:]} vs {expected_shape}")
                data.close()
                continue
            
            all_data.append(current_data)
            all_labels.append(current_labels)
            data.close()  # 关闭文件
            
        except Exception as e:
            logger.error(f"加载临时文件 {temp_file} 时出错: {e}")
            continue
    
    if not all_data:
        logger.warning("没有成功加载任何数据")
        return False
    
    # 合并所有数据
    final_data = np.concatenate(all_data, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    
    try:
        # 保存最终文件
        np.savez_compressed(final_output_path, data=final_data, labels=final_labels)
        
        logger.info(f"合并完成: {final_data.shape[0]} 样本 -> {final_output_path}")
        
        # 清理临时文件
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        # 强制垃圾回收
        del all_data, all_labels, final_data, final_labels
        gc.collect()
        
        # 如果temp目录为空，删除它
        temp_dir = "temp"
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            try:
                os.rmdir(temp_dir)
                logger.info(f"已清理空目录: {temp_dir}")
            except:
                pass
        
        return True
        
    except Exception as e:
        logger.error(f"保存最终文件时出错: {e}")
        return False

def process_single_file_generator(edf_path: str, summary_path: str, window_size: float = 2.0,
                                 l_freq: float = 0.5, h_freq: float = 40.0):
    """
    处理单个EEG文件的生成器版本，避免内存累积
    
    Args:
        edf_path: EDF文件路径
        summary_path: summary文件路径
        window_size: 窗口大小（秒）
        l_freq: 低频截止频率
        h_freq: 高频截止频率
        
    Yields:
        (window_data, label) 元组
    """
    logger.info(f"正在读取文件: {edf_path}")
    logger.info(f"内存使用: {get_memory_usage():.1f} MB")
    
    try:
        # 使用lazy loading，不预加载数据
        raw = mne.io.read_raw_edf(edf_path, preload=False)
        
        # 加载数据到内存以进行滤波处理
        logger.info(f"正在加载数据到内存...")
        raw.load_data()
        
        logger.info(f"正在应用带通滤波 ({l_freq}-{h_freq} Hz)...")
        raw_filtered = apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)
        
        # 解析summary文件获取发作期信息
        seizure_info = parse_summary_file(summary_path)
        filename = os.path.basename(edf_path)
        
        if filename not in seizure_info:
            logger.warning(f"在summary文件中未找到 {filename} 的信息")
            seizure_periods = []
        else:
            seizure_periods = seizure_info[filename]
            logger.info(f"找到 {len(seizure_periods)} 个发作期")
        
        # 使用生成器逐个处理窗口，避免内存累积
        yield from window_seizure_data_generator(raw_filtered, seizure_periods, window_size, step_size=0.5)
        yield from window_non_seizure_data_generator(raw_filtered, seizure_periods, window_size, step_size=2.0)
        
    except Exception as e:
        logger.error(f"处理文件 {edf_path} 时出错: {e}")
        # 即使出错也确保清理资源
        if 'raw' in locals():
            try:
                raw.close()
            except:
                pass
        return
    
    finally:
        # 关闭raw对象，释放内存
        if 'raw' in locals():
            try:
                raw.close()
                del raw
                gc.collect()
                logger.info(f"文件 {edf_path} 处理完成，内存使用: {get_memory_usage():.1f} MB")
            except:
                pass

def process_single_file_memory_efficient(edf_path: str, summary_path: str, 
                                       window_size: float = 2.0, 
                                       l_freq: float = 0.5, h_freq: float = 40.0):
    """内存高效处理单个EDF文件（兼容旧版本）"""
    try:
        logger.info(f"正在读取文件: {edf_path}")
        # 使用preload=False减少内存占用
        raw = mne.io.read_raw_edf(edf_path, preload=False)
        
        # 只加载需要的数据范围
        raw.load_data()
        
        logger.info(f"正在应用带通滤波 ({l_freq}-{h_freq} Hz)...")
        raw_filtered = apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)
        
        # 解析summary文件获取发作期信息
        seizure_info = parse_summary_file(summary_path)
        filename = os.path.basename(edf_path)
        
        if filename not in seizure_info:
            logger.warning(f"在summary文件中未找到 {filename} 的信息")
            seizure_periods = []
        else:
            seizure_periods = seizure_info[filename]
            logger.info(f"找到 {len(seizure_periods)} 个发作期")
        
        # 窗口切片 - 内存高效版本
        logger.info(f"正在对发作期数据进行重叠窗口切片...")
        seizure_windows = window_seizure_data_memory_efficient(
            raw_filtered, seizure_periods, window_size=window_size, step_size=0.5
        )
        
        logger.info(f"正在对非发作期数据进行窗口切片...")
        non_seizure_windows = window_non_seizure_data_memory_efficient(
            raw_filtered, seizure_periods, window_size=window_size, step_size=2.0
        )
        
        # 创建标签
        labels = [1] * len(seizure_windows) + [0] * len(non_seizure_windows)
        
        # 合并窗口
        all_windows = seizure_windows + non_seizure_windows
        
        logger.info(f"处理完成: 发作期窗口 {len(seizure_windows)} 个, 非发作期窗口 {len(non_seizure_windows)} 个")
        
        # 清理内存
        del raw, raw_filtered
        gc.collect()
        
        return all_windows, labels
        
    except Exception as e:
        logger.error(f"处理文件 {edf_path} 时出错: {e}")
        return [], []

def balance_data_chunked(X_list, y_list, target_ratio=0.3, max_samples_per_class=5000):
    """分块平衡数据，限制每类最大样本数"""
    logger.info("开始分块数据平衡...")
    
    # 合并所有数据
    if not X_list:
        return np.array([]), np.array([])
    
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    logger.info(f"合并后数据分布: {Counter(y_all)}")
    
    # 计算各类别数量
    n_positive = np.sum(y_all == 1)
    n_negative = np.sum(y_all == 0)
    
    # 限制每类最大数量
    max_per_class = min(max_samples_per_class, max(n_positive, n_negative))
    
    # 平衡到目标比例
    target_n_positive = int(max_per_class * target_ratio)
    target_n_negative = int(max_per_class * (1 - target_ratio))
    
    # 确保至少有一定数量的样本
    target_n_positive = max(target_n_positive, min(100, n_positive))
    target_n_negative = max(target_n_negative, min(100, n_negative))
    
    # 实际采样
    if n_positive > 0 and target_n_positive > 0:
        positive_indices = np.where(y_all == 1)[0]
        if len(positive_indices) > target_n_positive:
            selected_positive = np.random.choice(positive_indices, target_n_positive, replace=False)
        else:
            # 如果样本不足，允许重复采样
            selected_positive = np.random.choice(positive_indices, target_n_positive, replace=True)
    else:
        selected_positive = np.array([])
    
    if n_negative > 0 and target_n_negative > 0:
        negative_indices = np.where(y_all == 0)[0]
        if len(negative_indices) > target_n_negative:
            selected_negative = np.random.choice(negative_indices, target_n_negative, replace=False)
        else:
            selected_negative = np.random.choice(negative_indices, target_n_negative, replace=True)
    else:
        selected_negative = np.array([])
    
    # 合并选择的索引
    selected_indices = np.concatenate([selected_positive, selected_negative])
    np.random.shuffle(selected_indices)
    
    X_balanced = X_all[selected_indices]
    y_balanced = y_all[selected_indices]
    
    logger.info(f"平衡后数据分布: {Counter(y_balanced)}")
    logger.info(f"平衡后数据形状: {X_balanced.shape}")
    
    return X_balanced, y_balanced

def process_patient_with_memory_control(patient_dir: Path, summary_file: Path,
                                      window_size: float = 2.0, batch_size: int = 1000):
    """
    以内存控制的方式处理单个病人的数据
    
    Args:
        patient_dir: 病人数据目录
        summary_file: summary文件路径
        window_size: 窗口大小
        batch_size: 批处理大小
        
    Returns:
        临时文件列表
    """
    patient_id = patient_dir.name
    logger.info(f"\n处理患者 {patient_id}...")
    logger.info(f"初始内存使用: {get_memory_usage():.1f} MB")
    
    temp_files = []
    
    # 获取所有EDF文件
    edf_files = list(patient_dir.glob('*.edf'))
    logger.info(f"找到 {len(edf_files)} 个EDF文件")
    
    for edf_file in edf_files:
        try:
            # 为每个文件创建生成器
            data_generator = process_single_file_generator(
                str(edf_file), str(summary_file), window_size
            )
            
            # 分批保存当前文件的数据
            file_temp_files = save_data_in_batches(
                data_generator,
                batch_size=batch_size,
                output_prefix=f"temp_{patient_id}_{edf_file.stem}"
            )
            
            if file_temp_files:  # 只有当成功生成临时文件时才添加
                temp_files.extend(file_temp_files)
                logger.info(f"文件 {edf_file.name} 处理完成，生成 {len(file_temp_files)} 个临时文件")
            
            # 强制垃圾回收
            gc.collect()
            logger.info(f"文件 {edf_file.name} 处理完成，内存使用: {get_memory_usage():.1f} MB")
            
        except Exception as e:
            logger.error(f"处理文件 {edf_file} 时出错: {e}")
            continue
    
    logger.info(f"患者 {patient_id} 处理完成，生成 {len(temp_files)} 个临时文件")
    return temp_files

def prepare_balanced_data_memory_efficient(data_dir="./database/physionet.org/files/chbmit/1.0.0/", 
                                         test_patients=['chb05'],
                                         window_size=2.0, target_ratio=0.3,
                                         max_samples_per_patient=2000, batch_size=1000):
    """
    内存高效的平衡数据集准备
    
    Args:
        data_dir: CHB-MIT数据目录
        test_patients: 用作测试集的患者列表
        window_size: 窗口大小（秒）
        target_ratio: 目标正类比例
        max_samples_per_patient: 每个患者最大样本数
        batch_size: 批处理大小
    
    Returns:
        成功返回True，失败返回False
    """
    logger.info("=== 开始内存高效的平衡数据集准备 ===")
    logger.info(f"批处理大小: {batch_size}")
    
    # 获取所有患者目录
    all_patient_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir() and d.name.startswith('chb')]
    
    # 标准化患者ID
    def normalize_patient_id(patient_id):
        return patient_id.lower().replace(' ', '')
    
    test_patients_normalized = [normalize_patient_id(p) for p in test_patients]
    train_patients = [p for p in all_patient_dirs if normalize_patient_id(p.name) not in test_patients_normalized]
    
    logger.info(f"训练集患者: {[p.name for p in train_patients]}")
    logger.info(f"测试集患者: {test_patients}")
    
    # 处理训练集患者 - 使用生成器模式
    train_temp_files = []
    for i, patient_dir in enumerate(train_patients):
        patient_id = patient_dir.name
        summary_file = patient_dir / f"{patient_id}-summary.txt"
        
        if not summary_file.exists():
            logger.warning(f"{patient_id} 的summary文件不存在，跳过")
            continue
        
        logger.info(f"\n[{i+1}/{len(train_patients)}] 处理训练患者 {patient_id}...")
        
        patient_temp_files = process_patient_with_memory_control(
            patient_dir, summary_file, window_size=window_size, batch_size=batch_size
        )
        
        if patient_temp_files:
            train_temp_files.extend(patient_temp_files)
        
        logger.info(f"  患者 {patient_id} 处理完成，生成 {len(patient_temp_files)} 个临时文件")
        
        # 每处理完一个患者就清理内存
        gc.collect()
    
    # 处理测试集患者
    test_temp_files = []
    for i, patient_name in enumerate(test_patients_normalized):
        patient_dir = Path(data_dir) / patient_name
        summary_file = patient_dir / f"{patient_name}-summary.txt"
        
        if not summary_file.exists():
            logger.warning(f"{patient_name} 的summary文件不存在，跳过")
            continue
        
        logger.info(f"\n[{i+1}/{len(test_patients)}] 处理测试患者 {patient_name}...")
        
        patient_temp_files = process_patient_with_memory_control(
            patient_dir, summary_file, window_size=window_size, batch_size=batch_size
        )
        
        if patient_temp_files:
            test_temp_files.extend(patient_temp_files)
        
        logger.info(f"  患者 {patient_name} 处理完成，生成 {len(patient_temp_files)} 个临时文件")
        gc.collect()
    
    if not train_temp_files or not test_temp_files:
        logger.error("错误: 没有成功处理任何数据")
        return False
    
    # 合并训练集数据
    logger.info(f"\n合并训练集数据...")
    train_merge_success = merge_batch_files(train_temp_files, './data/train_data_balanced.npz')
    if not train_merge_success:
        logger.error("错误: 训练集数据合并失败")
        return False
    
    # 合并测试集数据
    logger.info("合并测试集数据...")
    test_merge_success = merge_batch_files(test_temp_files, './data/test_data_balanced.npz')
    if not test_merge_success:
        logger.error("错误: 测试集数据合并失败")
        return False
    
    # 加载合并后的数据
    train_data = np.load('./data/train_data_balanced.npz')
    test_data = np.load('./data/test_data_balanced.npz')
    
    X_train_all = train_data['data']
    y_train_all = train_data['labels']
    X_test_all = test_data['data']
    y_test_all = test_data['labels']
    
    train_data.close()
    test_data.close()
    
    logger.info(f"原始数据合并完成:")
    logger.info(f"训练集: {len(X_train_all)} 样本，形状: {X_train_all.shape}")
    logger.info(f"测试集: {len(X_test_all)} 样本，形状: {X_test_all.shape}")
    
    # 限制每个患者的样本数量
    if len(X_train_all) > max_samples_per_patient * len(train_patients):
        logger.info(f"训练集样本数超过限制，进行随机采样...")
        indices = np.random.choice(len(X_train_all), max_samples_per_patient * len(train_patients), replace=False)
        X_train_all = X_train_all[indices]
        y_train_all = y_train_all[indices]
    
    if len(X_test_all) > max_samples_per_patient * len(test_patients_normalized):
        logger.info(f"测试集样本数超过限制，进行随机采样...")
        indices = np.random.choice(len(X_test_all), max_samples_per_patient * len(test_patients_normalized), replace=False)
        X_test_all = X_test_all[indices]
        y_test_all = y_test_all[indices]
    
    # 平衡训练集
    logger.info(f"\n平衡训练集（目标比例: {target_ratio}）...")
    X_train_balanced, y_train_balanced = balance_data_chunked(
        [X_train_all], [y_train_all], target_ratio=target_ratio
    )
    
    # 平衡测试集
    logger.info(f"\n平衡测试集（目标比例: {target_ratio}）...")
    X_test_balanced, y_test_balanced = balance_data_chunked(
        [X_test_all], [y_test_all], target_ratio=target_ratio
    )
    
    # 确保通道数一致（都使用23通道）
    if X_train_balanced.shape[1] != 23:
        logger.warning(f"训练数据通道数({X_train_balanced.shape[1]})不是23，进行调整")
        if X_train_balanced.shape[1] > 23:
            X_train_balanced = X_train_balanced[:, :23, :]
        else:
            # 如果通道数不足，用0填充
            padding = np.zeros((X_train_balanced.shape[0], 23 - X_train_balanced.shape[1], X_train_balanced.shape[2]), dtype=np.float32)
            X_train_balanced = np.concatenate([X_train_balanced, padding], axis=1)
    
    if X_test_balanced.shape[1] != 23:
        logger.warning(f"测试数据通道数({X_test_balanced.shape[1]})不是23，进行调整")
        if X_test_balanced.shape[1] > 23:
            X_test_balanced = X_test_balanced[:, :23, :]
        else:
            # 如果通道数不足，用0填充
            padding = np.zeros((X_test_balanced.shape[0], 23 - X_test_balanced.shape[1], X_test_balanced.shape[2]), dtype=np.float32)
            X_test_balanced = np.concatenate([X_test_balanced, padding], axis=1)
    
    # 保存平衡后的数据
    logger.info("\n保存平衡后的数据...")
    np.save('./data/X_train_balanced.npy', X_train_balanced)
    np.save('./data/y_train_balanced.npy', y_train_balanced)
    np.save('./data/X_test_balanced.npy', X_test_balanced)
    np.save('./data/y_test_balanced.npy', y_test_balanced)
    
    # 清理临时文件
    try:
        os.remove('./data/train_data_balanced.npz')
        os.remove('./data/test_data_balanced.npz')
    except:
        pass
    
    logger.info(f"\n平衡后最终数据分布:")
    logger.info(f"训练集: {Counter(y_train_balanced)} - 正类比例: {np.sum(y_train_balanced == 1) / len(y_train_balanced):.4f}")
    logger.info(f"测试集: {Counter(y_test_balanced)} - 正类比例: {np.sum(y_test_balanced == 1) / len(y_test_balanced):.4f}")
    
    logger.info("\n✅ 内存高效的平衡数据集准备完成！")
    return True

def main():
    """主函数"""
    logger.info("=== CHB-MIT 内存高效平衡数据集准备工具 ===")
    logger.info("解决类别不平衡问题，支持大文件处理")
    logger.info("新增中间文件生成功能，优化内存使用")
    
    # 检查是否已有临时文件，如果有则直接合并
    temp_dir = "temp"
    if os.path.exists(temp_dir) and os.listdir(temp_dir):
        logger.info("发现已有临时文件，直接进行合并...")
        
        # 创建输出目录
        output_dir = "./data"
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有临时文件
        import glob
        all_temp_files = glob.glob(os.path.join(temp_dir, "temp_*.npz"))
        
        if not all_temp_files:
            logger.error("错误: 没有找到临时文件")
            return
        
        logger.info(f"找到 {len(all_temp_files)} 个临时文件，开始合并...")
        
        # 分离训练集和测试集文件（基于文件名中的患者ID）
        train_temp_files = []
        test_temp_files = []
        
        for temp_file in all_temp_files:
            filename = os.path.basename(temp_file)
            # 从文件名中提取患者ID
            if 'chb05' in filename:
                test_temp_files.append(temp_file)
            else:
                train_temp_files.append(temp_file)
        
        if not train_temp_files:
            logger.error("错误: 没有找到训练集临时文件")
            return
            
        if not test_temp_files:
            logger.error("错误: 没有找到测试集临时文件")
            return
        
        # 合并训练集数据
        logger.info("\n合并训练集数据...")
        train_merge_success = merge_batch_files(train_temp_files, os.path.join(output_dir, "train_data_balanced.npz"))
        if not train_merge_success:
            logger.error("错误: 训练集数据合并失败")
            return
        
        # 合并测试集数据
        logger.info("合并测试集数据...")
        test_merge_success = merge_batch_files(test_temp_files, os.path.join(output_dir, "test_data_balanced.npz"))
        if not test_merge_success:
            logger.error("错误: 测试集数据合并失败")
            return
        
        # 加载最终数据
        train_data = np.load(os.path.join(output_dir, "train_data_balanced.npz"))
        test_data = np.load(os.path.join(output_dir, "test_data_balanced.npz"))
        
        X_train_all = train_data['data']
        y_train_all = train_data['labels']
        X_test_all = test_data['data']
        y_test_all = test_data['labels']
        
        train_data.close()
        test_data.close()
        
        logger.info(f"\n数据预处理完成:")
        logger.info(f"训练集: {len(X_train_all)} 样本，形状: {X_train_all.shape}")
        logger.info(f"测试集: {len(X_test_all)} 样本，形状: {X_test_all.shape}")
        
        # 进行数据平衡处理
        target_ratio = 0.3
        logger.info(f"\n平衡训练集（目标比例: {target_ratio}）...")
        X_train_balanced, y_train_balanced = balance_data_chunked(
            [X_train_all], [y_train_all], target_ratio=target_ratio
        )
        
        logger.info(f"\n平衡测试集（目标比例: {target_ratio}）...")
        X_test_balanced, y_test_balanced = balance_data_chunked(
            [X_test_all], [y_test_all], target_ratio=target_ratio
        )
        
        # 确保通道数一致（都使用23通道）
        if X_train_balanced.shape[1] != 23:
            logger.warning(f"训练数据通道数({X_train_balanced.shape[1]})不是23，进行调整")
            if X_train_balanced.shape[1] > 23:
                X_train_balanced = X_train_balanced[:, :23, :]
            else:
                # 如果通道数不足，用0填充
                padding = np.zeros((X_train_balanced.shape[0], 23 - X_train_balanced.shape[1], X_train_balanced.shape[2]), dtype=np.float32)
                X_train_balanced = np.concatenate([X_train_balanced, padding], axis=1)
        
        if X_test_balanced.shape[1] != 23:
            logger.warning(f"测试数据通道数({X_test_balanced.shape[1]})不是23，进行调整")
            if X_test_balanced.shape[1] > 23:
                X_test_balanced = X_test_balanced[:, :23, :]
            else:
                # 如果通道数不足，用0填充
                padding = np.zeros((X_test_balanced.shape[0], 23 - X_test_balanced.shape[1], X_test_balanced.shape[2]), dtype=np.float32)
                X_test_balanced = np.concatenate([X_test_balanced, padding], axis=1)
        
        # 保存平衡后的数据
        logger.info("\n保存平衡后的数据...")
        np.save('./data/X_train_balanced.npy', X_train_balanced)
        np.save('./data/y_train_balanced.npy', y_train_balanced)
        np.save('./data/X_test_balanced.npy', X_test_balanced)
        np.save('./data/y_test_balanced.npy', y_test_balanced)
        
        # 清理临时文件
        try:
            os.remove(os.path.join(output_dir, "train_data_balanced.npz"))
            os.remove(os.path.join(output_dir, "test_data_balanced.npz"))
        except:
            pass
        
        logger.info(f"\n平衡后最终数据分布:")
        logger.info(f"训练集: {Counter(y_train_balanced)} - 正类比例: {np.sum(y_train_balanced == 1) / len(y_train_balanced):.4f}")
        logger.info(f"测试集: {Counter(y_test_balanced)} - 正类比例: {np.sum(y_test_balanced == 1) / len(y_test_balanced):.4f}")
        
        logger.info("\n✅ 基于中间文件的平衡数据集准备完成！")
        logger.info("现在可以使用以下文件进行训练:")
        logger.info("- X_train_balanced.npy")
        logger.info("- y_train_balanced.npy")
        logger.info("- X_test_balanced.npy")
        logger.info("- y_test_balanced.npy")
        
    else:
        logger.info("没有找到临时文件，重新生成...")
        # 参数设置
        data_dir = "./database/physionet.org/files/chbmit/1.0.0/"
        test_patients = ['chb05']  # 用作测试集的患者
        window_size = 2.0  # 2秒窗口
        target_ratio = 0.3  # 目标正类比例30%
        max_samples_per_patient = 2000  # 每个患者最大样本数
        batch_size = 1000  # 批处理大小
        
        # 准备平衡数据
        success = prepare_balanced_data_memory_efficient(
            data_dir=data_dir,
            test_patients=test_patients,
            window_size=window_size,
            target_ratio=target_ratio,
            max_samples_per_patient=max_samples_per_patient,
            batch_size=batch_size
        )
        
        if success:
            logger.info("\n🎉 内存高效的平衡数据集准备成功！")
            logger.info("现在可以使用以下文件进行训练:")
            logger.info("- X_train_balanced.npy")
            logger.info("- y_train_balanced.npy")
            logger.info("- X_test_balanced.npy")
            logger.info("- y_test_balanced.npy")
        else:
            logger.error("\n❌ 数据集准备失败！")

if __name__ == "__main__":
    main()