import mne
import numpy as np
import os
import re
from typing import List, Tuple, Dict
from pathlib import Path


def parse_summary_file(summary_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    解析summary文件，提取每个文件的发作期时间段
    
    Args:
        summary_path: summary文件路径
        
    Returns:
        字典，key为文件名，value为发作期时间段列表[(start, end), ...]
    """
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
    """
    对EEG数据进行带通滤波
    
    Args:
        raw: MNE Raw对象
        l_freq: 低频截止频率，默认0.5Hz
        h_freq: 高频截止频率，默认40Hz
        
    Returns:
        滤波后的Raw对象
    """
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', skip_by_annotation='edge')
    return raw_filtered


def is_seizure_period(t_start: float, t_end: float, seizure_periods: List[Tuple[float, float]]) -> bool:
    """
    判断时间段是否与发作期重叠
    
    Args:
        t_start: 窗口开始时间（秒）
        t_end: 窗口结束时间（秒）
        seizure_periods: 发作期时间段列表
        
    Returns:
        如果窗口与任何发作期重叠，返回True
    """
    for seizure_start, seizure_end in seizure_periods:
        # 检查窗口是否与发作期重叠
        if not (t_end <= seizure_start or t_start >= seizure_end):
            return True
    return False


def window_seizure_data(raw, seizure_periods: List[Tuple[float, float]], 
                       window_size: float, step_size: float = 1.0):
    """
    对发作期数据进行重叠窗口切片
    
    Args:
        raw: MNE Raw对象
        seizure_periods: 发作期时间段列表
        window_size: 窗口大小（秒）
        step_size: 移动步长（秒），默认1秒
        
    Returns:
        窗口数据列表，每个元素为(n_channels, n_samples)的numpy数组
    """
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


def window_non_seizure_data(raw, seizure_periods: List[Tuple[float, float]], 
                            window_size: float):
    """
    对非发作期数据进行非重叠窗口切片
    
    Args:
        raw: MNE Raw对象
        seizure_periods: 发作期时间段列表
        window_size: 窗口大小（秒）
        
    Returns:
        窗口数据列表，每个元素为(n_channels, n_samples)的numpy数组
    """
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
    
    # 对每个非发作期时间段进行非重叠窗口切片
    for range_start, range_end in non_seizure_ranges:
        current_start = range_start
        while current_start + window_size <= range_end:
            start_idx = int(current_start * sfreq)
            end_idx = start_idx + window_samples
            
            if end_idx <= raw.n_times:
                window_data = raw[:, start_idx:end_idx][0]  # 获取数据数组
                windows.append(window_data)
            
            current_start += window_size  # 非重叠，步长等于窗口大小
    
    return windows


def process_eeg_file(edf_path: str, summary_path: str, window_size: float = 2.0, 
                     l_freq: float = 0.5, h_freq: float = 40.0):
    """
    处理单个EEG文件：滤波、窗口切片
    
    Args:
        edf_path: EDF文件路径
        summary_path: summary文件路径
        window_size: 窗口大小（秒），默认2秒
        l_freq: 低频截止频率，默认0.5Hz
        h_freq: 高频截止频率，默认40Hz
        
    Returns:
        (seizure_windows, non_seizure_windows, labels)
        seizure_windows: 发作期窗口列表
        non_seizure_windows: 非发作期窗口列表
        labels: 标签列表（1表示发作期，0表示非发作期）
    """
    # 读取EDF文件
    print(f"正在读取文件: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    
    # 带通滤波
    print(f"正在应用带通滤波 ({l_freq}-{h_freq} Hz)...")
    raw_filtered = apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)
    
    # 解析summary文件获取发作期信息
    seizure_info = parse_summary_file(summary_path)
    filename = os.path.basename(edf_path)
    
    if filename not in seizure_info:
        print(f"警告: 在summary文件中未找到 {filename} 的信息")
        seizure_periods = []
    else:
        seizure_periods = seizure_info[filename]
        print(f"找到 {len(seizure_periods)} 个发作期")
    
    # 窗口切片
    print(f"正在对发作期数据进行重叠窗口切片（窗口大小: {window_size}秒，步长: 1秒）...")
    seizure_windows = window_seizure_data(raw_filtered, seizure_periods, 
                                         window_size=window_size, step_size=1.0)
    
    print(f"正在对非发作期数据进行非重叠窗口切片（窗口大小: {window_size}秒）...")
    non_seizure_windows = window_non_seizure_data(raw_filtered, seizure_periods, 
                                                  window_size=window_size)
    
    # 创建标签
    labels = [1] * len(seizure_windows) + [0] * len(non_seizure_windows)
    
    # 合并窗口
    all_windows = seizure_windows + non_seizure_windows
    
    print(f"处理完成: 发作期窗口 {len(seizure_windows)} 个, 非发作期窗口 {len(non_seizure_windows)} 个")
    
    return seizure_windows, non_seizure_windows, all_windows, labels


def process_all_files(data_dir: str, summary_path: str, window_size: float = 2.0,
                     l_freq: float = 0.5, h_freq: float = 40.0):
    """
    处理目录下所有EDF文件
    
    Args:
        data_dir: 包含EDF文件的目录
        summary_path: summary文件路径
        window_size: 窗口大小（秒），默认2秒
        l_freq: 低频截止频率，默认0.5Hz
        h_freq: 高频截止频率，默认40Hz
        
    Returns:
        (all_windows, all_labels)
        all_windows: 所有窗口数据列表
        all_labels: 所有标签列表
    """
    all_windows = []
    all_labels = []
    
    # 获取所有EDF文件
    edf_files = list(Path(data_dir).glob('*.edf'))
    
    print(f"找到 {len(edf_files)} 个EDF文件")
    
    for edf_file in edf_files:
        try:
            seizure_windows, non_seizure_windows, windows, labels = process_eeg_file(
                str(edf_file), summary_path, window_size, l_freq, h_freq
            )
            all_windows.extend(windows)
            all_labels.extend(labels)
        except Exception as e:
            print(f"处理文件 {edf_file} 时出错: {e}")
            continue
    
    print(f"\n总共处理了 {len(all_windows)} 个窗口")
    print(f"发作期窗口: {sum(all_labels)} 个")
    print(f"非发作期窗口: {len(all_labels) - sum(all_labels)} 个")
    
    return all_windows, all_labels


if __name__ == "__main__":
    # 示例用法
    data_dir = r'database\physionet.org\files\chbmit\1.0.0\chb01'
    summary_path = r'database\physionet.org\files\chbmit\1.0.0\chb01\chb01-summary.txt'
    
    # 处理单个文件示例
    edf_path = os.path.join(data_dir, 'chb01_03.edf')
    if os.path.exists(edf_path):
        seizure_windows, non_seizure_windows, all_windows, labels = process_eeg_file(
            edf_path, summary_path, window_size=2.0
        )
        print(f"\n窗口数据形状示例: {all_windows[0].shape if all_windows else '无数据'}")
    
    # 处理所有文件示例（可选）
    # all_windows, all_labels = process_all_files(data_dir, summary_path, window_size=2.0)

