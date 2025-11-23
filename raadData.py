import mne
import matplotlib
matplotlib.use('Qt5Agg')  # 设置 matplotlib 后端，确保图像能够显示

# 加载某个病人的 EEG 文件
raw = mne.io.read_raw_edf(r'physionet.org\files\chbmit\1.0.0\chb01\chb01_01.edf', preload=True)

# 查看数据基本信息
print(raw.info)

# 绘制 EEG 数据，block=True 保持窗口打开
raw.plot(duration=10, n_channels=30, scalings='auto', block=True)
