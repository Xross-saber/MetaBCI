# -*- coding: utf-8 -*-
"""
SSVEP 离线分析
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from metabci.brainda.algorithms.decomposition import FBDSP
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.feature_analysis.freq_analysis import FrequencyAnalysis
from metabci.brainda.algorithms.feature_analysis.time_analysis import TimeAnalysis
from metabci.brainda.algorithms.feature_analysis.time_freq_analysis import TimeFrequencyAnalysis
from metabci.brainda.algorithms.utils.model_selection import EnhancedLeaveOneGroupOut
from metabci.brainda.datasets.tsinghua import Wang2016
from metabci.brainda.paradigms import SSVEP
from mne.filter import resample
from scipy import signal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import warnings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('offline_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# 对原始数据应用带通滤波
def raw_hook(raw, caches):
    logger.info("应用 4-60 Hz 带通滤波")
    raw.filter(4, 60, l_trans_bandwidth=2, h_trans_bandwidth=5,
               phase='zero-double')  # 调整为 4-60 Hz 保留 8-15.8 Hz
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches


# 重新排列标签为 0,1,2,...
def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


class MaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        X = X.reshape((-1, X.shape[-1]))
        y = np.argmax(X, axis=-1)
        return y


# 训练 FBDSP 模型
def train_model(X, y, srate=1000):
    logger.info("开始训练模型")
    y = np.reshape(y, (-1))
    X = resample(X, up=256, down=srate)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]]
    ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]]
    filterweights = np.arange(1, 6) ** (-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 256)

    freqs = np.arange(8, 16, 0.2)
    Yf = generate_cca_references(freqs, srate=256, T=0.5, n_harmonics=5)

    model = FBDSP(
        filterbank,
        n_components=1,
        transform_method="corr",
        filterweights=np.array(filterweights),
        n_jobs=-1
    )

    logger.info("开始优化滤波器权重")
    new_weights = model.optimize_weights(X, y, n_splits=5)
    logger.info(f"优化后的滤波器权重: {new_weights}")

    model = FBDSP(
        filterbank,
        n_components=1,
        transform_method="corr",
        filterweights=new_weights,
        n_jobs=-1
    )

    model = model.fit(X, y, Yf=Yf)
    logger.info("模型训练完成")
    return model


# 预测标签
def model_predict(X, srate=250, model=None):
    logger.info("开始模型预测")
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=256, down=srate)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    p_labels = model.predict(X)
    logger.info("模型预测完成")
    return p_labels


# 计算离线准确率
def offline_validation(X, y, srate=250):
    logger.info("开始离线验证")
    unique_classes = np.unique(y)
    y = np.reshape(y, (-1))
    kfold_accs = []
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])
        model = train_model(X_train, y_train, srate=srate)
        p_labels = model_predict(X_test, srate=srate, model=model)
        kfold_accs.append(np.mean(p_labels == y_test))
        logger.info("离线验证单折完成")
    logger.info(f"平均准确率: {np.mean(kfold_accs):.2f}")
    return np.mean(kfold_accs)


# 时域分析
def time_feature(X, meta, dataset, event, channel, latency=0):
    logger.info(f"开始时域分析，事件: {event}，通道: {channel}")
    Feature_R = TimeAnalysis(X, meta, dataset, event=event, latency=latency,
                             channel=channel)
    plt.figure(1)
    data_mean = Feature_R.stacking_average(np.squeeze(
        Feature_R.data[:, Feature_R.chan_ID, :]), _axis=[0])
    logger.info(f"均值数据形状: {data_mean.shape}")
    ax = plt.subplot(2, 1, 1)
    sample_num = int(Feature_R.fs * Feature_R.data_length)
    loc, amp, ax = Feature_R.plot_single_trial(data_mean,
                                               sample_num=sample_num,
                                               axes=ax,
                                               amp_mark='peak',
                                               time_start=0,
                                               time_end=sample_num - 1)
    plt.title("(a)", x=0.03, y=0.86)
    ax = plt.subplot(2, 1, 2)
    ax = Feature_R.plot_multi_trials(
        np.squeeze(Feature_R.data[:, Feature_R.chan_ID, :]),
        sample_num=sample_num, axes=ax)
    plt.title("(b)", x=0.03, y=0.86)
    fig2 = plt.figure(2)
    data_map = Feature_R.stacking_average(Feature_R.data, _axis=0)
    Feature_R.plot_topomap(data_map, loc, fig=fig2,
                           channels=Feature_R.All_channel, axes=ax)
    plt.show()
    logger.info("时域分析完成")


# 频域分析
def frequency_feature(X, chan_names, event, SNRchannels, plot_ch, srate=250):
    """
    执行频域分析并绘制指定通道的功率谱密度（PSD）图。

    参数
    ----------
    X : np.ndarray
        EEG 数据，形状 (n_trials, n_channels, n_samples)
    chan_names : list
        通道名称列表
    event : str
        目标事件（例如 '8.0' 表示 8 Hz）
    SNRchannels : str
        用于 SNR 计算的通道（例如 'PO5'）
    plot_ch : int
        要绘制的通道索引
    srate : int
        采样率（默认 250 Hz，降采样后调整为 256 Hz）

    返回
    -------
    SNRchannels : str
        选择的通道名称
    snr : float
        目标频率的信噪比
    """
    logger.info(f"开始频域分析，事件: {event}，通道: {SNRchannels}")
    channellist = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']

    if not all(ch in channellist for ch in chan_names):
        logger.error(f"无效通道名称: {chan_names}，必须在 {channellist} 中")
        raise ValueError(f"无效通道名称: {chan_names}")
    if SNRchannels not in chan_names:
        logger.error(f"SNR 通道 {SNRchannels} 不在 chan_names 中: {chan_names}")
        raise ValueError(f"SNR 通道 {SNRchannels} 不在 chan_names 中")

    chan_nums = [channellist.index(ch) for ch in chan_names]
    X = X[:, chan_nums, :]
    logger.info(f"输入 X 形状: {X.shape}")
    logger.info(f"输入 X 是否包含 NaN: {np.any(np.isnan(X))}")
    logger.info(f"输入 X 均值: {np.mean(X):.2e}，标准差: {np.std(X):.2e}")

    dataset = Wang2016()
    paradigm = SSVEP(
        channels=dataset.channels,
        events=dataset.events,
        intervals=[(0, 2.0)],
        srate=srate)
    meta = paradigm.get_data(dataset, subjects=list(range(1, 3)), return_concat=True, verbose=False)[2]

    try:
        target_freq = float(event)
        event_str = f"{target_freq:.1f}"  # 转换为 '8.0' 格式
        y = meta['event'].values
        target_indices = np.where(y == event_str)[0]
        if len(target_indices) == 0:
            logger.error(f"未找到事件 {event_str} 的试次，可用事件: {np.unique(y)}")
            raise ValueError(f"未找到事件 {event_str} 的试次")
        X = X[target_indices, :, :]
        logger.info(f"筛选后 X 形状，事件 {event_str}: {X.shape}")
    except ValueError as e:
        logger.error(f"事件筛选失败: {e}")
        raise

    Feature_R = FrequencyAnalysis(X, meta, event_str, srate=256)
    mean_data = Feature_R.stacking_average(data=Feature_R.data, _axis=0)
    logger.info(f"均值数据形状: {mean_data.shape}")
    logger.info(f"均值数据是否包含 NaN: {np.any(np.isnan(mean_data))}")
    logger.info(f"均值数据均值: {np.mean(mean_data):.2e}，标准差: {np.std(mean_data):.2e}")
    if mean_data.size == 0:
        logger.error("均值数据在 stacking_average 后为空")
        raise ValueError("均值数据在 stacking_average 后为空")

    mean_data = np.nan_to_num(mean_data, nan=0.0)
    SNR_chan_idx = chan_names.index(SNRchannels)
    logger.info(f"选择的通道: {SNRchannels}，索引: {SNR_chan_idx}")

    f, den = signal.periodogram(mean_data[SNR_chan_idx], fs=256, window="boxcar", scaling="spectrum")
    logger.info(f"PSD 频率（前 10 个）: {f[:10]}")
    logger.info(f"PSD 值（前 10 个）: {den[:10]}")
    if np.max(den) == 0:
        logger.warning("PSD 值全为零，请检查输入数据或预处理")

    target_freq_idx = np.argmin(np.abs(f - target_freq))
    noise_window = 2
    noise_indices = \
    np.where((f >= target_freq - noise_window) & (f <= target_freq + noise_window) & (np.abs(f - target_freq) > 0.1))[0]
    snr = den[target_freq_idx] / np.mean(den[noise_indices]) if len(noise_indices) > 0 else den[target_freq_idx]
    logger.info(f"{target_freq} Hz 的 SNR: {snr:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(f, den, label=f'通道 {SNRchannels}')
    for freq in [target_freq, target_freq * 2, target_freq * 3]:
        freq_idx = np.argmin(np.abs(f - freq))
        plt.text(freq, den[freq_idx], f'{den[freq_idx]:.2e}', fontsize=15, ha='center')
    plt.title(f'{SNRchannels} 的 PSD (事件: {event} Hz)')
    plt.xlabel('频率 [Hz]')
    plt.ylabel('PSD [V^2/Hz]')
    plt.xlim([0, 60])
    plt.ylim([0, max(np.max(den) * 1.2, 1e-6)])
    plt.grid(True)
    plt.legend()
    plt.show()
    logger.info("PSD 计算和绘图完成")

    return SNRchannels, snr


# 时频域分析
def time_frequency_feature(X, y, chan_names, srate=250):
    logger.info("开始时频域分析")
    channellist = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    chan_nums = [channellist.index(ch) for ch in chan_names]
    X = X[:, chan_nums, :]
    index_8hz = np.where(y == 0)
    data_8hz = np.squeeze(X[index_8hz, :, :])
    mean_data_8hz = np.mean(data_8hz, axis=0)
    fs = srate

    Feature_R = TimeFrequencyAnalysis(fs)
    nfft = mean_data_8hz.shape[1]
    f, t, Zxx = Feature_R.fun_stft(mean_data_8hz, nperseg=256, axis=1, nfft=nfft)
    Zxx_Pz = Zxx[-4, :, :]
    plt.pcolormesh(t, f, np.abs(Zxx_Pz))
    plt.ylim(0, 25)
    plt.title('STFT 幅度')
    plt.ylabel('频率 [Hz]')
    plt.xlabel('时间 [秒]')
    plt.colorbar()
    plt.show()
    logger.info("STFT 分析完成")

    mean_Pz_data_8hz = mean_data_8hz[-4, :]
    N = mean_Pz_data_8hz.shape[0]
    t_index = np.linspace(0, N / fs, num=N, endpoint=False)
    omega = 2
    sigma = 1
    data_test = np.reshape(mean_Pz_data_8hz, newshape=(1, mean_Pz_data_8hz.shape[0]))
    P, S = Feature_R.func_morlet_wavelet(data_test, f, omega, sigma)
    f_lim = np.array([min(f[np.where(f > 0)]), 30])
    f_idx = np.array(np.where((f <= f_lim[1]) & (f >= f_lim[0])))[0]
    t_lim = np.array([0, 1])
    t_idx = np.array(np.where((t_index <= t_lim[1]) & (t_index >= t_lim[0])))[0]
    PP = P[0, f_idx, :]
    plt.pcolor(t_index[t_idx], f[f_idx], PP[:, t_idx])
    plt.xlabel('时间 [秒]')
    plt.ylabel('频率 [Hz]')
    plt.xlim(t_lim)
    plt.ylim(f_lim)
    plt.plot([0, 0], [0, fs / 2], 'w--')
    plt.title('尺度图 (ω = {}, σ = {})'.format(omega, sigma))
    plt.text(t_lim[1] + 0.04, f_lim[1] / 2, '功率 (\muV^2/Hz)', rotation=90,
             verticalalignment='center', horizontalalignment='center')
    plt.colorbar()
    plt.show()
    logger.info("莫雷小波变换完成")

    charray = np.mean(data_8hz, axis=1)
    tarray = charray[0, :]
    N1 = tarray.shape[0]
    analytic_signal, realEnv, imagEnv, angle, envModu = Feature_R.fun_hilbert(tarray)
    time = np.linspace(0, N1 / fs, num=N1, endpoint=False)
    plt.plot(time, realEnv, "k", marker='o', markerfacecolor='white', label="实部")
    plt.plot(time, imagEnv, "b", label="虚部")
    plt.plot(time, angle, "c", linestyle='-', label="相位")
    plt.plot(time, analytic_signal, "grey", label="信号")
    plt.ylabel('相位或幅度')
    plt.xlabel('时间 [秒]')
    plt.legend()
    plt.show()
    logger.info("希尔伯特变换完成")


if __name__ == '__main__':
    srate = 1000
    stim_interval = [(0.14, 1.14)]
    subjects = list(range(1, 3))
    pick_chs = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    dataset = Wang2016()
    paradigm = SSVEP(
        channels=dataset.channels,
        events=dataset.events,
        intervals=stim_interval,
        srate=srate)
    paradigm.register_raw_hook(raw_hook)
    X, y, meta = paradigm.get_data(
        dataset,
        subjects=subjects,
        return_concat=True,
        n_jobs=-1,
        verbose=False)
    y = label_encoder(y, np.unique(y))
    logger.info("数据加载成功")


    frequency_feature(X[..., :int(srate)], pick_chs, '8.0', 'PO5', 5, 1000)
    time_frequency_feature(X[..., :int(srate)], y, pick_chs)