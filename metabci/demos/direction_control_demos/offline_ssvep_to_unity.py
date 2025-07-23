import numpy as np
import mne
from mne.filter import resample
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from metabci.brainda.algorithms.decomposition import FBDSP
from metabci.brainda.utils import upper_ch_names
from metabci.brainda.paradigms import SSVEP as SSVEPParadigm
from metabci.brainda.datasets.tsinghua import Wang2016
from sklearn.base import BaseEstimator, ClassifierMixin

# 工具函数

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

def read_wang2016_data(subjects, stim_interval, pick_chs):
    """使用Wang2016数据集读取训练数据"""
    dataset = Wang2016()
    paradigm_intervals = [(stim_interval[0], stim_interval[1])]
    paradigm = SSVEPParadigm(
        channels=dataset.channels,
        events=dataset.events,
        intervals=paradigm_intervals,
        srate=250)
    def raw_hook(raw, caches):
        raw.filter(5, 55, l_trans_bandwidth=2, h_trans_bandwidth=5,
                   phase='zero-double')
        caches['raw_stage'] = caches.get('raw_stage', -1) + 1
        return raw, caches
    paradigm.register_raw_hook(raw_hook)
    X, y, meta = paradigm.get_data(
        dataset,
        subjects=subjects,
        return_concat=True,
        n_jobs=-1,
        verbose=False)
    ch_names = [ch.upper() for ch in pick_chs]
    ch_ind = [dataset.channels.index(ch) for ch in ch_names if ch in dataset.channels]
    X = X[:, ch_ind, :]
    unique_labels = np.unique(y)
    y = label_encoder(y, unique_labels)
    return X, y, ch_ind

# 下面是原始的train_model和model_predict实现

def train_model(X, y, srate=250):
    y = np.reshape(y, (-1))
    X = resample(X, up=256, down=srate)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    wp = [
        [6, 88], [14, 88], [22, 88], [30, 88], [38, 88]
    ]
    ws = [
        [4, 90], [12, 90], [20, 90], [28, 90], [36, 90]
    ]
    filterweights = np.arange(1, 6)**(-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 256)
    freqs = Wang2016._FREQS
    Yf = generate_cca_references(freqs, srate=256, T=0.5, n_harmonics=5)
    model = FBDSP(filterbank, filterweights=np.array(filterweights))
    model = model.fit(X, y, Yf=Yf)
    return model

def model_predict(X, srate=250, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=256, down=srate)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    p_labels = model.predict(X)
    return p_labels

def offline_validation(X, y, srate=250):
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)
    kfold_accs = []
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])
        model = train_model(X_train, y_train, srate=srate)
        p_labels = model_predict(X_test, srate=srate, model=model)
        kfold_accs.append(np.mean(p_labels == y_test))
    return np.mean(kfold_accs)

if __name__ == "__main__":
    # 配置参数
    srate = 1000
    stim_interval = [0.5, 5.5]  # Wang2016使用0.5-5.5秒
    stim_labels = list(range(1, 41))  # Wang2016有40个目标
    subjects = list(range(1, 3))  # 使用Wang2016数据集的前2个被试
    pick_chs = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']

    # 读取数据
    X, y, ch_ind = read_wang2016_data(
        subjects=subjects,
        stim_interval=stim_interval,
        pick_chs=pick_chs)
    print("离线数据读取完成，样本数：", X.shape[0])

    # 训练模型
    model = train_model(X, y, srate=srate)
    print("模型训练完成")

    # 预测（这里直接用训练集做演示）
    pred_labels = model_predict(X, srate=srate, model=model)
    print("预测完成，预测标签：", pred_labels)

    # 保存结果到txt，便于Unity端读取
    np.savetxt("ssvep_pred_labels.txt", pred_labels, fmt="%d")
    print("预测结果已保存到 ssvep_pred_labels.txt") 