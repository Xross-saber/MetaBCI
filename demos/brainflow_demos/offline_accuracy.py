# -*- coding: utf-8 -*-
"""
SSVEP offline analysis with logging.
"""
import numpy as np
from metabci.brainda.algorithms.decomposition import FBDSP
from sklearn.base import BaseEstimator, ClassifierMixin
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from mne.filter import resample
import warnings
from metabci.brainda.datasets.tsinghua import Wang2016
from metabci.brainflow.logger import get_logger
from scipy import signal

# 配置日志
logger = get_logger("offline_accuracy")

warnings.filterwarnings('ignore')

# 对raw操作,例如滤波
def raw_hook(raw, caches):
    raw.filter(5, 55, l_trans_bandwidth=2, h_trans_bandwidth=5,
               phase='zero-double')
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches

# 按照0,1,2,...重新排列标签
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

# 训练模型
def train_model(X, y, srate=250):
    logger.info("train_model开始运行")
    y = np.reshape(y, (-1))
    logger.info(f"输入 X 形状: {X.shape}")
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    # 滤波器组设置，匹配Wang2016频率范围
    wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]]
    ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]]
    filterweights = np.arange(1, 6) ** (-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 250)

    # 使用Wang2016的40个频率
    freqs = Wang2016._FREQS
    Yf = generate_cca_references(freqs, srate=250, T=0.5, n_harmonics=5)

    # 初始化 FBDSP 模型，禁用权重以避免广播问题
    model = FBDSP(
        filterbank,
        n_components=5,
        transform_method="corr",
        filterweights=None,
        n_jobs=-1
    )

    # 训练模型
    model = model.fit(X, y, Yf=Yf)
    logger.info("train_model结束运行")
    return model

# 预测标签
def model_predict(X, srate=250, model=None):
    logger.info("model_predict开始运行")
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    logger.info(f"X 形状传入 predict: {X.shape}")
    p_labels = model.predict(X)
    logger.info("model_predict结束运行")
    return p_labels

# 计算离线正确率
def offline_validation(X, y, srate=250):
    logger.info("offline_validation开始运行")
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
        logger.info("offline_validation结束运行")
    return np.mean(kfold_accs)

if __name__ == '__main__':
    # 初始化参数
    srate = 250
    stim_interval = [(0.5, 5.5)]
    subjects = list(range(1, 3))
    paradigm = 'ssvep'

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
    logger.info(f"X 形状: {X.shape}, y 形状: {y.shape}")

    # 计算离线正确率
    acc = offline_validation(X, y, srate=srate)
    logger.info("Current Model accuracy: {:.2f}".format(acc))
    print("Current Model accuracy:{:.2f}".format(acc))