import numpy as np
import os
import pickle
import scipy
from numba import prange, njit
from scipy import signal
from scipy.signal import lfilter, lfilter_zi, resample_poly
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def cca(data, template):
    data = data.T
    # qr分解,data:length*channel
    q_temp = np.linalg.qr(data)[0]

    template = template.T
    q_cs = np.linalg.qr(template)[0]
    data_svd = np.dot(q_temp.T, q_cs)
    [u, s, v] = np.linalg.svd(data_svd)
    weight = [1.25, 0.67, 0.5]
    rho = sum(s[:3] * weight[:len(s[:3])])
    return rho

def get_template_list(frequency_set, data_len, sample_rate=1000, set_phase=True, multi_times=5, qr=True):
    if set_phase:
        phase_set = [i * 0.35 % 2 for i in range(len(frequency_set))]
    else:
        phase_set = [0] * len(frequency_set)

    n = np.arange(0, data_len) / sample_rate
    if qr:
        target_list = np.zeros((len(frequency_set), data_len, multi_times * 2))
    else:
        target_list = np.zeros((len(frequency_set), multi_times * 2, data_len))
    raw = np.zeros((multi_times * 2, data_len))
    for i in range(len(frequency_set)):
        for j in range(multi_times):
            raw[j * 2] = np.cos((j + 1) * frequency_set[i] * np.pi * 2 * n + phase_set[i] * np.pi)
            raw[j * 2 + 1] = np.sin((j + 1) * frequency_set[i] * np.pi * 2 * n + phase_set[i] * np.pi)
        if qr:
            target_list[i] = np.linalg.qr(raw.T)[0]
        else:
            target_list[i] = raw
    return target_list

class CCA(object):
    def __init__(self, frequency_set, data_len=None, template_list=None, digit_num = None, target_set = None):
        self.frequency_set = frequency_set
        self.digit_num = digit_num
        self.target_set = target_set
        if data_len:
            self.target_list = get_template_list(np.asarray(self.frequency_set), data_len)
        elif template_list is not None:
            self.target_list = template_list
        else:
            self.target_list = []

    def fit(self, data):
        if len(self.target_list) == 0:
            self.target_list = get_template_list(np.asarray(self.frequency_set), data.shape[-1])
        p = []
        for template in self.target_list:
            rho = cca(data, np.asarray(template)[:, :data.shape[1]].T)
            p.append(rho)
        result = p.index(max(p))
        result = result + 1
        return result, p
        # return p

    def get_cor_list(self, digit_result_list):
        cor_list = np.zeros(len(self.target_set))
        for i in range(len(self.target_set)):
            for j in range(self.digit_num):
                cor_list[i] += digit_result_list[j][int(self.target_set[i][j])]
        return list(cor_list)

if __name__ == '__main__':
    freqs = np.arange(8, 17, 0.5)
    method = CCA(frequency_set=freqs, data_len = 4000)

    for i in range(2,18):
        trial_num = i
        with open('data/{}.pkl'.format(trial_num), 'rb') as f:
            signal = pickle.load(f)

        data = signal[16]
        data = data[500:4500]
        data = np.array([data] * 9)
        result = method.fit(data)
        print(result)
