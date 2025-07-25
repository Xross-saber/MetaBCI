import numpy as np
import pickle

def get_template_list(frequency_set, data_len, sample_rate=250, set_phase=True, multi_times=5, qr=True):
    if set_phase:
        phase_set = [i % 4 * 0.5 for i in range(len(frequency_set))]
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


# frequency_set = np.arange(8, 16, 0.2)
# data_list = get_template_list(frequency_set=frequency_set, data_len=2000, sample_rate=1000, set_phase=True, multi_times=4, qr=False)
# data = data_list[0]
# print(data)

with open('1.pkl', 'rb') as f:
    loaded_matrix1 = pickle.load(f)
with open('2.pkl', 'rb') as f:
    loaded_matrix2 = pickle.load(f)
print(loaded_matrix1.shape)