import numpy as np
import os
import glob
import pickle
import time
from pylsl import StreamInfo, StreamOutlet
from metabci.brainda.algorithms.decomposition.cca import CCA
import re
import sys
import threading

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
            template_data = np.asarray(template)[:, :data.shape[1]].T
            rho = cca(data, template_data)
            p.append(rho)
        result = p.index(max(p))
        result = result + 1
        return result, p


def extract_label(filename):
    match = re.match(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f'文件名 {filename} 不含数字标签')

def sort_key(x):
    basename = os.path.splitext(os.path.basename(x))[0]
    return int(re.match(r'(\d+)', basename).group(1))

def read_data_for_cca(data_dir):
    pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')), key=sort_key)
    print(f"总文件数: {len(pkl_files)}")
    Xs, ys = [], []
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"{pkl_file} 加载失败，跳过。原因：{e}")
            continue
        if not (isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 5400 and data.shape[0] > 16):
            print(f"{pkl_file} shape不符，跳过")
            continue
        seg = data[16, 500:4500]  # shape (4000,)
        seg = np.tile(seg, (9, 1))
        Xs.append(seg)
        basename = os.path.splitext(os.path.basename(pkl_file))[0]
        try:
            label = extract_label(basename)
        except Exception as e:
            print(f"{pkl_file} 标签提取失败，跳过。原因：{e}")
            continue
        ys.append(label)
    if not Xs:
        raise RuntimeError('无有效数据可用于训练/预测！')
    Xs = np.stack(Xs)  # (n_trials, 9, 4000)
    ys = np.array(ys)
    return Xs, ys

def check_quit_flag(quit_flag):
    # 监听键盘输入线程
    while True:
        if input().strip().lower() == 'q':
            quit_flag[0] = True
            print('检测到q，准备退出...')
            break

if __name__ == '__main__':
    # 1. 训练CCA模型（用离线数据）
    # train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CCA/data'))
    # print(f'训练集目录: {train_dir}')
    # X_train, y_train = read_data_for_cca(train_dir)
    # freqs = np.arange(8, 8 + len(np.unique(y_train)), 1)
    freqs = np.arange(8, 17, 0.5)
    cca_model = CCA(frequency_set=freqs, data_len=3500)

    # 2. 创建LSL流
    stream_name = "meta_feedback"
    stream_source_id = "meta_online_worker"
    info = StreamInfo(stream_name, 'Markers', 1, 0, 'int32', stream_source_id)
    outlet = StreamOutlet(info)
    print(f"LSL流已创建: {stream_name}")

    # 3. 顺序编号检测逻辑
    # online_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'online_data'))
    online_data_dir = os.path.abspath(os.path.join('D:\MetaBCI-master\MetaBCI-master\metabci\demos/brainflow_demos\ReceiverSystem'))
    print(f"监听目录: {online_data_dir}")
    already_processed = set()
    start_idx = 2
    end_idx = 1000  # 可根据实际最大编号调整
    print("开始监听在线数据... (按q回车退出)")
    quit_flag = [False]
    threading.Thread(target=check_quit_flag, args=(quit_flag,), daemon=True).start()
    try:
        while not quit_flag[0]:
            for idx in range(start_idx, end_idx):
                pkl_path = os.path.join(online_data_dir, f"{idx}.pkl")
                if os.path.exists(pkl_path) and idx not in already_processed:
                    print(f"检测到新数据: {pkl_path}")
                    with open(pkl_path, 'rb') as f:
                        signal = pickle.load(f)
                    if isinstance(signal, np.ndarray) and signal.ndim == 2 and signal.shape[0] > 16 and signal.shape[1] >= 4000:
                        data = signal[16]
                        data = data[500:4000]
                        data = np.array([data] * 9)
                        result = cca_model.fit(data)
                        # pred_label, _ = cca_model.fit(data)
                        pred_label = result[0]
                        print(f"预测标签: {pred_label}")
                        outlet.push_sample([pred_label])
                        print(f"已发送到LSL: {pred_label}")
                    else:
                        print("数据格式不符，跳过。")
                    already_processed.add(idx)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("在线分析已停止。")
    print("程序已退出。")
