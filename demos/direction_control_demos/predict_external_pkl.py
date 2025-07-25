import pickle
import glob
import os
import re
import numpy as np
from metabci.brainda.algorithms.decomposition.cca import CCA

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
        # 复制9次，shape (9, 4000)
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

if __name__ == '__main__':
    # 训练集目录
    train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CCA/data'))
    print(f'训练集目录: {train_dir}')
    X_train, y_train = read_data_for_cca(train_dir)
    # 测试集目录
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    print(f'测试集目录: {test_dir}')
    X_test, y_test = read_data_for_cca(test_dir)
    # 频率集（可根据实际情况调整）
    freqs = np.arange(8, 8 + len(np.unique(y_train)), 1)
    cca_model = CCA(frequency_set=freqs, data_len=4000)
    # 预测
    pred_test = []
    for x in X_test:
        pred, _ = cca_model.fit(x)
        pred_test.append(pred)
    pred_test = np.array(pred_test)
    print('预测标签:', pred_test)
    print('真实标签:', y_test)
    print('测试集准确率:', np.mean(pred_test == y_test))