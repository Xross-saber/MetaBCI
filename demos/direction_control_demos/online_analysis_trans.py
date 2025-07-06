import math
import time
import numpy as np
import mne
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from psychopy import monitors
from psychopy.tools.monitorunittools import deg2pix
from metabci.brainstim.paradigm import (
    SSVEP, paradigm
)
from metabci.brainstim.framework import Experiment
from metabci.brainflow.amplifiers import NeuroScan, Marker
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from metabci.brainda.algorithms.decomposition import FBDSP
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.base import BaseEstimator, ClassifierMixin
from offline_analysis import raw_hook, frequency_feature
from metabci.brainda.algorithms.decomposition import FBCCA  # 替换 FBDSP
from metabci.brainda.algorithms.dynamic_stopping.bayes import Bayes
import pickle
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

def read_data(run_files, chs, interval, labels):
    Xs, ys = [], []
    for run_file in run_files:
        raw = read_raw_cnt(run_file, preload=True, verbose=False)
        raw = upper_ch_names(raw)
        events = mne.events_from_annotations(
            raw, event_id=lambda x: int(x), verbose=False)[0]
        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        epochs = mne.Epochs(raw, events,
                            event_id=labels,
                            tmin=interval[0],
                            tmax=interval[1],
                            baseline=None,
                            picks=ch_picks,
                            verbose=False)

        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]
            Xs.append(X)
            ys.append(np.ones((len(X)))*label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)
    return Xs, ys, ch_picks

def train_model(X, y, srate=250):
    y = np.reshape(y, (-1))
    # 降采样
    X = resample(X, up=256, down=srate)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    # 滤波器组设置
    wp = [
        [6, 88], [14, 88], [22, 88], [30, 88], [38, 88]
    ]
    ws = [
        [4, 90], [12, 90], [20, 90], [28, 90], [36, 90]
    ]
    filterweights = np.arange(1, 6)**(-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 256)

    freqs = np.arange(8, 16, 0.2)
    Yf = generate_cca_references(freqs, srate=256, T=0.5, n_harmonics=5)
    model =FBDSP(filterbank, filterweights=np.array(filterweights))
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

class FeedbackWorker(ProcessWorker):
    def __init__(self, run_files, pick_chs, stim_interval, stim_labels,
                 srate, lsl_source_id, timeout, worker_name):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        X, y, ch_ind = read_data(run_files=self.run_files,
                                 chs=self.pick_chs,
                                 interval=self.stim_interval,
                                 labels=self.stim_labels)
        print("Loading train data successfully")
        acc = offline_validation(X, y, srate=self.srate)
        print("Current Model accuracy:{:.2f}".format(acc))
        self.estimator = train_model(X, y, srate=self.srate)
        self.ch_ind = ch_ind
        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)
        print('Waiting connection...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('Connected')

    def consume(self, data):
        data = np.array(data, dtype=np.float64).T
        data = data[self.ch_ind]
        p_labels = model_predict(data, srate=self.srate, model=self.estimator)
        p_labels = np.array([int(p_labels + 1)])
        print('predict_id_paradigm', p_labels)
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels.tolist())

    def post(self):
        pass

if __name__ == "__main__":
    # 配置显示器
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,
        verbose=False,
    )
    mon.setSizePix([1920, 1080])
    mon.save()

    # 配置实验窗口
    bg_color_warm = np.array([0, 0, 0])
    win_size = np.array([1440, 960])
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,
        screen_id=0,
        win_size=win_size,
        is_fullscr=False,
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()

    # 配置 SSVEP 范式
    n_elements, rows, columns = 40, 5, 8
    stim_length, stim_width = 70, 70
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]
    fps = 240
    stim_time = 0.54  # 匹配 Online_ssvep.py 的 stim_interval=[0.14, 0.68]
    stim_opacities = 1
    freqs = np.arange(8, 16, 0.2)
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])

    basic_ssvep = SSVEP(win=win)
    basic_ssvep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_ssvep.config_text(tex_color=tex_color)
    basic_ssvep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=stim_color,
        stim_opacities=stim_opacities,
        freqs=freqs,
        phases=phases,
    )
    basic_ssvep.config_index()
    basic_ssvep.config_response()

    # 配置 EEG 采集参数
    srate = 1000
    stim_interval = [0.14, 0.68]
    stim_labels = list(range(1, 41))
    cnts = 1
    filepath = "data\\train\\sub1"
    runs = list(range(1, cnts+1))
    run_files = ['{:s}\\{:d}.cnt'.format(filepath, run) for run in runs]
    pick_chs = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    # 初始化 FeedbackWorker 和 Marker
    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        stim_labels=stim_labels,
        srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name,
    )
    marker = Marker(interval=stim_interval, srate=srate, events=stim_labels)

    # 初始化 NeuroScan
    ns = NeuroScan(
        device_address=('192.168.1.30', 4000),
        srate=srate,
        num_chans=68,
    )

    # 注册范式
    bg_color = np.array([0.3, 0.3, 0.3])
    display_time = 1
    index_time = 1
    rest_time = 0.5
    response_time = 1
    port_addr = None
    nrep = 2
    online = True
    ex.register_paradigm(
        "进入操控界面",
        paradigm,
        VSObject=basic_ssvep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="ssvep",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    # 启动 EEG 采集和刺激呈现
    ns.connect_tcp()
    ns.start_acq()
    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)
    time.sleep(0.5)
    ns.start_trans()
    ex.run()  # 运行 SSVEP 范式

    input('按任意键退出\n')
    ns.down_worker(feedback_worker_name)
    time.sleep(1)
    ns.stop_trans()
    ns.stop_acq()
    ns.close_connection()
    ns.clear()
    print('bye')