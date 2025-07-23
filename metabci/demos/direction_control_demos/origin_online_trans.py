import math
import time
import numpy as np
import mne
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from psychopy import monitors
from metabci.brainstim.paradigm import (
    SSVEP, paradigm
)
from metabci.brainstim.framework import Experiment
from metabci.brainflow.amplifiers import Neuracle, Marker
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from metabci.brainda.algorithms.decomposition import FBDSP
from metabci.brainda.utils import upper_ch_names
from metabci.brainda.paradigms import SSVEP as SSVEPParadigm
from metabci.brainda.datasets.tsinghua import Wang2016
from sklearn.base import BaseEstimator, ClassifierMixin


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
    # 将列表格式转换为SSVEPParadigm期望的元组列表格式
    paradigm_intervals = [(stim_interval[0], stim_interval[1])]
    paradigm = SSVEPParadigm(
        channels=dataset.channels,
        events=dataset.events,
        intervals=paradigm_intervals,
        srate=250)
    
    # 对raw操作,例如滤波
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
    
    # 选择指定通道
    ch_names = [ch.upper() for ch in pick_chs]
    ch_ind = [dataset.channels.index(ch) for ch in ch_names if ch in dataset.channels]
    
    # 只保留选定的通道
    X = X[:, ch_ind, :]
    
    # 标签编码
    unique_labels = np.unique(y)
    y = label_encoder(y, unique_labels)
    
    return X, y, ch_ind

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

    # 使用Wang2016的40个频率
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

class FeedbackWorker(ProcessWorker):
    def __init__(self, subjects, pick_chs, stim_interval, stim_labels,
                 srate, lsl_source_id, timeout, worker_name):
        self.subjects = subjects
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        X, y, ch_ind = read_wang2016_data(
            subjects=self.subjects,
            stim_interval=self.stim_interval,
            pick_chs=self.pick_chs)
        print("Loading Wang2016 train data successfully")
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
        print("[DEBUG] 即将接收数据，consume方法被调用")
        data = np.array(data, dtype=np.float64).T
        print(f"[DEBUG] 已接收到数据，data shape: {data.shape}")
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

    # 配置 SSVEP 范式 - 使用Wang2016的40个频率
    n_elements, rows, columns = 40, 5, 8
    stim_length, stim_width = 70, 70
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]
    fps = 240
    stim_time = 0.54
    stim_opacities = 1
    # 使用Wang2016的频率
    freqs = Wang2016._FREQS
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
    stim_interval = [0.5, 5.5]  # Wang2016使用0.5-5.5秒
    stim_labels = list(range(1, 41))  # Wang2016有40个目标
    # 使用Wang2016数据集的前2个被试
    subjects = list(range(1, 3))
    pick_chs = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    # 用原始worker替换同步worker
    worker = FeedbackWorker(
        subjects=subjects,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        stim_labels=stim_labels,
        srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name,
    )

    # 其余流程不变
    marker = Marker(interval=stim_interval, srate=srate, events=stim_labels)

    # 初始化 Neuracle
    ns = Neuracle(
        device_address=('127.0.0.1', 8712),  # 修改为Neuracle默认端口
        srate=srate,
        num_chans=9,  # 修改为Neuracle默认通道数
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
    ns.start_trans()  # Neuracle无需start_acq
    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)
    time.sleep(0.5)
    ex.run()  # 运行 SSVEP 范式

    input('按任意键退出\n')
    ns.down_worker(feedback_worker_name)
    time.sleep(1)
    ns.stop_trans()
    ns.close_connection()
    ns.clear()
    print('bye') 