# -*- coding: utf-8 -*-
"""
SwarmRobotDemos 的在线 SSVEP 算法端。

本文件默认对应 ``stim/NewFunc2/stim_customized_service.py`` 的在线模式：

1. 从博睿康 Neuracle / Neusen W DataService 接收 EEG 数据；
2. 根据最后一个 trigger 通道切分单 trial epoch；
3. 使用无需训练数据的 FBSCCA 解码 SSVEP；
4. 通过 LSL ``source_id=meta_online_worker`` 把预测标签发回刺激端。

使用前请确保刺激端已经开启 online，并通过博睿康 trigger box 发送标签。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from multiprocessing import Event
from pathlib import Path
from time import monotonic, perf_counter, strftime
from typing import Iterable, List

import numpy as np
from pylsl import StreamInfo, StreamOutlet
from scipy.signal import resample


ALGO_DIR = Path(__file__).resolve().parent
SWARM_DIR = ALGO_DIR.parent
REPOSITORY_ROOT = SWARM_DIR.parents[1]
DEFAULT_STIM_CONFIG = SWARM_DIR / "stim" / "NewFunc2" / "config.json"
RESULT_FILE = ALGO_DIR / "result.txt"

for search_path in (REPOSITORY_ROOT, ALGO_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from metabci.brainda.algorithms.decomposition import FBSCCA  # noqa: E402
from metabci.brainda.algorithms.decomposition.base import (  # noqa: E402
    generate_cca_references,
    generate_filterbank,
)
from metabci.brainflow.amplifiers import Marker, Neuracle  # noqa: E402
from metabci.brainflow.workers import ProcessWorker  # noqa: E402


def trace(stage: str, message: str) -> None:
    """输出便于联调检索的实时流程日志。"""
    print("[{}][{}] {}".format(strftime("%H:%M:%S"), stage, message), flush=True)


class LoggingNeuracle(Neuracle):
    """按秒汇总博睿康 TCP 数据，避免每个数据包都刷屏。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.chunk_count = 0
        self.sample_count = 0
        self.last_report_time = 0.0

    def recv(self):
        samples = super().recv()
        if not samples:
            return samples

        self.chunk_count += 1
        self.sample_count += len(samples)
        current_time = monotonic()
        if self.chunk_count == 1 or current_time - self.last_report_time >= 1.0:
            nonzero_triggers = sorted(
                {
                    int(round(sample[-1]))
                    for sample in samples
                    if int(round(sample[-1])) != 0
                }
            )
            trigger_text = nonzero_triggers if nonzero_triggers else "none"
            trace(
                "数据",
                "收到TCP数据块：序号={}，本次数={}，通道数={}，累计采样数={}，"
                "非零Trigger={}".format(
                    self.chunk_count,
                    len(samples),
                    len(samples[0]),
                    self.sample_count,
                    trigger_text,
                ),
            )
            self.last_report_time = current_time
        return samples


class LoggingMarker(Marker):
    """打印 Trigger 检出以及对应 Epoch 就绪时刻。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_trigger_event = None

    def __call__(self, event: int):
        event = int(event)
        is_new_event = (
            event != 0
            and self.is_rising
            and (self.events is None or event in self.events)
        )
        if is_new_event:
            self.last_trigger_event = event
            trace("事件", "从最后一个数据通道检测到Trigger={}".format(event))

        epoch_ready = super().__call__(event)
        if epoch_ready:
            trace(
                "分段",
                "Trigger={}对应的数据窗已就绪，共{}个采样点，准备送入FBSCCA".format(
                    self.last_trigger_event,
                    self.epoch_ind[1] - self.epoch_ind[0],
                ),
            )
        return epoch_ready


class FBSCCADecoder:
    """对 MetaBCI 训练自由 FBSCCA 解码器做一层在线预测封装。"""

    def __init__(
        self,
        frequencies: np.ndarray,
        phases: np.ndarray,
        input_rate: float,
        output_rate: float,
        eeg_channel_indices: Iterable[int],
        epoch_duration: float,
        harmonics: int = 3,
    ) -> None:
        self.input_rate = float(input_rate)
        self.output_rate = float(output_rate)
        self.eeg_channel_indices = np.asarray(list(eeg_channel_indices), dtype=int)
        self.output_samples = int(round(epoch_duration * self.output_rate))
        if self.output_samples <= 0:
            raise ValueError("epoch duration is too short for decoder output")

        # 滤波器组沿用 MetaBCI 在线 SSVEP demo 的常用设置。
        passbands = [[6.0, 90.0], [14.0, 90.0], [22.0, 90.0]]
        stopbands = [[4.0, 92.0], [12.0, 92.0], [20.0, 92.0]]
        filter_weights = np.asarray([1.25, 0.67, 0.5], dtype=float)
        filterbank = generate_filterbank(passbands, stopbands, srate=int(output_rate))
        references = generate_cca_references(
            frequencies,
            srate=int(output_rate),
            T=self.output_samples / self.output_rate,
            phases=phases,
            n_harmonics=int(harmonics),
        )

        self.estimator = FBSCCA(
            filterbank=filterbank,
            n_components=1,
            filterweights=filter_weights,
            n_jobs=1,
        )
        dummy_x = np.zeros(
            (1, len(self.eeg_channel_indices), self.output_samples),
            dtype=float,
        )
        self.estimator.fit(dummy_x, np.zeros(1, dtype=int), Yf=references)

    def predict(self, samples: np.ndarray) -> int:
        """返回 brainstim 可直接接收的 1 起始标签。"""

        if samples.ndim != 2:
            raise ValueError("online epoch must be a 2D array: samples x channels")
        if samples.shape[1] <= int(np.max(self.eeg_channel_indices)):
            raise ValueError("eeg_channel_indices exceeds online data channels")

        # 博睿康数据是 samples x channels；FBSCCA 需要 trials x channels x samples。
        epoch = samples[:, self.eeg_channel_indices].T
        if not np.all(np.isfinite(epoch)):
            raise ValueError("online epoch contains NaN or Inf")

        channel_std = np.std(epoch, axis=-1, keepdims=True)
        if np.any(channel_std < 1e-12):
            raise ValueError("flat EEG channel detected")
        epoch = (epoch - np.mean(epoch, axis=-1, keepdims=True)) / channel_std
        epoch = resample(epoch, self.output_samples, axis=-1)

        # FBSCCA 返回 0 起始类别；brainstim 接收 1 起始标签后会在内部减 1。
        return int(self.estimator.predict(epoch[np.newaxis])[0]) + 1


class FBSCCAFeedbackWorker(ProcessWorker):
    """解码博睿康 epoch，并把预测标签发送给刺激端。"""

    def __init__(
        self,
        frequencies: np.ndarray,
        phases: np.ndarray,
        sample_rate: float,
        resample_rate: float,
        eeg_channel_indices: List[int],
        epoch_interval: List[float],
        lsl_source_id: str,
        lsl_stream_name: str,
        harmonics: int,
        timeout: float = 0.05,
        name: str = "feedback_worker",
    ) -> None:
        self.frequencies = frequencies
        self.phases = phases
        self.sample_rate = sample_rate
        self.resample_rate = resample_rate
        self.eeg_channel_indices = eeg_channel_indices
        self.epoch_interval = epoch_interval
        self.lsl_source_id = lsl_source_id
        self.lsl_stream_name = lsl_stream_name
        self.harmonics = harmonics
        # Windows下Worker使用独立进程；主进程通过该事件确认解码器和LSL均已就绪。
        self.startup_ready = Event()
        super().__init__(timeout=timeout, name=name)

    def pre(self) -> None:
        trace("进程", "FBSCCA Worker子进程已启动，开始初始化")
        try:
            epoch_duration = self.epoch_interval[1] - self.epoch_interval[0]
            self.decoder = FBSCCADecoder(
                frequencies=self.frequencies,
                phases=self.phases,
                input_rate=self.sample_rate,
                output_rate=self.resample_rate,
                eeg_channel_indices=self.eeg_channel_indices,
                epoch_duration=epoch_duration,
                harmonics=self.harmonics,
            )
            trace(
                "算法",
                "FBSCCA初始化完成：目标数={}，脑电通道数={}，输入采样率={}Hz，"
                "重采样率={}Hz，数据窗={}秒".format(
                    len(self.frequencies),
                    len(self.eeg_channel_indices),
                    self.sample_rate,
                    self.resample_rate,
                    epoch_duration,
                ),
            )

            # 刺激端 online=True 时会通过这个 source_id 查找LSL流。
            info = StreamInfo(
                name=self.lsl_stream_name,
                type="Markers",
                channel_count=1,
                nominal_srate=0,
                channel_format="int32",
                source_id=self.lsl_source_id,
            )
            self.outlet = StreamOutlet(info)
            trace(
                "回传",
                "LSL输出流已创建：名称={!r}，source_id={!r}".format(
                    self.lsl_stream_name, self.lsl_source_id
                ),
            )
            self.startup_ready.set()
        except Exception as exc:
            trace(
                "错误",
                "Worker初始化失败，未能创建可用LSL流：{}: {}".format(
                    type(exc).__name__, exc
                ),
            )
            traceback.print_exc()
            raise

        trace("回传", "等待刺激端连接LSL输出流")
        while not self._exit.is_set():
            if self.outlet.wait_for_consumers(1e-3):
                break
        if self._exit.is_set():
            trace("回传", "等待连接期间收到停止信号")
        else:
            trace("回传", "刺激端已连接LSL输出流")

    def consume(self, data) -> None:
        samples = np.asarray(data, dtype=float)
        trace(
            "分段",
            "算法进程收到数据窗，形状={}（采样点×通道）".format(samples.shape),
        )
        started_at = perf_counter()
        try:
            predicted = self.decoder.predict(samples)
        except Exception as exc:
            # 20 个有效刺激标签之外，使用 21 明确表示本轮判决失败。
            predicted = len(self.frequencies) + 1
            trace(
                "算法",
                "FBSCCA判决失败：{}: {}；发送失败标签={}".format(
                    type(exc).__name__, exc, predicted
                ),
            )

        elapsed_ms = (perf_counter() - started_at) * 1000
        if 1 <= predicted <= len(self.frequencies):
            frequency = self.frequencies[predicted - 1]
            trace(
                "算法",
                "判决完成：类别={}，对应频率={:.1f}Hz，耗时={:.1f}毫秒".format(
                    predicted, frequency, elapsed_ms
                ),
            )
        else:
            trace(
                "算法",
                "本轮无有效判决：失败标签={}，耗时={:.1f}毫秒".format(
                    predicted, elapsed_ms
                ),
            )

        try:
            RESULT_FILE.write_text(str(predicted), encoding="utf-8")
            trace("结果文件", "已覆写{}，最新标签={}".format(RESULT_FILE, predicted))
        except OSError as exc:
            trace("结果文件", "写入失败：{}: {}".format(type(exc).__name__, exc))

        if self.outlet.have_consumers():
            self.outlet.push_sample([predicted])
            trace("回传", "已通过LSL向刺激端发送判决类别={}".format(predicted))
        else:
            trace("回传", "判决类别={}未发送：当前没有刺激端接收者".format(predicted))

    def post(self) -> None:
        trace("算法", "FBSCCA反馈进程已停止")


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_stimulus_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"n_elements", "fps", "stim_time"}
    missing = required - set(config)
    if missing:
        raise ValueError("stim config missing keys: {}".format(", ".join(missing)))
    return config


def build_stimulus_timing(config: dict, epoch_start: float, epoch_end: float | None):
    stim_time = float(config["stim_time"])
    if epoch_end is None:
        epoch_end = min(stim_time, 1.14)
    if not 0 <= epoch_start < epoch_end <= stim_time:
        raise ValueError(
            "epoch interval must satisfy 0 <= start < end <= stim_time; "
            "got [{}, {}], stim_time={}".format(epoch_start, epoch_end, stim_time)
        )
    return [float(epoch_start), float(epoch_end)]


def build_frequencies_and_phases(n_elements: int):
    # 必须和 stim/NewFunc1/stim_customized_service.py 中的频率、相位规则一致。
    frequencies = 8.0 + 0.2 * np.arange(n_elements)
    phases = np.asarray([i * 0.35 % 2 for i in range(n_elements)], dtype=float)
    return frequencies, phases


def wait_for_worker_startup(worker, timeout: float) -> None:
    """等待子进程确认FBSCCA和LSL就绪，并及时暴露子进程启动失败。"""
    deadline = monotonic() + float(timeout)
    while monotonic() < deadline:
        if worker.startup_ready.wait(timeout=0.1):
            trace(
                "进程",
                "Worker启动完成：pid={}，FBSCCA和LSL均已就绪".format(worker.pid),
            )
            return
        if worker.exitcode is not None:
            raise RuntimeError(
                "FBSCCA Worker启动失败，子进程退出码={}；请查看上方完整异常".format(
                    worker.exitcode
                )
            )
    raise TimeoutError(
        "等待FBSCCA Worker创建LSL流超过{:.1f}秒".format(float(timeout))
    )


def run_online(args: argparse.Namespace) -> None:
    stim_config = load_stimulus_config(Path(args.stim_config))
    n_elements = int(stim_config["n_elements"])
    frequencies, phases = build_frequencies_and_phases(n_elements)
    epoch_interval = build_stimulus_timing(
        stim_config,
        epoch_start=float(args.epoch_start),
        epoch_end=args.epoch_end,
    )
    eeg_channels = parse_int_list(args.eeg_channels)
    if not eeg_channels:
        raise ValueError("at least one EEG channel index is required")
    if max(eeg_channels) >= int(args.num_channels) - 1:
        raise ValueError(
            "the last Neuracle channel is treated as trigger; "
            "eeg channel indices must be smaller than num_channels - 1"
        )

    trace(
        "配置",
        "刺激配置={!s}，目标数={}，设备地址={}:{}，采样率={}Hz，"
        "通道数={}（最后一个通道为Trigger），数据窗={}".format(
            args.stim_config,
            n_elements,
            args.host,
            args.port,
            args.sample_rate,
            args.num_channels,
            epoch_interval,
        ),
    )

    worker_name = "feedback_worker"
    worker = FBSCCAFeedbackWorker(
        frequencies=frequencies,
        phases=phases,
        sample_rate=float(args.sample_rate),
        resample_rate=float(args.resample_rate),
        eeg_channel_indices=eeg_channels,
        epoch_interval=epoch_interval,
        lsl_source_id=args.lsl_source_id,
        lsl_stream_name=args.lsl_stream_name,
        harmonics=int(args.harmonics),
        name=worker_name,
    )
    marker = LoggingMarker(
        interval=epoch_interval,
        srate=float(args.sample_rate),
        events=list(range(1, n_elements + 1)),
    )
    amplifier = LoggingNeuracle(
        device_address=(args.host, int(args.port)),
        srate=float(args.sample_rate),
        num_chans=int(args.num_channels),
    )

    connected = False
    worker_started = False
    streaming = False
    try:
        trace(
            "连接",
            "正在连接博睿康DataService {}:{}".format(args.host, args.port),
        )
        amplifier.connect_tcp()
        connected = True
        trace("连接", "博睿康TCP连接成功")
        amplifier.register_worker(worker_name, worker, marker)
        trace("进程", "已注册Trigger检测器和FBSCCA反馈进程")
        amplifier.up_worker(worker_name)
        worker_started = True
        wait_for_worker_startup(worker, timeout=float(args.worker_startup_timeout))
        amplifier.start_trans()
        streaming = True
        trace("数据", "博睿康数据流已启动，正在等待脑电数据和Trigger")
        trace("运行", "请启动NewFunc2刺激程序，并在刺激窗口中按Enter开始")
        input("需要停止算法时，请在此终端按Enter。\n")
    finally:
        if streaming:
            amplifier.stop_trans()
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)
        elif worker_started:
            if worker.is_alive():
                worker.stop()
            worker.join(timeout=3.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)
        if connected:
            amplifier.close_connection()
        trace("停止", "在线SSVEP算法程序已关闭")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Neuracle + FBSCCA online SSVEP worker for SwarmRobotDemos"
    )
    parser.add_argument("--stim-config", default=str(DEFAULT_STIM_CONFIG))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--sample-rate", type=float, default=1000.0)
    parser.add_argument(
        "--num-channels",
        type=int,
        default=65,
        help="Neuracle data channels including the final trigger channel.",
    )
    parser.add_argument(
        "--eeg-channels",
        default=",".join(str(index) for index in range(64)),
        help="Comma-separated EEG channel indices, excluding trigger channel.",
    )
    parser.add_argument("--epoch-start", type=float, default=0.14)
    parser.add_argument(
        "--epoch-end",
        type=float,
        default=None,
        help="Defaults to min(stim_time, 1.14) from the stimulus config.",
    )
    parser.add_argument("--resample-rate", type=float, default=250.0)
    parser.add_argument("--harmonics", type=int, default=3)
    parser.add_argument(
        "--worker-startup-timeout",
        type=float,
        default=30.0,
        help="等待FBSCCA子进程和LSL输出流就绪的最长秒数。",
    )
    parser.add_argument("--lsl-stream-name", default="meta_feedback")
    parser.add_argument("--lsl-source-id", default="meta_online_worker")
    return parser


if __name__ == "__main__":
    run_online(make_parser().parse_args())
