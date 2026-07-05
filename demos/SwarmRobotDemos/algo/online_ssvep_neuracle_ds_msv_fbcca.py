# -*- coding: utf-8 -*-
"""博睿康 + DS-MSV-FBCCA动态停止在线示例，使用240/241和枕区8导。"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from multiprocessing import Event
from pathlib import Path
from time import perf_counter
from typing import List, Sequence

import numpy as np
from pylsl import StreamInfo, StreamOutlet
from scipy import signal
from scipy.signal import resample


ALGO_DIR = Path(__file__).resolve().parent
SWARM_DIR = ALGO_DIR.parent
REPOSITORY_ROOT = SWARM_DIR.parents[1]
DEFAULT_STIM_CONFIG = SWARM_DIR / "stim" / "NewFunc2" / "config.json"
DEFAULT_RESULT_FILE = ALGO_DIR / "result.txt"

for search_path in (REPOSITORY_ROOT, ALGO_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ds_msv_fbcca import (  # noqa: E402
    DEFAULT_DECISION_THRESHOLDS,
    DEFAULT_DECISION_WINDOWS,
    DEFAULT_PASSBANDS,
    DSMSVFBCCA,
)
from metabci.brainflow.workers import ProcessWorker  # noqa: E402
from online_ssvep_neuracle_fbscca import (  # noqa: E402
    LoggingNeuracle,
    OCCIPITAL_CHANNEL_INDICES,
    OCCIPITAL_CHANNEL_NAMES,
    START_TRIGGER,
    STOP_TRIGGER,
    parse_int_list,
    trace,
    wait_for_worker_startup,
)


def generate_references(
    frequencies: np.ndarray,
    phases: np.ndarray,
    srate: float,
    sample_count: int,
    harmonics: int,
) -> np.ndarray:
    """按精确采样点数生成正余弦参考，避免浮点时长造成少一个采样点。"""
    time_points = np.arange(int(sample_count), dtype=float) / float(srate)
    references = []
    for frequency, phase in zip(frequencies, phases):
        components = []
        for harmonic in range(1, int(harmonics) + 1):
            angle = (
                2 * np.pi * harmonic * float(frequency) * time_points
                + np.pi * float(phase)
            )
            components.extend((np.sin(angle), np.cos(angle)))
        references.append(np.stack(components))
    return np.stack(references)


class ProgressiveDecisionMarker:
    """以240开始、241结束，并在各动态停止时间窗输出累积Epoch。"""

    def __init__(
        self,
        decision_windows: Sequence[float],
        srate: float,
        start_trigger: int = START_TRIGGER,
        stop_trigger: int = STOP_TRIGGER,
    ) -> None:
        self.start_trigger = int(start_trigger)
        self.stop_trigger = int(stop_trigger)
        self.window_samples = [
            int(round(float(window) * float(srate))) for window in decision_windows
        ]
        if not self.window_samples or any(count <= 0 for count in self.window_samples):
            raise ValueError("动态停止时间窗必须大于0")
        if self.window_samples != sorted(set(self.window_samples)):
            raise ValueError("动态停止时间窗必须严格递增")
        self.clear()

    def clear(self) -> None:
        self.buffer = []
        self.ready_epoch = None
        self.next_window_index = 0
        self.active = False
        self.armed = True
        self.current_event = None

    def append(self, sample) -> None:
        event = int(round(sample[-1]))
        if event == self.start_trigger and self.armed:
            self.buffer = [sample]
            self.ready_epoch = None
            self.next_window_index = 0
            self.active = True
            self.armed = False
            self.current_event = event
            trace("事件", "检测到Trigger=240，开始累积动态停止时间窗")
            return

        if event != self.start_trigger:
            self.armed = True
        if self.active:
            self.buffer.append(sample)
            if event == self.stop_trigger:
                self.active = False
                trace(
                    "事件",
                    "检测到Trigger=241，本轮停止，共缓存{}个采样点".format(
                        len(self.buffer)
                    ),
                )

    def __call__(self, event: int) -> bool:
        if not self.active or self.next_window_index >= len(self.window_samples):
            return False
        required = self.window_samples[self.next_window_index]
        if len(self.buffer) < required:
            return False

        self.ready_epoch = list(self.buffer[:required])
        trace(
            "分段",
            "Trigger={}的第{}个累积窗就绪：{}个采样点".format(
                self.current_event, self.next_window_index + 1, required
            ),
        )
        self.next_window_index += 1
        return True

    def get_epoch(self):
        return self.ready_epoch


class OriginalPreprocessor:
    """复现压缩包中的50Hz陷波和90Hz低通预处理。"""

    def __init__(
        self,
        input_rate: float,
        output_rate: float,
        eeg_channel_indices: Sequence[int],
        offset_seconds: float,
    ) -> None:
        self.input_rate = float(input_rate)
        self.output_rate = float(output_rate)
        self.eeg_channel_indices = np.asarray(eeg_channel_indices, dtype=int)
        self.offset_samples = int(round(float(offset_seconds) * self.output_rate))
        if self.output_rate <= 200:
            raise ValueError("复现原90Hz低通时，重采样率必须大于200Hz")

        self.notch_b, self.notch_a = signal.iircomb(
            50.0, 13.0, ftype="notch", fs=self.output_rate
        )
        nyquist = self.output_rate / 2.0
        order, cutoff = signal.ellipord(90.0 / nyquist, 100.0 / nyquist, 3, 60)
        self.low_b, self.low_a = signal.ellip(order, 1, 60, cutoff)

    def transform(self, samples: np.ndarray) -> np.ndarray:
        if samples.ndim != 2:
            raise ValueError("Epoch必须为二维数组：采样点×通道")
        if samples.shape[1] <= int(np.max(self.eeg_channel_indices)):
            raise ValueError("脑电通道索引超出输入数据范围")

        eeg = samples[:, self.eeg_channel_indices].T
        if not np.all(np.isfinite(eeg)):
            raise ValueError("Epoch包含NaN或Inf")
        output_count = int(round(eeg.shape[-1] * self.output_rate / self.input_rate))
        eeg = resample(eeg, output_count, axis=-1)
        if eeg.shape[-1] <= self.offset_samples:
            raise ValueError("Epoch长度不足以移除视觉延迟")
        eeg = eeg[:, self.offset_samples :]
        eeg = signal.filtfilt(self.notch_b, self.notch_a, eeg, axis=-1)
        eeg = signal.filtfilt(self.low_b, self.low_a, eeg, axis=-1)
        return eeg


class DSMSVFeedbackWorker(ProcessWorker):
    """逐时间窗执行DS-MSV-FBCCA并回传首个可信判决。"""

    def __init__(
        self,
        frequencies: np.ndarray,
        phases: np.ndarray,
        sample_rate: float,
        resample_rate: float,
        eeg_channel_indices: List[int],
        offset_seconds: float,
        decision_windows: Sequence[float],
        decision_thresholds: Sequence[float],
        harmonics: int,
        lsl_stream_name: str,
        lsl_source_id: str,
        result_file: Path,
        timeout: float = 0.05,
        name: str = "ds_msv_feedback_worker",
    ) -> None:
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.phases = np.asarray(phases, dtype=float)
        self.sample_rate = float(sample_rate)
        self.resample_rate = float(resample_rate)
        self.eeg_channel_indices = list(eeg_channel_indices)
        self.offset_seconds = float(offset_seconds)
        self.decision_windows = tuple(float(value) for value in decision_windows)
        self.decision_thresholds = tuple(float(value) for value in decision_thresholds)
        self.harmonics = int(harmonics)
        self.lsl_stream_name = lsl_stream_name
        self.lsl_source_id = lsl_source_id
        self.result_file = Path(result_file)
        self.startup_ready = Event()
        self.decision_sent = False
        super().__init__(timeout=timeout, name=name)

    def pre(self) -> None:
        trace("进程", "DS-MSV-FBCCA Worker子进程已启动")
        try:
            effective_samples = int(
                round(
                    (self.decision_windows[-1] - self.offset_seconds)
                    * self.resample_rate
                )
            )
            references = generate_references(
                self.frequencies,
                self.phases,
                srate=self.resample_rate,
                sample_count=effective_samples,
                harmonics=self.harmonics,
            )
            filterbank = DSMSVFBCCA.generate_filterbank(
                DEFAULT_PASSBANDS, srate=self.resample_rate
            )
            self.estimator = DSMSVFBCCA(
                filterbank=filterbank,
                decision_thresholds=self.decision_thresholds,
            ).fit(Yf=references)
            self.preprocessor = OriginalPreprocessor(
                input_rate=self.sample_rate,
                output_rate=self.resample_rate,
                eeg_channel_indices=self.eeg_channel_indices,
                offset_seconds=self.offset_seconds,
            )
            trace(
                "算法",
                "DS-MSV-FBCCA初始化完成：目标数={}，动态窗={}".format(
                    len(self.frequencies), self.decision_windows
                ),
            )

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
            trace("错误", "DS-MSV-FBCCA初始化失败：{}: {}".format(type(exc).__name__, exc))
            traceback.print_exc()
            raise

        trace("回传", "等待刺激端连接LSL输出流")
        while not self._exit.is_set():
            if self.outlet.wait_for_consumers(1e-3):
                break
        if not self._exit.is_set():
            trace("回传", "刺激端已连接LSL输出流")

    def _publish(self, label: int) -> None:
        try:
            self.result_file.write_text(str(label), encoding="utf-8")
            trace("结果文件", "已覆写{}，最新标签={}".format(self.result_file, label))
        except OSError as exc:
            trace("结果文件", "写入失败：{}: {}".format(type(exc).__name__, exc))

        if self.outlet.have_consumers():
            self.outlet.push_sample([int(label)])
            trace("回传", "已通过LSL发送标签={}".format(label))
        else:
            trace("回传", "标签={}未发送：没有刺激端接收者".format(label))

    def consume(self, data) -> None:
        samples = np.asarray(data, dtype=float)
        first_window_count = int(round(self.decision_windows[0] * self.sample_rate))
        if samples.shape[0] == first_window_count:
            self.decision_sent = False
        if self.decision_sent:
            return

        window_counts = np.asarray(
            [int(round(value * self.sample_rate)) for value in self.decision_windows]
        )
        decision_index = int(np.argmin(np.abs(window_counts - samples.shape[0])))
        is_final = decision_index == len(self.decision_windows) - 1
        started_at = perf_counter()

        try:
            eeg = self.preprocessor.transform(samples)
            labels, confidences = self.estimator.predict_with_confidence(eeg[np.newaxis])
            label = int(labels[0]) + 1
            confidence = float(confidences[0])
            accepted = is_final or self.estimator.should_stop(confidence, decision_index)
            trace(
                "算法",
                "时间窗={:.2f}秒，候选类别={}，置信度差={:.6f}，阈值={:.6f}，{}".format(
                    self.decision_windows[decision_index],
                    label,
                    confidence,
                    self.decision_thresholds[decision_index],
                    "接受判决" if accepted else "继续等待",
                ),
            )
        except Exception as exc:
            trace(
                "算法",
                "第{}个时间窗判决失败：{}: {}".format(
                    decision_index + 1, type(exc).__name__, exc
                ),
            )
            if not is_final:
                return
            label = len(self.frequencies) + 1
            accepted = True

        trace("算法", "本次计算耗时={:.1f}毫秒".format((perf_counter() - started_at) * 1000))
        if accepted:
            self._publish(label)
            self.decision_sent = True

    def post(self) -> None:
        trace("算法", "DS-MSV-FBCCA反馈进程已停止")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("n_elements", "fps", "stim_time"):
        if key not in config:
            raise ValueError("刺激配置缺少{}".format(key))
    return config


def build_frequencies_and_phases(n_elements: int):
    frequencies = 8.0 + 0.2 * np.arange(n_elements)
    phases = np.asarray([index * 0.35 % 2 for index in range(n_elements)])
    return frequencies, phases


def run_online(args: argparse.Namespace) -> None:
    config = load_config(Path(args.stim_config))
    n_elements = int(config["n_elements"])
    if DEFAULT_DECISION_WINDOWS[-1] > float(config["stim_time"]):
        raise ValueError("刺激时长必须不小于最终动态停止时间窗1.48秒")

    eeg_channels = parse_int_list(args.eeg_channels)
    if not eeg_channels:
        raise ValueError("至少需要一个脑电通道")
    if max(eeg_channels) >= int(args.num_channels) - 1:
        raise ValueError("最后一个博睿康通道保留给Trigger，不能作为脑电通道")
    selected_names = [
        name
        for index, name in zip(OCCIPITAL_CHANNEL_INDICES, OCCIPITAL_CHANNEL_NAMES)
        if index in eeg_channels
    ]
    trace(
        "配置",
        "DS-MSV-FBCCA使用枕区8导：索引={}，导联={}".format(
            eeg_channels, selected_names or "按自定义索引"
        ),
    )

    frequencies, phases = build_frequencies_and_phases(n_elements)
    # BaseAmplifier.up_worker当前固定使用feedback_worker键，保持该名称以兼容框架。
    worker_name = "feedback_worker"
    worker = DSMSVFeedbackWorker(
        frequencies=frequencies,
        phases=phases,
        sample_rate=args.sample_rate,
        resample_rate=args.resample_rate,
        eeg_channel_indices=eeg_channels,
        offset_seconds=args.epoch_offset,
        decision_windows=DEFAULT_DECISION_WINDOWS,
        decision_thresholds=DEFAULT_DECISION_THRESHOLDS,
        harmonics=args.harmonics,
        lsl_stream_name=args.lsl_stream_name,
        lsl_source_id=args.lsl_source_id,
        result_file=Path(args.result_file),
        name=worker_name,
    )
    marker = ProgressiveDecisionMarker(
        decision_windows=DEFAULT_DECISION_WINDOWS,
        srate=args.sample_rate,
        start_trigger=START_TRIGGER,
        stop_trigger=STOP_TRIGGER,
    )
    amplifier = LoggingNeuracle(
        device_address=(args.host, args.port),
        srate=args.sample_rate,
        num_chans=args.num_channels,
        valid_events=(START_TRIGGER, STOP_TRIGGER),
    )

    connected = False
    worker_started = False
    streaming = False
    try:
        trace("连接", "正在连接博睿康DataService {}:{}".format(args.host, args.port))
        amplifier.connect_tcp()
        connected = True
        trace("连接", "博睿康TCP连接成功")
        amplifier.register_worker(worker_name, worker, marker)
        amplifier.up_worker(worker_name)
        worker_started = True
        wait_for_worker_startup(worker, timeout=args.worker_startup_timeout)
        amplifier.start_trans()
        streaming = True
        trace("数据", "数据流已启动，等待Trigger=240/241和渐进时间窗")
        input("需要停止算法时，请在此终端按Enter。\n")
    finally:
        if streaming:
            amplifier.stop_trans()
            worker.join(timeout=5.0)
        elif worker_started:
            if worker.is_alive():
                worker.stop()
            worker.join(timeout=3.0)
        if worker_started and worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)
        if connected:
            amplifier.close_connection()
        trace("停止", "DS-MSV-FBCCA在线程序已关闭")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Neuracle + DS-MSV-FBCCA动态停止在线SSVEP示例"
    )
    parser.add_argument("--stim-config", default=str(DEFAULT_STIM_CONFIG))
    parser.add_argument("--result-file", default=str(DEFAULT_RESULT_FILE))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--sample-rate", type=float, default=1000.0)
    parser.add_argument("--num-channels", type=int, default=65)
    parser.add_argument(
        "--eeg-channels",
        default=",".join(str(index) for index in OCCIPITAL_CHANNEL_INDICES),
        help="默认使用指定的8个枕区导联；最后一个设备通道保留给Trigger。",
    )
    parser.add_argument("--epoch-offset", type=float, default=0.14)
    parser.add_argument("--resample-rate", type=float, default=250.0)
    parser.add_argument("--harmonics", type=int, default=4)
    parser.add_argument("--lsl-stream-name", default="meta_feedback")
    parser.add_argument("--lsl-source-id", default="meta_online_worker")
    parser.add_argument("--worker-startup-timeout", type=float, default=30.0)
    return parser


if __name__ == "__main__":
    run_online(make_parser().parse_args())
