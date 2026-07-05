# -*- coding: utf-8 -*-
"""监听SSVEP判决结果，并按DCMM状态向1-56号小车发送串口指令。"""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


FLOW_DIR = Path(__file__).resolve().parent
SWARM_DIR = FLOW_DIR.parent
STIM_DIR = SWARM_DIR / "stim" / "NewFunc2"
DEFAULT_FLOW_CONFIG = FLOW_DIR / "vehicle_command_config.json"
DEFAULT_STIM_CONFIG = STIM_DIR / "config.json"
DEFAULT_STATUS_CONFIG = STIM_DIR / "status_display.json"
DEFAULT_RESULT_FILE = SWARM_DIR / "algo" / "result.txt"

if str(STIM_DIR) not in sys.path:
    sys.path.insert(0, str(STIM_DIR))

from dynamic_content_mapping import DynamicContentMapper  # noqa: E402


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_vehicle_frame(vehicle_id: int, command_code: int) -> bytes:
    """生成 ``AA ID CMD CHECKSUM DD`` 协议帧。"""
    vehicle_id = int(vehicle_id)
    command_code = int(command_code)
    if not 1 <= vehicle_id <= 56:
        raise ValueError("车辆ID必须在1-56之间")
    if not 0 <= command_code <= 255:
        raise ValueError("命令码必须在0-255之间")
    checksum = (0xFF - ((0xAA + vehicle_id + command_code) & 0xFF)) & 0xFF
    return bytes((0xAA, vehicle_id, command_code, checksum, 0xDD))


class ResultFileWatcher(threading.Thread):
    """独立监听结果文件，避免批量发送车辆指令时漏掉后续判决。"""

    def __init__(
        self,
        path: Path,
        output_queue: queue.Queue,
        poll_interval: float,
        process_existing: bool = False,
    ) -> None:
        super().__init__(name="result_file_watcher", daemon=True)
        self.path = Path(path)
        self.output_queue = output_queue
        self.poll_interval = float(poll_interval)
        self.process_existing = bool(process_existing)
        self.exit_event = threading.Event()
        self.last_mtime_ns = 0

    def stop(self) -> None:
        self.exit_event.set()

    def _read_label(self) -> Optional[int]:
        for _ in range(5):
            try:
                text = self.path.read_text(encoding="utf-8", errors="ignore").strip()
            except (FileNotFoundError, PermissionError, OSError):
                time.sleep(0.01)
                continue
            if re.fullmatch(r"\d+", text):
                return int(text)
            time.sleep(0.01)
        return None

    def run(self) -> None:
        if self.path.exists() and not self.process_existing:
            self.last_mtime_ns = self.path.stat().st_mtime_ns

        while not self.exit_event.is_set():
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                self.exit_event.wait(self.poll_interval)
                continue

            if stat.st_mtime_ns != self.last_mtime_ns:
                label = self._read_label()
                if label is not None:
                    self.last_mtime_ns = stat.st_mtime_ns
                    self.output_queue.put(label)
            self.exit_event.wait(self.poll_interval)


class DryRunSerial:
    """不连接硬件的协议联调串口。"""

    def write(self, frame: bytes) -> int:
        return len(frame)

    def flush(self) -> None:
        return None


class VehicleCommandFlow:
    """维护与刺激端相同的DCMM状态，并分发车辆串口命令。"""

    def __init__(
        self,
        serial_port,
        flow_config: dict,
        stim_config: dict,
        status_config: dict,
    ) -> None:
        self.serial_port = serial_port
        self.flow_config = flow_config
        self.command_codes: Dict[str, int] = {
            str(name): int(value)
            for name, value in flow_config["command_codes"].items()
        }
        self.stimulus_commands = {
            str(label): str(command)
            for label, command in flow_config["stimulus_commands"].items()
        }
        self.gear_commands = {
            int(gear): str(command)
            for gear, command in flow_config["gear_commands"].items()
        }
        self.special_commands = {
            str(label): str(command)
            for label, command in flow_config["special_commands"].items()
        }
        self.send_gap = float(flow_config["serial"]["send_gap"])
        self.max_vehicles_per_group = int(flow_config["max_vehicles_per_group"])

        self.mapper = DynamicContentMapper(stim_config)
        initial_state = status_config["初始状态"]
        self.mapper.control_scope = str(initial_state.get("控制范围", "单车"))
        self.mapper.selected_group = str(initial_state.get("组号", "A"))
        self.mapper.selected_vehicle = str(initial_state.get("小车号", "A1"))
        self.mapper.gear = int(initial_state.get("挡位", 1))

        group_count = int(stim_config["车辆组数"])
        group_names = tuple("ABCDEFGH")[:group_count]
        counts = stim_config["每组车辆数"]
        self.group_vehicle_ids = {
            group: [
                self.vehicle_name_to_id("{}{}".format(group, number))
                for number in range(1, int(counts[group]) + 1)
            ]
            for group in group_names
        }
        self.all_vehicle_ids = [
            vehicle_id
            for group in group_names
            for vehicle_id in self.group_vehicle_ids[group]
        ]

    def vehicle_name_to_id(self, vehicle_name: str) -> int:
        match = re.fullmatch(r"([A-H])(\d)", str(vehicle_name).upper())
        if match is None:
            raise ValueError("无效车辆名称：{}".format(vehicle_name))
        group = ord(match.group(1)) - ord("A")
        number = int(match.group(2))
        if not 1 <= number <= self.max_vehicles_per_group:
            raise ValueError("组内车辆编号必须在1-{}之间".format(self.max_vehicles_per_group))
        return group * self.max_vehicles_per_group + number

    def current_targets(self) -> List[int]:
        if self.mapper.control_scope == "全体":
            return list(self.all_vehicle_ids)
        if self.mapper.control_scope == "组":
            return list(self.group_vehicle_ids.get(self.mapper.selected_group, []))
        if self.mapper.selected_vehicle:
            return [self.vehicle_name_to_id(self.mapper.selected_vehicle)]
        return []

    def send_frame(self, vehicle_id: int, command_name: str) -> None:
        if command_name not in self.command_codes:
            raise KeyError("未配置车辆命令：{}".format(command_name))
        frame = build_vehicle_frame(vehicle_id, self.command_codes[command_name])
        self.serial_port.write(frame)
        self.serial_port.flush()
        print(
            "[车辆发送] ID={}，命令={}，帧={}".format(
                vehicle_id, command_name, frame.hex(" ").upper()
            ),
            flush=True,
        )
        time.sleep(self.send_gap)

    def send_command(self, command_name: str, targets: Iterable[int]) -> None:
        targets = list(targets)
        if not targets:
            print("[车辆发送] 当前没有有效控制目标，忽略命令={}".format(command_name))
            return
        for vehicle_id in targets:
            self.send_frame(vehicle_id, command_name)

    def initialize_all(self) -> None:
        for vehicle_id in self.all_vehicle_ids:
            self.send_frame(vehicle_id, "stop")
            self.send_frame(vehicle_id, "speed_mid")

    def handle_label(self, label: int) -> None:
        label = int(label)
        if label == 0:
            return
        if label == 21:
            print("[车辆流程] 收到判决失败标签21，本轮不发送车辆命令", flush=True)
            return
        if not 1 <= label <= 20:
            print("[车辆流程] 忽略未知标签={}".format(label), flush=True)
            return

        result = self.mapper.handle_prediction(label - 1)
        print("[车辆流程] 判决标签={}，DCMM结果={}".format(label, result), flush=True)
        action = result["动作"]

        if action == "车辆控制":
            command_name = self.stimulus_commands.get(result["命令"])
            if command_name is None:
                print("[车辆流程] 未配置刺激命令映射：{}".format(result["命令"]))
                return
            self.send_command(command_name, self.current_targets())
        elif action == "切换挡位":
            command_name = self.gear_commands.get(self.mapper.gear)
            if command_name is None:
                print("[车辆流程] 未配置挡位{}的车辆命令".format(self.mapper.gear))
                return
            self.send_command(command_name, self.current_targets())
        elif action == "特殊功能":
            special = self.special_commands.get(result["命令"])
            if special == "stop_all":
                self.send_command("stop", self.all_vehicle_ids)
            elif special == "initialize_all":
                self.initialize_all()
            else:
                print("[车辆流程] 未配置特殊功能：{}".format(result["命令"]))
        else:
            # 进入组、选择车辆、选择全体、返回和空刺激只改变DCMM状态。
            print("[车辆流程] 状态已更新，本轮不发送串口运动命令", flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP动态映射56车串口指令发送程序")
    parser.add_argument("--flow-config", default=str(DEFAULT_FLOW_CONFIG))
    parser.add_argument("--stim-config", default=str(DEFAULT_STIM_CONFIG))
    parser.add_argument("--status-config", default=str(DEFAULT_STATUS_CONFIG))
    parser.add_argument("--result-file", default=str(DEFAULT_RESULT_FILE))
    parser.add_argument("--port", default=None, help="覆盖配置中的串口，例如COM3")
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--process-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只打印协议帧，不打开串口")
    return parser


def main(args: argparse.Namespace) -> None:
    flow_config = load_json(Path(args.flow_config))
    stim_config = load_json(Path(args.stim_config))
    status_config = load_json(Path(args.status_config))
    serial_config = flow_config["serial"]
    port_name = args.port or serial_config["port"]
    baudrate = args.baudrate or int(serial_config["baudrate"])
    result_queue = queue.Queue()
    watcher = ResultFileWatcher(
        Path(args.result_file),
        result_queue,
        poll_interval=float(flow_config["result_poll_interval"]),
        process_existing=args.process_existing,
    )

    print("[车辆流程] 结果文件={}".format(args.result_file))
    print("[车辆流程] 串口={}，波特率={}，dry_run={}".format(port_name, baudrate, args.dry_run))

    serial_port = None
    try:
        if args.dry_run:
            serial_port = DryRunSerial()
        else:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError(
                    "真实串口模式需要安装pyserial：pip install pyserial"
                ) from exc
            serial_port = serial.Serial(
                port_name,
                baudrate,
                timeout=float(serial_config["timeout"]),
            )
            time.sleep(float(serial_config["startup_delay"]))

        command_flow = VehicleCommandFlow(
            serial_port, flow_config, stim_config, status_config
        )
        print(
            "[车辆流程] 初始控制范围={}，车辆={}，挡位={}".format(
                command_flow.mapper.control_scope,
                command_flow.mapper.selected_vehicle,
                command_flow.mapper.gear,
            ),
            flush=True,
        )
        watcher.start()
        while True:
            try:
                label = result_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            command_flow.handle_label(label)
    except KeyboardInterrupt:
        print("\n[车辆流程] 收到退出指令", flush=True)
    finally:
        watcher.stop()
        if watcher.is_alive():
            watcher.join(timeout=1.0)
        if serial_port is not None and not args.dry_run:
            serial_port.close()
        print("[车辆流程] 程序已退出", flush=True)


if __name__ == "__main__":
    main(make_parser().parse_args())
