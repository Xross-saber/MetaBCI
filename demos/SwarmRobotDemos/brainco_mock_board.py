# -*- coding: utf-8 -*-
"""BrainCo/Neuracle mock board used by the SwarmRobot online demo.

The process exposes the same raw TCP stream consumed by
``metabci.brainflow.amplifiers.Neuracle``.  A small UDP bridge represents the
otherwise absent COM4 trigger box and accepts both ASCII trigger numbers and
the five-byte Neuracle serial frame (``01 E1 01 00 XX``).

Trial protocol
--------------
* 240 starts acquisition for the next command in ``4, 4, 1, 2``.
* 241 stops acquisition and releases the buffered trial to the TCP client.
* The released trigger channel contains the selected command (1-20), because
  that is the event expected by the online Marker/FBSCCA pipeline.

The virtual COM bridge is intentionally application-level.  Creating a real
Windows COM device requires a kernel driver such as com0com; the companion
``stim_customized_service_mock.py`` talks to this bridge without such a
machine-wide installation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


OCCIPITAL_CHS = ["PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"]
START_TRIGGER = 240
STOP_TRIGGER = 241
DEFAULT_COMMAND_SEQUENCE = (4, 4, 1, 2)
NEURACLE_TRIGGER_HEADER = bytes((0x01, 0xE1, 0x01, 0x00))
VIRTUAL_COM_STATUS = Path(__file__).with_name("brainco_mock_com4.json")
VIRTUAL_COM_PING = b"BRAINCO_MOCK_COM4_PING"
VIRTUAL_COM_PONG = b"BRAINCO_MOCK_COM4_READY"


def command_frequency(command: int) -> float:
    """Return the physical SSVEP frequency for a 1-based command number."""

    if not 1 <= int(command) <= 20:
        raise ValueError("command must be in the range 1..20")
    return 8.0 + 0.2 * (int(command) - 1)


def parse_trigger_payload(payload: bytes) -> int:
    """Decode an ASCII trigger or a Neuracle trigger-box serial frame."""

    if len(payload) >= 5 and payload[:4] == NEURACLE_TRIGGER_HEADER:
        return int(payload[4])
    try:
        return int(payload.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("unsupported trigger payload {!r}".format(payload)) from exc


def parse_command_sequence(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        commands = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        commands = tuple(int(item) for item in value)
    if not commands:
        raise ValueError("command sequence cannot be empty")
    invalid = [command for command in commands if not 1 <= command <= 20]
    if invalid:
        raise ValueError("command sequence values must be in 1..20: {}".format(invalid))
    return commands


class MockBrainCoBoard:
    """Buffered trial simulator compatible with MetaBCI's Neuracle client."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        data_port: int = 8712,
        trigger_port: int = 8713,
        srate: int = 1000,
        eeg_chans: int = 64,
        packet_samples: int = 40,
        command_sequence: Sequence[int] = DEFAULT_COMMAND_SEQUENCE,
        minimum_trial_seconds: float = 1.2,
        virtual_com_name: str = "COM4",
        verbose_stream: bool = False,
    ) -> None:
        self.host = host
        self.data_port = int(data_port)
        self.trigger_port = int(trigger_port)
        self.srate = int(srate)
        self.eeg_chans = int(eeg_chans)
        self.num_chans = self.eeg_chans + 1
        self.packet_samples = int(packet_samples)
        self.command_sequence = parse_command_sequence(command_sequence)
        self.minimum_trial_samples = int(round(float(minimum_trial_seconds) * self.srate))
        self.virtual_com_name = str(virtual_com_name)
        self.verbose_stream = bool(verbose_stream)

        if self.srate <= 0 or self.eeg_chans <= 0 or self.packet_samples <= 0:
            raise ValueError("srate, eeg_chans and packet_samples must be positive")

        self._exit = threading.Event()
        self._state_lock = threading.RLock()
        self._ready_trials: queue.Queue[tuple[int, int, np.ndarray]] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._server: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._trigger_socket: Optional[socket.socket] = None

        self._trial_active = False
        self._trial_number = 0
        self._sequence_index = 0
        self._current_command: Optional[int] = None
        self._trial_sample_index = 0
        self._trial_packets: list[np.ndarray] = []
        self._trial_started_at = 0.0
        self._rng = np.random.default_rng(20260702)

    def start(self) -> None:
        """Bind both services synchronously, then start acquisition threads."""

        self._exit.clear()
        try:
            trigger_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            trigger_socket.bind((self.host, self.trigger_port))
            trigger_socket.settimeout(0.2)
            self._trigger_socket = trigger_socket

            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.data_port))
            server.listen(1)
            server.settimeout(0.2)
            self._server = server
        except Exception:
            self._close_sockets()
            raise

        self._write_virtual_com_status("occupied")
        self._threads = [
            threading.Thread(target=self._trigger_loop, name="virtual_com4", daemon=True),
            threading.Thread(target=self._acquisition_loop, name="mock_acquisition", daemon=True),
            threading.Thread(target=self._data_loop, name="neuracle_data_service", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

        print(
            "Mock BrainCo board online: data=tcp://{}:{}, channels={} ({} EEG + Trigger)".format(
                self.host, self.data_port, self.num_chans, self.eeg_chans
            ),
            flush=True,
        )
        print(
            "virtual {} occupied: udp://{}:{}; trigger protocol 240=start, 241=stop".format(
                self.virtual_com_name, self.host, self.trigger_port
            ),
            flush=True,
        )
        print(
            "synthetic command sequence: {}".format(
                " -> ".join(str(command) for command in self.command_sequence)
            ),
            flush=True,
        )

    def stop(self) -> None:
        self._exit.set()
        self._close_sockets()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._write_virtual_com_status("available")

    def _close_sockets(self) -> None:
        for sock in (self._client, self._server, self._trigger_socket):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._client = None
        self._server = None
        self._trigger_socket = None

    def _write_virtual_com_status(self, state: str, **extra) -> None:
        status = {
            "port": self.virtual_com_name,
            "state": state,
            "transport": "udp",
            "host": self.host,
            "trigger_port": self.trigger_port,
            "pid": os.getpid(),
            "updated_at": time.time(),
        }
        status.update(extra)
        temporary = VIRTUAL_COM_STATUS.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(VIRTUAL_COM_STATUS)
        except OSError as exc:
            print("warning: cannot update virtual COM status: {}".format(exc), flush=True)

    def push_trigger(self, trigger: int) -> None:
        """Inject a control trigger, also used by the interactive CLI."""

        trigger = int(trigger)
        if trigger == START_TRIGGER:
            self._start_trial()
        elif trigger == STOP_TRIGGER:
            self._stop_trial()
        else:
            print("ignored control trigger {} (expected 240 or 241)".format(trigger), flush=True)

    def _trigger_loop(self) -> None:
        udp = self._trigger_socket
        if udp is None:
            return
        while not self._exit.is_set():
            try:
                payload, address = udp.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            if payload == VIRTUAL_COM_PING:
                try:
                    udp.sendto(VIRTUAL_COM_PONG, address)
                except OSError:
                    break
                continue
            try:
                trigger = parse_trigger_payload(payload)
            except ValueError as exc:
                print("ignored invalid virtual COM payload from {}: {}".format(address, exc), flush=True)
                continue
            print(
                "virtual {} received trigger {} from {}:{}".format(
                    self.virtual_com_name, trigger, address[0], address[1]
                ),
                flush=True,
            )
            self.push_trigger(trigger)

    def _start_trial(self) -> None:
        with self._state_lock:
            if self._trial_active:
                print("trigger 240 ignored: a trial is already active", flush=True)
                return
            command = self.command_sequence[self._sequence_index]
            self._sequence_index = (self._sequence_index + 1) % len(self.command_sequence)
            self._trial_number += 1
            self._current_command = command
            self._trial_sample_index = 0
            self._trial_packets = []
            self._trial_started_at = time.perf_counter()
            self._trial_active = True
            self._write_virtual_com_status(
                "occupied",
                trial_state="acquiring",
                trial=self._trial_number,
                command=command,
                frequency_hz=command_frequency(command),
            )
            print(
                "trial {} started: command={}, frequency={:.1f} Hz".format(
                    self._trial_number, command, command_frequency(command)
                ),
                flush=True,
            )

    def _stop_trial(self) -> None:
        with self._state_lock:
            if not self._trial_active or self._current_command is None:
                print("trigger 241 ignored: no active trial", flush=True)
                return

            command = self._current_command
            while self._trial_sample_index < self.minimum_trial_samples:
                self._trial_packets.append(self._make_trial_packet(command))

            trial_data = np.concatenate(self._trial_packets, axis=0)
            trial_number = self._trial_number
            elapsed = time.perf_counter() - self._trial_started_at
            self._trial_active = False
            self._current_command = None
            self._trial_packets = []
            self._write_virtual_com_status(
                "occupied",
                trial_state="buffered",
                trial=trial_number,
                command=command,
                samples=len(trial_data),
            )

        self._ready_trials.put((trial_number, command, trial_data))
        print(
            "trial {} stopped after {:.3f}s: buffered {} samples for command {}; queued for TCP".format(
                trial_number, elapsed, len(trial_data), command
            ),
            flush=True,
        )

    def _acquisition_loop(self) -> None:
        period = self.packet_samples / float(self.srate)
        next_tick = time.perf_counter()
        while not self._exit.is_set():
            now = time.perf_counter()
            if now < next_tick:
                self._exit.wait(next_tick - now)
                continue
            with self._state_lock:
                if self._trial_active and self._current_command is not None:
                    self._trial_packets.append(self._make_trial_packet(self._current_command))
            next_tick += period
            if next_tick < now - period:
                next_tick = now + period

    def _make_trial_packet(self, command: int) -> np.ndarray:
        start = self._trial_sample_index
        stop = start + self.packet_samples
        t = np.arange(start, stop, dtype=np.float64) / float(self.srate)
        data = np.zeros((self.packet_samples, self.num_chans), dtype=np.float32)
        frequency = command_frequency(command)
        stimulus_phase = ((command - 1) * 0.35 % 2.0) * math.pi

        # Similar SSVEP content on all channels, with channel-specific gains,
        # phases, harmonics and low background noise.
        for channel in range(self.eeg_chans):
            gain = 18.0 + 3.0 * math.cos(2.0 * math.pi * channel / max(1, self.eeg_chans))
            spatial_phase = 0.025 * (channel % 8)
            fundamental = gain * np.sin(
                2.0 * math.pi * frequency * t + stimulus_phase + spatial_phase
            )
            harmonic = 0.35 * gain * np.sin(
                2.0 * math.pi * 2.0 * frequency * t + 2.0 * stimulus_phase + spatial_phase
            )
            background = 1.2 * np.sin(2.0 * math.pi * 6.5 * t + channel * 0.11)
            noise = self._rng.normal(0.0, 0.8, size=self.packet_samples)
            data[:, channel] = fundamental + harmonic + background + noise

        pulse_samples = max(1, int(round(0.02 * self.srate)))
        if start < pulse_samples:
            pulse_end = min(self.packet_samples, pulse_samples - start)
            data[:pulse_end, -1] = float(command)

        self._trial_sample_index = stop
        return data

    def _data_loop(self) -> None:
        server = self._server
        if server is None:
            return
        while not self._exit.is_set():
            try:
                client, address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._client = client
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("MetaBCI Neuracle client connected from {}:{}".format(*address), flush=True)
            try:
                self._send_stream(client)
            finally:
                try:
                    client.close()
                except OSError:
                    pass
                self._client = None
                print("MetaBCI Neuracle client disconnected", flush=True)

    def _send_stream(self, client: socket.socket) -> None:
        packet_period = self.packet_samples / float(self.srate)
        while not self._exit.is_set():
            try:
                trial_number, command, trial_data = self._ready_trials.get(timeout=0.2)
            except queue.Empty:
                continue

            print(
                "sending trial {} to algorithm: command={}, samples={}, bytes={}".format(
                    trial_number, command, len(trial_data), trial_data.nbytes
                ),
                flush=True,
            )
            sent_samples = 0
            next_send = time.perf_counter()
            try:
                for offset in range(0, len(trial_data), self.packet_samples):
                    packet = trial_data[offset: offset + self.packet_samples]
                    now = time.perf_counter()
                    if now < next_send:
                        self._exit.wait(next_send - now)
                    client.sendall(packet.astype("<f4", copy=False).tobytes())
                    sent_samples += len(packet)
                    next_send += len(packet) / float(self.srate)
                    if self.verbose_stream and sent_samples % self.srate < self.packet_samples:
                        print(
                            "trial {} stream progress: {}/{} samples".format(
                                trial_number, sent_samples, len(trial_data)
                            ),
                            flush=True,
                        )
            except OSError:
                # Keep the complete trial available for a reconnect.  Replaying
                # from its marker is safer than resuming a partial epoch.
                self._ready_trials.put((trial_number, command, trial_data))
                break
            print("trial {} fully sent to algorithm".format(trial_number), flush=True)


def encode_neuracle_trigger(trigger: int) -> bytes:
    trigger = int(trigger)
    if not 0 <= trigger <= 255:
        raise ValueError("Neuracle trigger must fit in one byte")
    return NEURACLE_TRIGGER_HEADER + bytes((trigger,))


def send_trigger(host: str, trigger_port: int, trigger: int, binary: bool = False) -> None:
    payload = encode_neuracle_trigger(trigger) if binary else str(int(trigger)).encode("ascii")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.sendto(payload, (host, int(trigger_port)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mock BrainCo/Neuracle board; 240 starts and 241 releases a buffered trial"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data-port", type=int, default=8712)
    parser.add_argument("--trigger-port", type=int, default=8713)
    parser.add_argument("--virtual-com-name", default="COM4")
    parser.add_argument("--srate", type=int, default=1000)
    parser.add_argument(
        "--eeg-chans",
        type=int,
        default=64,
        help="64 by default, matching online_ssvep_neuracle_fbscca.py",
    )
    parser.add_argument("--packet-samples", type=int, default=40)
    parser.add_argument("--sequence", default="4,4,1,2")
    parser.add_argument(
        "--minimum-trial-seconds",
        type=float,
        default=1.2,
        help="pad very short trials so the algorithm's 0.14-1.14s epoch is complete",
    )
    parser.add_argument("--verbose-stream", action="store_true")
    parser.add_argument("--trigger", type=int, help="send one trigger to a running board")
    parser.add_argument(
        "--binary-trigger",
        action="store_true",
        help="with --trigger, send the real five-byte Neuracle serial frame",
    )
    args = parser.parse_args()

    if args.trigger is not None:
        send_trigger(args.host, args.trigger_port, args.trigger, binary=args.binary_trigger)
        return

    board = MockBrainCoBoard(
        host=args.host,
        data_port=args.data_port,
        trigger_port=args.trigger_port,
        srate=args.srate,
        eeg_chans=args.eeg_chans,
        packet_samples=args.packet_samples,
        command_sequence=parse_command_sequence(args.sequence),
        minimum_trial_seconds=args.minimum_trial_seconds,
        virtual_com_name=args.virtual_com_name,
        verbose_stream=args.verbose_stream,
    )
    board.start()
    try:
        while True:
            raw = input("control trigger (240=start, 241=stop), or q to quit> ").strip()
            if raw.lower() in {"q", "quit", "exit"}:
                break
            if raw:
                board.push_trigger(int(raw))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        board.stop()


if __name__ == "__main__":
    main()
