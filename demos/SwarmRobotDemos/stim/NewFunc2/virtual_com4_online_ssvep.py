# -*- coding: utf-8 -*-
"""Virtual COM4 adapter for the BrainCo mock board.

The normal stimulus paradigm still calls ``NeuraclePort.setData(target)`` at
the first flicker frame and ``setData(0)`` after the short marker pulse.  This
adapter translates the first call into trigger 240 and schedules trigger 241
for the end of the visual stimulation.  The mock board itself chooses the
synthetic command from its configured 4,4,1,2 sequence.
"""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import Optional

NEURACLE_TRIGGER_HEADER = bytes((0x01, 0xE1, 0x01, 0x00))
VIRTUAL_COM_STATUS = Path(__file__).resolve().parents[2] / "brainco_mock_com4.json"
VIRTUAL_COM_PING = b"BRAINCO_MOCK_COM4_PING"
VIRTUAL_COM_PONG = b"BRAINCO_MOCK_COM4_READY"


class VirtualNeuraclePort:
    """Small ``NeuraclePort``-compatible UDP client named COM4."""

    def __init__(
        self,
        port_addr: str,
        stimulus_seconds: float,
        host: str = "127.0.0.1",
        trigger_port: int = 8713,
    ) -> None:
        if str(port_addr).upper() != "COM4":
            raise ValueError("the mock adapter only represents COM4")
        if stimulus_seconds <= 0:
            raise ValueError("stimulus_seconds must be positive")
        self.port_addr = str(port_addr)
        self.stimulus_seconds = float(stimulus_seconds)
        self.host = host
        self.trigger_port = int(trigger_port)
        self.port = self  # online_ssvep_paradigm sets port.port.write_timeout
        self.write_timeout = 0.1
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()
        self._active = False
        self._stop_timer: Optional[threading.Timer] = None
        self._verify_occupancy()

    def _verify_occupancy(self) -> None:
        try:
            status = json.loads(VIRTUAL_COM_STATUS.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "virtual COM4 is unavailable; start brainco_mock_board.py first"
            ) from exc
        if (
            status.get("port", "").upper() != self.port_addr.upper()
            or status.get("state") != "occupied"
        ):
            raise RuntimeError("virtual COM4 is not occupied by the mock board")
        # Follow the endpoint advertised by the running board rather than
        # silently sending to stale constructor defaults.
        self.host = str(status.get("host", self.host))
        self.trigger_port = int(status.get("trigger_port", self.trigger_port))
        target_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        previous_timeout = self._socket.gettimeout()
        try:
            self._socket.settimeout(0.5)
            self._socket.sendto(VIRTUAL_COM_PING, (target_host, self.trigger_port))
            response, _ = self._socket.recvfrom(64)
        except OSError as exc:
            raise RuntimeError(
                "virtual COM4 status is stale; brainco_mock_board.py is not responding"
            ) from exc
        finally:
            self._socket.settimeout(previous_timeout)
        if response != VIRTUAL_COM_PONG:
            raise RuntimeError("virtual COM4 returned an invalid handshake response")
        self.host = target_host

    @staticmethod
    def _frame(trigger: int) -> bytes:
        trigger = int(trigger)
        if not 0 <= trigger <= 255:
            raise ValueError("Neuracle trigger must fit in one byte")
        return NEURACLE_TRIGGER_HEADER + bytes((trigger,))

    def _send(self, trigger: int) -> None:
        self._socket.sendto(self._frame(trigger), (self.host, self.trigger_port))
        print(
            "[刺激端][虚拟COM4] 已发送 Trigger={} -> udp://{}:{}".format(
                trigger, self.host, self.trigger_port
            ),
            flush=True,
        )

    def setData(self, label: int) -> None:  # noqa: N802 - MetaBCI API name
        # Zero is merely the end of the 50-ms hardware marker pulse in the
        # original paradigm.  The real trial stop is scheduled at stim end.
        if int(label) == 0:
            return
        with self._lock:
            if self._active:
                raise RuntimeError("virtual COM4 received a new trial before trigger 241")
            self._verify_occupancy()
            self._send(240)
            self._active = True
            self._stop_timer = threading.Timer(self.stimulus_seconds, self._finish_trial)
            self._stop_timer.daemon = True
            self._stop_timer.start()

    def _finish_trial(self) -> None:
        with self._lock:
            if not self._active:
                return
            try:
                self._send(241)
            finally:
                self._active = False
                self._stop_timer = None

    def close(self) -> None:
        with self._lock:
            timer = self._stop_timer
            self._stop_timer = None
            if timer is not None:
                timer.cancel()
            if self._active:
                try:
                    self._send(241)
                finally:
                    self._active = False
            self._socket.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass


def virtual_com4_online_ssvep_paradigm(
    *args,
    virtual_trigger_host: str = "127.0.0.1",
    virtual_trigger_port: int = 8713,
    **kwargs,
):
    """Run the existing online paradigm through the virtual COM4 adapter."""

    # Keep the transport adapter importable for protocol tests on machines
    # that do not have the PsychoPy/LSL GUI dependencies installed.
    from demos.SwarmRobotDemos.stim import online_ssvep_paradigm as paradigm_module

    vs_object = kwargs.get("VSObject")
    if vs_object is None and args:
        vs_object = args[0]
    if vs_object is None:
        raise ValueError("VSObject is required")
    stimulus_seconds = float(vs_object.stim_frames) / float(vs_object.refresh_rate)
    instances = []

    def port_factory(port_addr):
        instance = VirtualNeuraclePort(
            port_addr,
            stimulus_seconds=stimulus_seconds,
            host=virtual_trigger_host,
            trigger_port=virtual_trigger_port,
        )
        instances.append(instance)
        return instance

    original_port_class = paradigm_module.NeuraclePort
    paradigm_module.NeuraclePort = port_factory
    try:
        return paradigm_module.online_ssvep_paradigm(*args, **kwargs)
    finally:
        paradigm_module.NeuraclePort = original_port_class
        for instance in instances:
            instance.close()
