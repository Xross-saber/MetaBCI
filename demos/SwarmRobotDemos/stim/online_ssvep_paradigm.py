"""SwarmRobotDemos 使用的非阻塞在线 SSVEP 范式。"""

from copy import copy
from time import perf_counter

import numpy as np
from psychopy import data, event
from pylsl import StreamInlet, resolve_byprop

from metabci.brainstim.utils import NeuraclePort, NeuroScanPort


def online_ssvep_paradigm(
    VSObject,
    win,
    bg_color,
    display_time=1.0,
    index_time=1.0,
    rest_time=0.5,
    response_time=2.0,
    port_addr=None,
    nrep=1,
    pdim="ssvep",
    lsl_source_id=None,
    online=False,
    device_type="NeuroScan",
    prediction_handler=None,
    prediction_timeout=5.0,
    lsl_connect_timeout=30.0,
):
    """运行 SSVEP，并以有超时的方式发送 trigger、接收预测。"""
    if pdim != "ssvep":
        raise ValueError("SwarmRobotDemos 的在线范式仅支持 SSVEP")

    win.color = bg_color
    fps = VSObject.refresh_rate
    if device_type == "NeuroScan":
        port = NeuroScanPort(port_addr, use_serial=True) if port_addr else None
    elif device_type == "Neuracle":
        port = None
        if port_addr:
            # 模拟板运行时会发布并响应“虚拟 COM4”状态。此时原始刺激
            # 服务无需改名或改参数，自动通过 UDP 桥发送 240/241；没有
            # 模拟板时仍打开真实串口，因此真实硬件流程保持不变。
            if str(port_addr).upper() == "COM4":
                try:
                    from demos.SwarmRobotDemos.stim.NewFunc2.virtual_com4_online_ssvep import (
                        VirtualNeuraclePort,
                    )

                    stimulus_seconds = float(VSObject.stim_frames) / float(fps)
                    port = VirtualNeuraclePort(
                        port_addr,
                        stimulus_seconds=stimulus_seconds,
                    )
                    print(
                        "[刺激端][触发] 检测到模拟板，COM4 已由虚拟 Trigger Box 占用",
                        flush=True,
                    )
                except RuntimeError:
                    port = None
            if port is None:
                port = NeuraclePort(port_addr)
        if port is not None:
            # 串口写发生在 win.flip() 回调中，必须限制阻塞时间。
            port.port.write_timeout = 0.1
    else:
        raise KeyError("Unknown device type: {}".format(device_type))
    port_frame = int(0.05 * fps)
    trigger_failed = False
    status_display = getattr(VSObject, "online_status_display", None)

    def draw_default_response():
        if online and status_display is None:
            VSObject.rect_response.draw()
            VSObject.text_response.draw()

    def draw_target_interface(draw_feedback=False):
        background = getattr(VSObject, "static_stimuli", None)
        if background is not None:
            background.draw()
        for text_stimulus in VSObject.text_stimuli:
            text_stimulus.draw()
        if online and status_display is not None:
            status_display.draw_header()
            if draw_feedback:
                status_display.draw_feedback()

    def send_trigger(label):
        nonlocal trigger_failed
        if port is None or trigger_failed:
            return
        try:
            port.setData(label)
            if label != 0 or device_type != "Neuracle":
                print(
                    "[刺激端][触发] 已在翻屏时刻发送：串口={}，设备={}，Trigger={}".format(
                        port_addr, device_type, label
                    ),
                    flush=True,
                )
        except Exception as error:
            trigger_failed = True
            print(
                "[刺激端][触发] 串口写入失败，后续Trigger发送已停用：{}: {}".format(
                    type(error).__name__, error
                )
            )

    inlet = None
    if online:
        VSObject.text_response.text = copy(VSObject.reset_res_text)
        VSObject.text_response.pos = copy(VSObject.reset_res_pos)
        VSObject.res_text_pos = copy(VSObject.reset_res_pos)
        VSObject.symbol_text = copy(VSObject.reset_res_text)
        response_position = VSObject.reset_res_pos
        if lsl_source_id:
            print(
                "[刺激端][回传] 正在查找LSL流，source_id={!r}".format(lsl_source_id),
                flush=True,
            )
            streams = resolve_byprop(
                "source_id", lsl_source_id, timeout=float(lsl_connect_timeout)
            )
            if not streams:
                print("[刺激端][回传] 未找到匹配的LSL流", flush=True)
                return
            inlet = StreamInlet(streams[0])
            print("[刺激端][回传] 已连接算法端LSL流", flush=True)

    conditions = [{"id": index} for index in range(VSObject.n_elements)]
    trials = data.TrialHandler(
        conditions,
        nrep,
        name="experiment",
        method="random",
    )

    frame = 0
    while frame < int(fps * display_time):
        draw_default_response()
        draw_target_interface()
        frame += 1
        win.flip()

    send_trigger(0)

    for trial in trials:
        if "q" in event.getKeys(["q"]):
            break

        target_index = int(trial["id"])
        index_position = VSObject.stim_pos[target_index] + np.array(
            [0, VSObject.stim_width / 2]
        )
        VSObject.index_stimuli.setPos(index_position)

        frame = 0
        while frame < int(fps * index_time):
            draw_default_response()
            draw_target_interface()
            if not (online and status_display is not None):
                VSObject.index_stimuli.draw()
            frame += 1
            win.flip()

        frame = 0
        while frame < int(fps * rest_time):
            draw_default_response()
            draw_target_interface()
            frame += 1
            win.flip()

        # 清理上一轮迟到的结果，避免它被误认为当前刺激的预测。
        if inlet is not None:
            inlet.flush()

        for stimulus_frame in range(VSObject.stim_frames):
            if stimulus_frame == 0 and port:
                VSObject.win.callOnFlip(send_trigger, target_index + 1)
            if stimulus_frame == port_frame and port:
                send_trigger(0)
            VSObject.flash_stimuli[stimulus_frame].draw()
            if getattr(VSObject, "draw_text_during_stimulation", False):
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
            if online and status_display is not None:
                status_display.draw_header()
            win.flip()

        if inlet is None:
            continue

        samples = None
        deadline = perf_counter() + float(prediction_timeout)
        while samples is None and perf_counter() < deadline:
            samples, _ = inlet.pull_sample(timeout=0.0)
            if samples is not None:
                break
            if "q" in event.getKeys(["q"]):
                return
            draw_default_response()
            draw_target_interface()
            win.flip()

        if samples is None:
            print(
                "[刺激端][回传] 等待判决结果超过{:.1f}秒，请检查Trigger通道和"
                "FBSCCA算法端输出。".format(prediction_timeout)
            )
            continue

        received_label = int(samples[0])
        failure_label = VSObject.n_elements + 1
        if received_label == failure_label:
            print(
                "[刺激端][回传] 收到失败标签={}，本轮不执行命令、不更新动态映射".format(
                    failure_label
                ),
                flush=True,
            )
            continue

        predict_index = received_label - 1
        if not 0 <= predict_index < VSObject.n_elements:
            raise ValueError("在线算法回传类别超出范围: {}".format(received_label))

        selected_label = VSObject.symbols[predict_index]
        print(
            "[刺激端][回传] 收到判决：类别={}，内部索引={}，当前文字={!r}".format(
                samples[0], predict_index, selected_label
            ),
            flush=True,
        )
        VSObject.symbol_text += selected_label
        if prediction_handler is not None:
            prediction_handler(predict_index, selected_label)
        if online and status_display is not None:
            status_display.set_feedback(predict_index)

        response_position = (
            response_position[0] + VSObject.symbol_height / 3,
            response_position[1],
        )
        frame = 0
        while frame < int(fps * response_time):
            draw_target_interface(
                draw_feedback=online and status_display is not None
            )
            if status_display is None:
                VSObject.rect_response.draw()
                VSObject.text_response.text = VSObject.symbol_text
                VSObject.text_response.pos = response_position
                VSObject.text_response.draw()
            frame += 1
            win.flip()
        if online and status_display is not None:
            status_display.clear_feedback()
