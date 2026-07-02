"""20 目标 UniSwarm-SSVEP 动态刺激界面。"""

import json
import sys
from pathlib import Path

import numpy as np
from psychopy import monitors, visual

STIM_DIR = Path(__file__).resolve().parents[1]
if str(STIM_DIR) not in sys.path:
    sys.path.insert(0, str(STIM_DIR))

from dynamic_content_mapping import DynamicContentMapper
from status_display import OnlineStatusDisplay, apply_initial_state, load_status_config
from metabci.brainstim.framework import Experiment
from metabci.brainstim.paradigm import SSVEP
from demos.SwarmRobotDemos.stim.online_ssvep_paradigm import online_ssvep_paradigm


def load_config():
    """从脚本所在目录读取配置，避免受启动目录影响。"""
    config_path = Path(__file__).with_name("config.json")
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def build_text_colors(config):
    """按物理区域生成 PsychoPy RGB 文字颜色。"""
    colors = [[1, 1, 1] for _ in range(int(config["n_elements"]))]
    interface = config["界面配置"]
    control = interface["车辆控制区"]
    special = interface["特殊功能区"]
    switching = interface["车辆切换区"]

    for stimulus_id, label in zip(control["刺激编号"], control["文字"]):
        colors[int(stimulus_id) - 1] = (
            [-1, -1, 1] if str(label).upper() == "S" else [1, -1, -1]
        )
    for stimulus_id, label in zip(special["刺激编号"], special["文字"]):
        colors[int(stimulus_id) - 1] = (
            [-1, 1, -1] if str(label).upper() == "ALL" else [1, 1, -1]
        )
    for stimulus_id in switching["刺激编号"]:
        colors[int(stimulus_id) - 1] = [-1, 1, -1]
    return colors


if __name__ == "__main__":
    config = load_config()
    mapper = DynamicContentMapper(config)
    status_config = load_status_config()
    apply_initial_state(mapper, status_config)

    monitor = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,
        verbose=False,
    )
    monitor.setSizePix([1920, 1080])
    monitor.save()

    experiment = Experiment(
        monitor=monitor,
        bg_color_warm=np.array([-1, -1, -1]),
        screen_id=0,
        win_size=np.array([1440, 960]),
        is_fullscr=True,
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = experiment.get_window()

    n_elements = int(config["n_elements"])
    rows = int(config["rows"])
    columns = int(config["columns"])
    fps = float(config["fps"])
    stim_time = float(config["stim_time"])

    # 物理刺激编码保持固定；DCMM 只改变文字和编号对应的逻辑语义。
    frequencies = 8.0 + 0.2 * np.arange(n_elements)
    phases = np.asarray([index * 0.35 % 2 for index in range(n_elements)])

    ssvep = SSVEP(win=win)
    ssvep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=130,
        stim_width=130,
    )
    ssvep.config_text(symbols=mapper.labels(),symbol_height=45, tex_color=[1, 1, 1])
    text_colors = build_text_colors(config)
    for text_stimulus, text_color in zip(ssvep.text_stimuli, text_colors):
        text_stimulus.setColor(text_color, colorSpace="rgb")
        if text_stimulus.text in {"↑", "↓", "←", "→"}:
            text_stimulus.setFont("Arial")
            text_stimulus.bold = True
            text_stimulus.setHeight(65)

    # 提示、休息和反馈阶段使用白色底块；闪烁阶段仍由亮度编码控制。
    ssvep.static_stimuli = visual.ElementArrayStim(
        win=win,
        units="pix",
        nElements=n_elements,
        sizes=ssvep.stim_sizes,
        xys=ssvep.stim_pos,
        colors=np.ones((n_elements, 3)),
        opacities=np.ones(n_elements),
        oris=np.zeros(n_elements),
        sfs=np.zeros(n_elements),
        contrs=np.ones(n_elements),
        elementTex=np.ones((64, 64)),
        elementMask=None,
        texRes=48,
    )
    ssvep.draw_text_during_stimulation = True
    ssvep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=[1, 1, 1],
        stim_opacities=1,
        freqs=frequencies,
        phases=phases,
    )
    ssvep.config_index()
    ssvep.config_response()
    ssvep.online_status_display = OnlineStatusDisplay(
        win, ssvep, mapper, status_config
    )

    def refresh_labels(labels):
        """原位更新文字；空字符串不会隐藏对应的闪烁刺激块。"""
        ssvep.symbols = list(labels)
        for text_stimulus, label in zip(ssvep.text_stimuli, labels):
            text_stimulus.text = label
            text_stimulus.name = label

    def handle_prediction(predict_index, selected_label):
        """LSL 收到 FBSCCA 类别后执行 DCMM 状态转换并刷新界面。"""
        result = mapper.handle_prediction(predict_index)
        refresh_labels(mapper.labels())
        print(
            "[动态映射] 刺激编号={刺激编号}，文字={文字!r}，动作={动作}，"
            "当前界面={当前界面}，控制范围={控制范围}，当前组={当前组}，"
            "当前车辆={当前车辆}，当前挡位={当前挡位}".format(**result)
        )

    experiment.register_paradigm(
        "进入操控界面",
        online_ssvep_paradigm,
        VSObject=ssvep,
        bg_color=np.array([-1, -1, -1]),
        display_time=1,
        index_time=2,
        rest_time=0.5,
        response_time=1,
        port_addr="COM12",
        nrep=2,
        pdim="ssvep",
        lsl_source_id="meta_online_worker",
        online=False,
        device_type="Neuracle",
        prediction_handler=handle_prediction,
        prediction_timeout=5.0,
        lsl_connect_timeout=30.0,
    )

    experiment.run()
