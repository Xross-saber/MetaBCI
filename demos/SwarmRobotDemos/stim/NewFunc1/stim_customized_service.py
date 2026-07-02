import math
import sys
from pathlib import Path

from psychopy import monitors
import numpy as np

STIM_DIR = Path(__file__).resolve().parents[1]
if str(STIM_DIR) not in sys.path:
    sys.path.insert(0, str(STIM_DIR))

from metabci.brainstim.paradigm import SSVEP
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix
import json
from demos.SwarmRobotDemos.stim.online_ssvep_paradigm import online_ssvep_paradigm

if __name__ == "__main__":
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    mon.setSizePix([1920, 1080])  # 显示器的分辨率
    mon.save()
    bg_color_warm = np.array([-1, -1, -1])
    win_size = np.array([1440, 960])
    # esc/q退出开始选择界面
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=False,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()

    # q退出范式界面
    """
    SSVEP
    """
    config_path = Path(__file__).with_name("config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 使用配置
    n_elements, rows, columns = config['n_elements'], config['rows'], config['columns']  # n_elements 指令数量;  rows 行;  columns 列
    fps = config['fps']  # 屏幕刷新率
    stim_time = config['stim_time']  # 刺激时长
    target_names = config.get("选中目标", [])
    if not isinstance(target_names, list):
        raise ValueError("config.json 中的 '选中目标' 必须是列表")
    target_names = [str(name).strip() for name in target_names if str(name).strip()]
    if len(target_names) < n_elements:
        raise ValueError(
            f"config.json 中的刺激块名称数量不足：需要 {n_elements} 个，实际 {len(target_names)} 个"
        )
    target_names = target_names[:n_elements]

    stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
    stim_length, stim_width = 100, 100  # ssvep单指令的尺寸
    stim_opacities = 1  # 刺激对比度
    freqs = np.arange(8, (8+n_elements*0.2), 0.2)  # 指令的频率
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位

    basic_ssvep = SSVEP(win=win)

    basic_ssvep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_ssvep.config_text(symbols=target_names, tex_color=tex_color)
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

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 2  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = "COM12"  #  0xdefc                                  # 采集主机端口
    # port_addr = None  #  0xdefc
    nrep = 2  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    device_type = "Neuracle"  # COM12 连接博睿康 trigger box
    # online = True
    ex.register_paradigm(
        "进入操控界面",
        online_ssvep_paradigm,
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
        device_type=device_type,
    )

    ex.run()
