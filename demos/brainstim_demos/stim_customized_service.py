import math
import numpy as np
from psychopy import monitors
from demos.brainstim_demos.paradigm_trans import SSVEP, paradigm
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix
import json

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
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except UnicodeDecodeError as e:
        print(f"文件解码错误：{e}")
        print("尝试使用 UTF-8-SIG 编码...")
        try:
            with open('config.json', 'r', encoding='utf-8-sig') as f:
                config = json.load(f)
        except UnicodeDecodeError as e:
            print(f"仍然无法解码：{e}")
            print("请检查 config.json 的编码格式（推荐 UTF-8），或使用文本编辑器将其转换为 UTF-8 编码。")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误：{e}")
            print("请检查 config.json 文件内容是否为合法 JSON 格式。")
            exit(1)
    except FileNotFoundError:
        print("错误：未找到 config.json 文件，请确保文件存在于正确路径。")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误：{e}")
        print("请检查 config.json 文件内容是否为合法 JSON 格式。")
        exit(1)

    # 使用配置
    try:
        n_elements, rows, columns = config['n_elements'], config['rows'], config['columns']  # n_elements 指令数量; rows 行; columns 列
        fps = config['fps']  # 屏幕刷新率
        stim_time = config['stim_time']  # 刺激时长
    except KeyError as e:
        print(f"配置错误：缺少必要的字段 {e}，请检查 config.json 是否包含 'n_elements', 'rows', 'columns', 'fps', 'stim_time'。")
        exit(1)

    stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
    stim_length, stim_width = 100, 100  # ssvep单指令的尺寸
    stim_opacities = 1  # 刺激对比度
    freqs = np.arange(8, 8 + n_elements * 0.2, 0.2)  # 指令的频率
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位

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

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 1  # 提示时长，转移视线
    rest_time = 1  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = "COM8"  # 采集主机端口
    port_addr = None  # 覆盖为 None
    nrep = 2  # block数目
    lsl_source_id = "meta_online_worker"  # source id
    online = False  # 在线实验的标志
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

    ex.run()