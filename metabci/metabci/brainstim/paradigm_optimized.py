# -*- coding: utf-8 -*-
import math

# load in basic modules
import os
import os.path as op
import string
import numpy as np
from math import pi
from psychopy import data, visual, event
from psychopy.visual.circle import Circle
from pylsl import StreamInlet, resolve_byprop  # type: ignore
from .utils import NeuroScanPort, NeuraclePort, _check_array_like
import threading
from copy import copy
import random
from scipy import signal
from PIL import Image
from trigger_config import TriggerConfig
from trigger.TriggerController import TriggerController

# prefunctions


def sinusoidal_sample(freqs, phases, srate, frames, stim_color):
    """
    Sinusoidal approximate sampling method.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        freqs: list of float
            Frequencies of each stimulus.
        phases: list of float
            Phases of each stimulus.
        srate: int or float
            Refresh rate of screen.
        frames: int
            Flashing frames.
        stim_color: list
            Color of stimu.

    Returns
    ----------
        color: ndarray
            shape(frames, len(fre), 3)

    """

    time = np.linspace(0, (frames - 1) / srate, frames)
    color = np.zeros((frames, len(freqs), 3))
    for ne, (freq, phase) in enumerate(zip(freqs, phases)):
        sinw = np.sin(2 * pi * freq * time + pi * phase) + 1
        color[:, ne, :] = np.vstack(
            (sinw * stim_color[0], sinw * stim_color[1], sinw * stim_color[2])
        ).T
        if stim_color == [-1, -1, -1]:
            pass
        else:
            if stim_color[0] == -1:
                color[:, ne, 0] = -1
            if stim_color[1] == -1:
                color[:, ne, 1] = -1
            if stim_color[2] == -1:
                color[:, ne, 2] = -1

    return color


def wave_new(stim_num, type):
    """determine the color of each offset dot according to "type".

    author: Jieyu Wu

    Created on: 2022-12-14

    update log:
        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        stim_num: int
            Number of stimuli dots of each target.
        type: int
            avep code.

    Returns
    ----------
        point: ndarray
            (stim_num, 3)

    """
    point = [[-1, -1, -1] for i in range(stim_num)]
    if type == 0:
        pass
    else:
        point[type - 1] = [1, 1, 1]
    point = np.array(point)
    return point


def pix2height(win_size, pix_num):
    height_num = pix_num / win_size[1]
    return height_num


def height2pix(win_size, height_num):
    pix_num = height_num * win_size[1]
    return pix_num


def code_sequence_generate(basic_code, sequences):
    """Quickly generate coding sequences for sub-stimuli using basic endcoding units and encoding sequences.

    author: Jieyu Wu

    Created on: 2023-09-18

    update log:
        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        basic_code: list
            Each basic encoding unit in the encoding sequence.
        sequences: list of array
            Encoding sequences for basic_code.

    Returns
    ----------
        code: ndarray
            coding sequences for sub-stimuli.

    """

    code = []
    for seq_i in range(len(sequences)):
        code_list = []
        seq_length = len(sequences[seq_i])
        for code_i in range(seq_length):
            code_list.append(basic_code[sequences[seq_i][code_i]])
        code.append(code_list)
    code = np.array(code)
    return code


# create interface for VEP-BCI-Speller


class KeyboardInterface(object):
    """Create the interface to the stimulus interface and initialize the window parameters.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        win:
            The window object.
        win_size: ndarray, shape(width, high)
            The size of the window in pixels.
        stim_length: int
            The length of the stimulus block in pixels.
        stim_width: int
            The width of the stimulus block in pixels.
        n_elements: int
            Number of stimulus blocks.
        stim_pos: ndarray, shape([x, y],...)
            Customize the position of the stimulus blocks with an array length
            that corresponds to the number of stimulus blocks.
        stim_sizes: ndarray, shape([length, width],...)
            The size of the stimulus block, the length of which corresponds to the number of stimulus blocks.
        symbols: str
            The text content of the stimulus block.
        symbol_height: int
            The height of the text in the stimulus block.
        symbol_color: ndarray, shape(3,)
            The color of the text in the stimulus block.
        text_stimuli: list
            The text stimulus object list.
        index_stimuli: visual.TextStim
            The index stimulus object.
        rect_response: visual.Rect
            The response rectangle object.
        text_response: visual.TextStim
            The response text object.

    Tip
    ----
    .. code-block:: python
        :caption: An example of creating keyboard interface.

        from psychopy import monitors
        import numpy as np
        from brainstim.framework import Experiment
        from brainstim.paradigm import KeyboardInterface

        win = ex.get_window()

        # press q to exit paradigm interface
        n_elements, rows, columns = 40, 5, 8
        tex_color = [1,1,1]                                         # color of text
        basic_KeyboardInterface = KeyboardInterface(win=win)
        basic_KeyboardInterface.config_pos(n_elements=n_elements, rows=rows, columns=columns)
        basic_KeyboardInterface.config_text(tex_color=tex_color)
        basic_KeyboardInterface.config_response(bg_color=[0,0,0])
        bg_color = np.array([0, 0, 0])                              # background color
        display_time = 1
        index_time = 0.5
        response_time = 2
        rest_time = 0.5
        port_addr = None
        nrep = 1                                                    # Number of blocks
        lsl_source_id = None
        online = False
        ex.register_paradigm('basic KeyboardInterface', paradigm, VSObject=basic_KeyboardInterface, bg_color=bg_color, display_time=display_time,
            index_time=index_time, rest_time=rest_time, response_time=response_time, port_addr=port_addr, nrep=nrep,
            pdim='keyboard', lsl_source_id=lsl_source_id, online=online)

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        # 启用垂直同步以提高呈现稳定性
        self.win = win
        self.win.waitBlanking = True  # 新增：启用垂直同步
        self.colorSpace = colorSpace
        self.allowGUI = allowGUI
        self.win_size = np.array([self.win.size[0], self.win.size[1]])
        self.stim_length = 150
        self.stim_width = 150
        self.n_elements = 40
        self.stim_pos = np.array([[0, 0]])
        self.stim_sizes = np.array([[self.stim_length, self.stim_width]])
        self.symbols = ["AB"]
        self.symbol_height = 0
        self.symbol_color = np.array([1, 1, 1])
        self.text_stimuli = []
        self.index_stimuli = visual.TextStim(
            win=self.win,
            text="",
            color=self.symbol_color,
            height=self.symbol_height,
        )
        self.rect_response = visual.Rect(
            win=self.win,
            units="pix",
            width=self.win_size[0],
            height=self.win_size[1],
            fillColor=[-1, -1, -1],
            lineColor=[-1, -1, -1],
        )
        self.text_response = visual.TextStim(
            win=self.win,
            text="Speller:  ",
            color=self.symbol_color,
            height=self.symbol_height,
        )

    def config_pos(
        self,
        n_elements=40,
        rows=5,
        columns=8,
        stim_pos=None,
        stim_length=150,
        stim_width=150,
    ):
        """Configure the position of the stimulus blocks.

        Parameters
        ----------
            n_elements: int
                Number of stimulus blocks.
            rows: int
                Number of rows.
            columns: int
                Number of columns.
            stim_pos: ndarray, shape([x, y],...)
                Customize the position of the stimulus blocks with an array length
                that corresponds to the number of stimulus blocks.
            stim_length: int
                The length of the stimulus block in pixels.
            stim_width: int
                The width of the stimulus block in pixels.

        """
        self.n_elements = n_elements
        self.stim_length = stim_length
        self.stim_width = stim_width
        self.stim_sizes = np.array([[self.stim_length, self.stim_width]])

        if stim_pos is None:
            self.stim_pos = np.zeros((n_elements, 2))
            for i in range(n_elements):
                self.stim_pos[i, 0] = (
                    (i % columns) - (columns - 1) / 2
                ) * self.stim_length
                self.stim_pos[i, 1] = (
                    (rows - 1) / 2 - (i // columns)
                ) * self.stim_width
        else:
            self.stim_pos = stim_pos

    def config_text(
        self, unit="pix", symbols=["AB","AB","AB","AB","AB",
                                   "AB","开启\n跟随","关闭\n跟随","AB","AB",
                                   "AB","左转\n30度","左行\n50cm","停止\n旋转","AB",
                                   "AB","前进\n50cm","旋转","后退\n50cm","AB",
                                   "AB","右转\n30度","右行\n50cm","平视","AB",
                                   "抬头","头左转\n30度","头右转\n30度","低头","AB",
                                   "开启\n发射", "攻击\n一次", "关闭\n发射", "AB", "AB",
                                   "AB","AB","AB","AB","AB",], symbol_height=0, tex_color=[1, 1, 1], text_offset=[0, 0]
    ):
        """Configure the text content of the stimulus blocks.

        Parameters
        ----------
            unit: str
                The unit of the text height.
            symbols: list
                The text content of the stimulus blocks.
            symbol_height: int
                The height of the text in the stimulus block.
            tex_color: list
                The color of the text in the stimulus block.
            text_offset: list
                The offset of the text position.

        """
        self.symbols = symbols
        self.symbol_height = symbol_height
        self.symbol_color = np.array(tex_color)
        self.text_offset = np.array(text_offset)

        self.text_stimuli = []
        for i in range(self.n_elements):
            self.text_stimuli.append(
                visual.TextStim(
                    win=self.win,
                    text=self.symbols[i],
                    color=self.symbol_color,
                    height=self.symbol_height,
                    pos=self.stim_pos[i] + self.text_offset,
                )
            )

    def config_response(
        self,
        symbol_text="Speller:  ",
        symbol_height=0,
        symbol_color=(1, 1, 1),
        bg_color=[-1, -1, -1],
    ):
        """Configure the response interface.

        Parameters
        ----------
            symbol_text: str
                The text content of the response interface.
            symbol_height: int
                The height of the text in the response interface.
            symbol_color: tuple
                The color of the text in the response interface.
            bg_color: list
                The background color of the response interface.

        """
        self.symbol_text = symbol_text
        self.symbol_height = symbol_height
        self.symbol_color = np.array(symbol_color)
        self.bg_color = np.array(bg_color)

        self.rect_response = visual.Rect(
            win=self.win,
            units="pix",
            width=self.win_size[0],
            height=self.win_size[1],
            fillColor=self.bg_color,
            lineColor=self.bg_color,
        )
        self.text_response = visual.TextStim(
            win=self.win,
            text=self.symbol_text,
            color=self.symbol_color,
            height=self.symbol_height,
        )


class VisualStim(KeyboardInterface):
    """Create visual stimuli.

    The subclass VisualStim inherits from the parent class KeyboardInterface, and duplicate properties are no longer listed.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        index_stimuli: visual.TextStim
            The index stimulus object.

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

    def config_index(self, index_height=0, units="pix"):
        """Configure the index stimulus.

        Parameters
        ----------
            index_height: int
                The height of the index stimulus.
            units: str
                The unit of the index stimulus.

        """
        self.index_stimuli = visual.TextStim(
            win=self.win,
            text="",
            color=self.symbol_color,
            height=index_height,
            units=units,
        )


class SemiCircle(Circle):
    """Create a semicircle stimulus.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    """

    def _calcVertices(self):
        # only draw half of a circle
        self.vertices = self._calcCircleVertices()
        self.vertices = self.vertices[: len(self.vertices) // 2]


class SSVEP(VisualStim):
    """Create SSVEP stimuli.

    The subclass SSVEP inherits from the parent class VisualStim, and duplicate properties are no longer listed.

    author: Qiaoyi Wu

    Created on: 2022-06-20

    update log:
        2022-06-26 by Jianhang Wu

        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win:
            The window object.
        colorspace: str
            The color space, default to rgb.
        allowGUI: bool
            Defaults to True, which allows frame-by-frame drawing and key-exit.

    Attributes
    ----------
        refresh_rate: int
            Screen refresh rate.
        stim_time: float
            Stimulation time.
        stim_color: list, (red, green, blue)
            The color of the stimulus block, ranging from -1.0 to 1.0.
        stim_opacities: float
            The opacity of the stimulus block.
        stim_frames: int
            Total stimulation frames.
        freqs: list
            Frequencies of each stimulus.
        phases: list
            Phases of each stimulus.
        stim_colors: ndarray
            Pre-computed stimulus colors for all frames.
        flash_stimuli: list
            Pre-generated stimulus objects for all frames.

    Tip
    ----
    .. code-block:: python
        :caption: An example of creating SSVEP stimuli.

        from psychopy import monitors
        import numpy as np
        from brainstim.framework import Experiment
        from brainstim.paradigm import SSVEP,paradigm

        win = ex.get_window()

        # press q to exit paradigm interface
        n_elements, rows, columns = 40, 5, 8
        tex_color = [1,1,1]                                         # color of text
        fps = 240                                                   # screen refresh rate
        stim_time = 0.54
        stim_color = [1,1,1]                                        # color of stimulus
        freqs = np.arange(8, 16, 0.2)                              # frequencies
        phases = np.array([i * 0.35 % 2 for i in range(n_elements)]) # phases
        basic_SSVEP = SSVEP(win=win)
        basic_SSVEP.config_pos(n_elements=n_elements, rows=rows, columns=columns)
        basic_SSVEP.config_text(tex_color=tex_color)
        basic_SSVEP.config_color(refresh_rate=fps, stim_time=stim_time, stim_color=stim_color, freqs=freqs, phases=phases)
        basic_SSVEP.config_index()
        basic_SSVEP.config_response(bg_color=[0,0,0])
        bg_color = np.array([0, 0, 0])                              # background color
        display_time = 1
        index_time = 0.5
        response_time = 2
        rest_time = 0.5
        port_addr = None
        nrep = 1                                                    # Number of blocks
        lsl_source_id = None
        online = False
        ex.register_paradigm('basic SSVEP', paradigm, VSObject=basic_SSVEP, bg_color=bg_color, display_time=display_time,
            index_time=index_time, rest_time=rest_time, response_time=response_time, port_addr=port_addr, nrep=nrep,
            pdim='ssvep', lsl_source_id=lsl_source_id, online=online)

    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

    def config_color(self, refresh_rate, stim_time, stim_color, stimtype="sinusoid", stim_opacities=1, **kwargs):
        """Configure SSVEP paradigm interface parameters, including screen refresh rate
        and stimulation time interval.

        Parameters
        ----------
            refresh_rate: int
                Refresh rate of screen.
            stim_time: float
                Stimulation time.
            stim_color: list
                The color of the stimulus block.
            stimtype: str
                The type of stimulation.
            stim_opacities: float
                The opacity of the stimulus block.
            **kwargs: dict
                Additional parameters.

        """
        self.refresh_rate = refresh_rate
        self.stim_time = stim_time
        self.stim_color = stim_color
        self.stim_opacities = stim_opacities
        self.stim_frames = int(stim_time * self.refresh_rate)

        if refresh_rate == 0:
            self.refresh_rate = np.floor(self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))

        self.stim_oris = np.zeros((self.n_elements,))
        self.stim_sfs = np.zeros((self.n_elements,))
        self.stim_contrs = np.ones((self.n_elements,))

        if "stim_oris" in kwargs: self.stim_oris = kwargs["stim_oris"]
        if "stim_sfs" in kwargs: self.stim_sfs = kwargs["stim_sfs"]
        if "stim_contrs" in kwargs: self.stim_contrs = kwargs["stim_contrs"]
        if "freqs" in kwargs: self.freqs = kwargs["freqs"]
        if "phases" in kwargs: self.phases = kwargs["phases"]

        # 预计算 stim_colors 以减少实时计算开销
        if stimtype == "sinusoid":
            self.stim_colors = sinusoidal_sample(
                freqs=self.freqs,
                phases=self.phases,
                srate=self.refresh_rate,
                frames=self.stim_frames,
                stim_color=stim_color
            ) - 1
            if self.stim_colors.shape[1] != self.n_elements:
                raise Exception("Please input correct num of stims!")

        # 预生成 flash_stimuli 数组以提高绘制效率
        self.flash_stimuli = []
        for sf in range(self.stim_frames):
            self.flash_stimuli.append(
                visual.ElementArrayStim(
                    win=self.win,
                    units="pix",
                    nElements=self.n_elements,
                    sizes=self.stim_sizes,
                    xys=self.stim_pos,
                    colors=self.stim_colors[sf, ...],
                    opacities=self.stim_opacities,
                    oris=self.stim_oris,
                    sfs=self.stim_sfs,
                    contrs=self.stim_contrs,
                    elementTex=np.ones((64, 64)),
                    elementMask=None,
                    texRes=48,
                )
            )


def paradigm(
    VSObject,
    win,
    bg_color,
    display_time=1.0,
    index_time=1.0,
    rest_time=0.5,
    response_time=2,
    image_time=2,
    port_addr=9045,
    nrep=1,
    pdim="ssvep",
    lsl_source_id=None,
    online=None,
    device_type="NeuroScan",
):
    """
    The classical paradigm is implemented, the task flow is defined, the ' q '
    exit paradigm is clicked, and the start selection interface is returned.

    author: Wei Zhao

    Created on: 2022-07-30

    update log:

        2022-08-10 by Wei Zhao

        2022-08-03 by Shengfu Wen

        2022-12-05 by Jie Mei

        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        VSObject:
            Examples of the three paradigms.
        win:
            window.
        bg_color: ndarray
            Background color.
        fps: int
            Display refresh rate.
        display_time: float
            Keyboard display time before 1st index.
        index_time: float
            Indicator display time.
        rest_time: float, optional
            SSVEP and P300 paradigm: the time interval between the target cue and the start of the stimulus.
            MI paradigm: the time interval between the end of stimulus presentation and the target cue.
        respond_time: float, optional
            Feedback time during online experiment.
        image_time: float, optional,
            MI paradigm: Image time.
        port_addr:
             Computer port , hexadecimal or decimal.
        nrep: int
            Num of blocks.
        pdim: str
            One of the three paradigms can be 'ssvep ', ' p300 ', ' mi ' and ' con-ssvep '.
        mi_flag: bool
            Flag of MI paradigm.
        lsl_source_id: str
            The id of communication with the online processing program needs to be consistent between the two parties.
        online: bool
            Flag of online experiment.
        device_type: str
            See support device list in brainstim README file

    """
    # 启用帧间隔记录以便调试
    win.recordFrameIntervals = True
    
    win.color = bg_color
    fps = VSObject.refresh_rate

    trig_ctrl = TriggerController(TriggerConfig.EEG_TYPE,
                                       TriggerConfig.TRIGGER_HANDLE,
                                       TriggerConfig.TRIGGER_PORT)
    trig_ctrl.open()

    # trial开始trigger
    trial_start_trig = None
    # 刺激开始输出trigger
    trial_start_trig: int = TriggerConfig.TRIAL_START_TRIGGER
    # 刺激结束输出trigger
    trial_end_trig: int = TriggerConfig.TRIAL_END_TRIGGER
    # block启动输出trigger
    block_start_trig: int = TriggerConfig.BLOCK_START_TRIGGER
    # block结束输出trigger
    block_end_trig: int = TriggerConfig.BLOCK_END_TRIGGER
    # 数据开始记录trigger
    record_start_trig: int = TriggerConfig.RECORD_START_TRIGGER
    # 数据停止记录trigger
    record_end_trig: int = TriggerConfig.RECORD_END_TRIGGER

    if not _check_array_like(bg_color, 3):
        raise ValueError("bg_color should be 3 elements array-like object.")
    win.color = bg_color
    fps = VSObject.refresh_rate

    if device_type == "NeuroScan":
        port = NeuroScanPort(port_addr, use_serial=True) if port_addr else None
    elif device_type == "Neuracle":
        port = NeuraclePort(port_addr) if port_addr else None
    else:
        raise KeyError(
            "Unknown device type: {}, please check your input".format(device_type))
    port_frame = int(0.05 * fps)

    inlet = False
    if online:
        if (
            pdim == "ssvep"
            or pdim == "p300"
            or pdim == "con-ssvep"
            or pdim == "avep"
            or pdim == "ssavep"
        ):
            VSObject.text_response.text = copy(VSObject.reset_res_text)
            VSObject.text_response.pos = copy(VSObject.reset_res_pos)
            VSObject.res_text_pos = copy(VSObject.reset_res_pos)
            VSObject.symbol_text = copy(VSObject.reset_res_text)
            res_text_pos = VSObject.reset_res_pos
        if lsl_source_id:
            inlet = True
            streams = resolve_byprop(
                "source_id", lsl_source_id, timeout=5
            )  # Resolve all streams by source_id
            if not streams:
                return
            inlet = StreamInlet(streams[0])  # receive stream data

    if pdim == "ssvep":
        # config experiment settings
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(
            conditions,
            nrep,
            name="experiment",
            method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + \
                np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            trig_ctrl.send(trial_start_trig)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)
                VSObject.flash_stimuli[sf].draw()
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()

    # 打印平均帧时间以便调试
    print(f"Average frame time: {win.getMsPerFrame():.2f} ms")
    win.close() 