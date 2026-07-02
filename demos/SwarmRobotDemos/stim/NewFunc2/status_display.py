"""NewFunc2 在线状态栏与目标反馈框。"""

import json
from pathlib import Path

import numpy as np
from psychopy import visual


def load_status_config():
    config_path = Path(__file__).with_name("status_display.json")
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def apply_initial_state(mapper, config):
    """把配置中的初始控制对象和挡位写入 DCMM。"""
    state = config["初始状态"]
    mapper.control_scope = str(state.get("控制范围", "单车"))
    mapper.selected_group = str(state.get("组号", "A"))
    mapper.selected_vehicle = str(state.get("小车号", "A1"))
    mapper.gear = int(state.get("挡位", 1))


class OnlineStatusDisplay:
    """绘制控制对象、挡位以及算法预测对应的蓝色方框。"""

    def __init__(self, win, ssvep, mapper, config):
        self.mapper = mapper
        self.config = config
        header_config = config["状态栏"]
        feedback_config = config["反馈框"]

        positions = np.asarray(ssvep.stim_pos)
        left_edge = float(np.min(positions[:, 0]) - ssvep.stim_length / 2)
        right_edge = float(np.max(positions[:, 0]) + ssvep.stim_length / 2)
        grid_top = float(np.max(positions[:, 1]) + ssvep.stim_width / 2)
        window_top = float(ssvep.win_size[1] / 2)
        gap = float(header_config.get("与刺激区间距", 5))
        requested_height = float(header_config.get("高度", 60))
        available_height = max(30.0, window_top - grid_top - gap)
        header_height = min(requested_height, available_height)
        header_width = right_edge - left_edge
        header_y = grid_top + gap + header_height / 2
        color = header_config.get("颜色", [1, 1, 1])

        self.header_border = visual.Rect(
            win=win,
            units="pix",
            pos=((left_edge + right_edge) / 2, header_y),
            width=header_width,
            height=header_height,
            lineColor=color,
            fillColor=None,
            lineWidth=float(header_config.get("边框宽度", 4)),
        )
        text_height = min(
            float(header_config.get("文字高度", 36)), header_height * 0.6
        )
        self.left_text = visual.TextStim(
            win=win,
            units="pix",
            pos=(left_edge + header_width * 0.23, header_y),
            height=text_height,
            color=color,
            bold=False,
        )
        self.right_text = visual.TextStim(
            win=win,
            units="pix",
            pos=(left_edge + header_width * 0.75, header_y),
            height=text_height,
            color=color,
            bold=False,
        )

        padding = float(feedback_config.get("外扩像素", 8))
        feedback_side = max(ssvep.stim_length, ssvep.stim_width) + 2 * padding
        self.feedback_box = visual.Rect(
            win=win,
            units="pix",
            width=feedback_side,
            height=feedback_side,
            lineColor=feedback_config.get("颜色", [-1, -1, 1]),
            fillColor=None,
            lineWidth=float(feedback_config.get("边框宽度", 10)),
        )
        self.stim_positions = positions
        self.feedback_index = None

    def _field_value(self, field):
        if field == "挡位":
            return str(self.mapper.gear)
        if field == "组号":
            return self.mapper.selected_group or "-"
        if field == "小车":
            if self.mapper.control_scope == "全体":
                return "ALL"
            if self.mapper.control_scope == "组":
                return "grp.{}".format(self.mapper.selected_group or "-")
            return self.mapper.selected_vehicle or "-"
        raise ValueError("未知状态栏字段: {}".format(field))

    def _format_side(self, side_name):
        side = self.config["状态栏"][side_name]
        return "{}{}{}".format(
            side.get("标题", ""),
            side.get("分隔符", ":"),
            self._field_value(side["字段"]),
        )

    def draw_header(self):
        self.left_text.text = self._format_side("左侧")
        self.right_text.text = self._format_side("右侧")
        self.header_border.draw()
        self.left_text.draw()
        self.right_text.draw()

    def set_feedback(self, predict_index):
        predict_index = int(predict_index)
        if not 0 <= predict_index < len(self.stim_positions):
            raise ValueError("反馈目标索引超出范围: {}".format(predict_index))
        self.feedback_index = predict_index
        self.feedback_box.pos = self.stim_positions[predict_index]

    def clear_feedback(self):
        self.feedback_index = None

    def draw_feedback(self):
        if self.feedback_index is not None:
            self.feedback_box.draw()
