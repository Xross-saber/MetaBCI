"""生成 NewFunc2 的 20 目标动态内容映射配置。"""

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


GROUP_NAMES = tuple("ABCDEFGH")
CONTROL_IDS = [1, 2, 3, 6, 7, 8, 11, 12, 13]
CONTROL_LABELS = ["L1", "↑", "R1", "←", "S", "→", "L2", "↓", "R2"]
SPECIAL_IDS = [16, 17, 18]
SPECIAL_LABELS = ["END", "INIT", "ALL"]
SWITCH_IDS = [4, 5, 9, 10, 14, 15, 19, 20]


def build_config(group_count, cars_per_group, fps=60.0, stim_time=2.0):
    """按组数和各组车辆数生成主界面、分界面及空白刺激文字。"""
    group_count = int(group_count)
    if not 1 <= group_count <= 8:
        raise ValueError("车辆组数必须在 1-8 之间")
    if float(fps) <= 0 or float(stim_time) <= 0:
        raise ValueError("刷新率和刺激时长必须大于 0")

    active_groups = GROUP_NAMES[:group_count]
    normalized_counts = {}
    sub_interfaces = {}
    for group in active_groups:
        count = int(cars_per_group[group])
        if not 1 <= count <= 7:
            raise ValueError("{} 组车辆数必须在 1-7 之间".format(group))
        normalized_counts[group] = count
        car_labels = ["{}{}".format(group, index) for index in range(1, count + 1)]
        sub_interfaces[group] = car_labels + [""] * (7 - count) + ["BACK"]

    main_labels = list(active_groups) + [""] * (8 - group_count)
    return {
        "n_elements": 20,
        "rows": 4,
        "columns": 5,
        "fps": float(fps),
        "stim_time": float(stim_time),
        "trigger_mode": "start_stop",
        "车辆组数": group_count,
        "每组车辆数": normalized_counts,
        "界面配置": {
            "初始界面": "主界面",
            "车辆控制区": {
                "刺激编号": CONTROL_IDS,
                "文字": CONTROL_LABELS,
            },
            "特殊功能区": {
                "刺激编号": SPECIAL_IDS,
                "文字": SPECIAL_LABELS,
            },
            "车辆切换区": {
                "刺激编号": SWITCH_IDS,
                "主界面": main_labels,
                "分界面": sub_interfaces,
            },
        },
    }


class FunctionSettingUI:
    """车辆组数与组内车辆数配置界面。"""

    def __init__(self, root):
        self.root = root
        self.root.title("UniSwarm-SSVEP 动态界面配置")
        self.root.geometry("620x620")
        self.root.minsize(560, 580)

        existing = self._load_existing_config()
        self.group_count_var = tk.IntVar(value=existing.get("车辆组数", 4))
        self.fps_var = tk.DoubleVar(value=existing.get("fps", 60.0))
        self.stim_time_var = tk.DoubleVar(value=existing.get("stim_time", 2.0))
        existing_counts = existing.get("每组车辆数", {})
        self.car_count_vars = {
            group: tk.IntVar(value=existing_counts.get(group, 3))
            for group in GROUP_NAMES
        }
        self.car_spinboxes = {}

        self._create_widgets()
        self.group_count_var.trace_add("write", self._on_group_count_changed)
        self._update_group_states()

    @staticmethod
    def _load_existing_config():
        config_path = Path(__file__).with_name("config.json")
        if not config_path.exists():
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                return json.load(config_file)
        except (OSError, ValueError):
            return {}

    def _create_widgets(self):
        root_frame = ttk.Frame(self.root, padding=20)
        root_frame.pack(fill=tk.BOTH, expand=True)

        layout_frame = ttk.LabelFrame(root_frame, text="刺激布局", padding=12)
        layout_frame.pack(fill=tk.X)
        ttk.Label(
            layout_frame,
            text="固定 20 个刺激块：3×3 车辆控制区、3 个特殊功能块、8 个车辆切换块",
            wraplength=530,
        ).pack(anchor=tk.W)

        basic_frame = ttk.LabelFrame(root_frame, text="实验参数", padding=12)
        basic_frame.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(basic_frame, text="刷新率 (Hz)").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(
            basic_frame, from_=1, to=360, textvariable=self.fps_var, width=10
        ).grid(row=0, column=1, padx=(12, 30), sticky=tk.W)
        ttk.Label(basic_frame, text="刺激时长 (s)").grid(row=0, column=2, sticky=tk.W)
        ttk.Spinbox(
            basic_frame,
            from_=0.1,
            to=30,
            increment=0.1,
            textvariable=self.stim_time_var,
            width=10,
        ).grid(row=0, column=3, padx=(12, 0), sticky=tk.W)

        group_frame = ttk.LabelFrame(root_frame, text="车辆切换区", padding=12)
        group_frame.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        ttk.Label(group_frame, text="车辆组数 (1-8)").grid(
            row=0, column=0, columnspan=2, sticky=tk.W
        )
        ttk.Spinbox(
            group_frame,
            from_=1,
            to=8,
            textvariable=self.group_count_var,
            width=8,
            state="readonly",
        ).grid(row=0, column=2, padx=10, sticky=tk.W)

        ttk.Separator(group_frame).grid(
            row=1, column=0, columnspan=4, pady=12, sticky=tk.EW
        )
        ttk.Label(group_frame, text="组名").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(group_frame, text="组内车辆数 (1-7)").grid(
            row=2, column=1, columnspan=2, sticky=tk.W
        )

        for row, group in enumerate(GROUP_NAMES, start=3):
            ttk.Label(group_frame, text="{} 组".format(group)).grid(
                row=row, column=0, pady=5, sticky=tk.W
            )
            spinbox = ttk.Spinbox(
                group_frame,
                from_=1,
                to=7,
                textvariable=self.car_count_vars[group],
                width=8,
                state="readonly",
            )
            spinbox.grid(row=row, column=1, pady=5, sticky=tk.W)
            self.car_spinboxes[group] = spinbox

        button_frame = ttk.Frame(root_frame)
        button_frame.pack(fill=tk.X, pady=(16, 0))
        ttk.Button(button_frame, text="保存配置", command=self.save_config).pack(
            side=tk.RIGHT
        )

    def _on_group_count_changed(self, *_):
        self._update_group_states()

    def _update_group_states(self):
        try:
            group_count = int(self.group_count_var.get())
        except (tk.TclError, ValueError):
            return
        for index, group in enumerate(GROUP_NAMES):
            state = "readonly" if index < group_count else "disabled"
            self.car_spinboxes[group].configure(state=state)

    def save_config(self):
        try:
            group_count = int(self.group_count_var.get())
            cars_per_group = {
                group: int(self.car_count_vars[group].get())
                for group in GROUP_NAMES[:group_count]
            }
            config = build_config(
                group_count=group_count,
                cars_per_group=cars_per_group,
                fps=float(self.fps_var.get()),
                stim_time=float(self.stim_time_var.get()),
            )
            config_path = Path(__file__).with_name("config.json")
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump(config, config_file, ensure_ascii=False, indent=4)
        except (KeyError, OSError, ValueError, tk.TclError) as error:
            messagebox.showerror("保存失败", str(error))
            return

        messagebox.showinfo("保存成功", "配置已写入 {}".format(config_path))


if __name__ == "__main__":
    app_root = tk.Tk()
    FunctionSettingUI(app_root)
    app_root.mainloop()
