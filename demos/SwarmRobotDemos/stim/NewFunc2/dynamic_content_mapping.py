"""NewFunc2 的动态内容映射状态机。"""


class DynamicContentMapper:
    """把 FBSCCA 类别解释为当前界面中的命令或切换动作。"""

    def __init__(self, config):
        self._validate_layout(config)
        interface_config = config["界面配置"]
        self.initial_interface = interface_config.get("初始界面", "主界面")

        control = interface_config["车辆控制区"]
        special = interface_config["特殊功能区"]
        switching = interface_config["车辆切换区"]
        self.control_blocks = self._to_block_map(control, "车辆控制区")
        self.special_blocks = self._to_block_map(special, "特殊功能区")
        self.switch_ids = [int(value) for value in switching["刺激编号"]]
        self.main_labels = self._validate_labels(switching["主界面"], "主界面")
        self.sub_interfaces = {
            str(group): self._validate_labels(labels, "{} 组分界面".format(group))
            for group, labels in switching["分界面"].items()
        }

        active_groups = [label for label in self.main_labels if label]
        if set(active_groups) != set(self.sub_interfaces):
            raise ValueError("主界面车辆组与分界面配置不一致")

        self.current_interface = self.initial_interface
        self.control_scope = "未选择"
        self.selected_group = None
        self.selected_vehicle = None
        self.gear = 1

    @staticmethod
    def _validate_layout(config):
        expected = (20, 4, 5)
        actual = (
            int(config.get("n_elements", 0)),
            int(config.get("rows", 0)),
            int(config.get("columns", 0)),
        )
        if actual != expected:
            raise ValueError("NewFunc2 的 n_elements、rows、columns 必须固定为 20、4、5")

    @staticmethod
    def _to_block_map(region, region_name):
        ids = region.get("刺激编号", [])
        labels = region.get("文字", [])
        if len(ids) != len(labels):
            raise ValueError("{} 的刺激编号与文字数量不一致".format(region_name))
        return {int(stimulus_id): str(label) for stimulus_id, label in zip(ids, labels)}

    def _validate_labels(self, labels, interface_name):
        if len(labels) != len(self.switch_ids):
            raise ValueError("{} 必须配置 8 个车辆切换区文字".format(interface_name))
        return [str(label) for label in labels]

    def _switch_labels(self):
        if self.current_interface == self.initial_interface:
            return self.main_labels
        return self.sub_interfaces[self.current_interface]

    def labels(self):
        """返回 20 个物理刺激块的当前文字，空字符串仍保留对应刺激块。"""
        labels = [""] * 20
        for stimulus_id, label in self.control_blocks.items():
            labels[stimulus_id - 1] = label
        for stimulus_id, label in self.special_blocks.items():
            labels[stimulus_id - 1] = label
        for stimulus_id, label in zip(self.switch_ids, self._switch_labels()):
            labels[stimulus_id - 1] = label
        return labels

    def handle_prediction(self, predict_index):
        """处理算法回传的零基类别并更新界面、控制对象或挡位。"""
        stimulus_id = int(predict_index) + 1
        if not 1 <= stimulus_id <= 20:
            raise ValueError("算法回传类别 {} 超出 0-19".format(predict_index))

        selected_label = self.labels()[stimulus_id - 1]
        action = "空刺激块"
        command = None

        if stimulus_id in self.control_blocks:
            command = selected_label
            if selected_label == "S":
                self.gear = 2 if self.gear == 1 else 1
                action = "切换挡位"
            else:
                action = "车辆控制"
        elif stimulus_id in self.special_blocks:
            command = selected_label
            if selected_label == "ALL":
                self.control_scope = "全体"
                self.selected_group = None
                self.selected_vehicle = None
                self.current_interface = self.initial_interface
                action = "选择全体"
            else:
                action = "特殊功能"
        elif stimulus_id in self.switch_ids and selected_label:
            if self.current_interface == self.initial_interface:
                self.selected_group = selected_label
                self.selected_vehicle = None
                self.control_scope = "组"
                self.current_interface = selected_label
                action = "进入分界面"
            elif selected_label == "BACK":
                self.current_interface = self.initial_interface
                action = "返回主界面"
            else:
                self.selected_vehicle = selected_label
                self.selected_group = selected_label[0]
                self.control_scope = "单车"
                self.current_interface = self.initial_interface
                action = "选择车辆"

        return {
            "刺激编号": stimulus_id,
            "文字": selected_label,
            "动作": action,
            "命令": command,
            "当前界面": self.current_interface,
            "控制范围": self.control_scope,
            "当前组": self.selected_group,
            "当前车辆": self.selected_vehicle,
            "当前挡位": self.gear,
        }
