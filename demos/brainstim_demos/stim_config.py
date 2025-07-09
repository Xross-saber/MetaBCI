import tkinter as tk
from tkinter import ttk, messagebox
import json


class FunctionSettingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("功能设置")
        self.root.geometry("700x500")
        self.root.resizable(True, True)

        # 存储所有设置值
        self.settings = {
            "n_elements": 18,
            "rows": 3,
            "columns": 6,
            "fps": 60,  # Hz
            "stim_time": 2  # 秒
        }

        # 目标选择相关
        self.targets = [
            "开启发射", "攻击一次", "关闭发射", "开启跟随", "关闭跟随", "停止",
            "左转30度", "左行50cm", "关闭避让", "前进50cm", "旋转", "后退50cm",
            "右转30度", "右行50cm", "打开避让", "头左转30度", "持续攻击", "头右转30度"
        ]
        self.selected_targets = set()  # 存储选中的目标

        # 当前页面（共设置三页）
        self.current_page = 0

        # 创建界面
        self.create_widgets()
        # 显示第一页
        self.show_page(0)

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建三个页面的框架
        self.pages = [
            ttk.Frame(main_frame),  # 第一页：功能数量、行数、列数
            ttk.Frame(main_frame),  # 第二页：刷新频率、刺激时长
            ttk.Frame(main_frame)  # 第三页：目标选择
        ]

        # 为每个页面添加内容
        self.create_page1()
        self.create_page2()
        self.create_page3()

        # 导航按钮框架
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # 上一页按钮（初始隐藏）
        self.prev_btn = ttk.Button(
            nav_frame,
            text="上一页",
            command=self.prev_page
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.prev_btn.pack_forget()  # 第一页时隐藏

        # 下一页按钮
        self.next_btn = ttk.Button(
            nav_frame,
            text="下一页",
            command=self.next_page
        )
        self.next_btn.pack(side=tk.RIGHT, padx=5)

    def create_page1(self):
        # 第一页内容：功能数量、行数、列数
        page = self.pages[0]
        page.pack(fill=tk.BOTH, expand=True)

        # 创建三个输入框和标签
        self.entries1 = {}
        for i, (key, value) in enumerate(self.settings.items()):
            if i >= 3:  # 只显示前三个设置项
                break

            # 创建标签
            label = ttk.Label(
                page,
                text=f"{key}：",
                font=("SimHei", 10)
            )
            label.grid(row=i, column=0, padx=5, pady=15, sticky=tk.W)

            # 创建输入框
            entry = ttk.Entry(
                page,
                width=10,
                font=("SimHei", 10)
            )
            entry.insert(0, str(value))
            entry.grid(row=i, column=1, padx=5, pady=15, sticky=tk.W)

            # 存储输入框引用
            self.entries1[key] = entry

        # 添加说明文字
        note_label = ttk.Label(
            page,
            text="输入正整数（如功能数量=18，行数=3，列数=6）",
            font=("SimHei", 9),
            foreground="#666666"
        )
        note_label.grid(row=3, column=0, columnspan=2, padx=5, pady=0, sticky=tk.W)

        # 应用数字验证
        validate_cmd = self.root.register(self.validate_number)
        for entry in self.entries1.values():
            entry.config(
                validate="key",
                validatecommand=(validate_cmd, "%P")
            )

    def create_page2(self):
        # 第二页内容：刷新频率、刺激时长
        page = self.pages[1]
        page.pack(fill=tk.BOTH, expand=True)

        # 创建两个输入框和标签
        self.entries2 = {}
        for i, (key, value) in enumerate(self.settings.items()):
            if i < 3:  # 只显示后两个设置项
                continue

            # 创建标签
            label = ttk.Label(
                page,
                text=f"{key}（{'单位：s' if key == '刺激时长' else '单位：Hz'}）：",
                font=("SimHei", 10)
            )
            label.grid(row=i - 3, column=0, padx=5, pady=15, sticky=tk.W)

            # 创建输入框
            entry = ttk.Entry(
                page,
                width=10,
                font=("SimHei", 10)
            )
            entry.insert(0, str(value))
            entry.grid(row=i - 3, column=1, padx=5, pady=15, sticky=tk.W)

            # 存储输入框引用
            self.entries2[key] = entry

        # 添加说明文字
        note_label = ttk.Label(
            page,
            text="输入正数（如刷新频率=60，刺激时长=2）",
            font=("SimHei", 9),
            foreground="#666666"
        )
        note_label.grid(row=2, column=0, columnspan=2, padx=5, pady=0, sticky=tk.W)

        # 应用数字验证（允许小数）
        validate_float_cmd = self.root.register(self.validate_float)
        for entry in self.entries2.values():
            entry.config(
                validate="key",
                validatecommand=(validate_float_cmd, "%P")
            )

    def create_page3(self):
        # 第三页内容：目标选择
        page = self.pages[2]
        page.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = ttk.Label(
            page,
            text="目标选择（点击按钮选择/取消目标）",
            font=("SimHei", 12, "bold")
        )
        title_label.pack(pady=10)

        # 创建目标按钮网格（3列×6行）
        buttons_frame = ttk.Frame(page)
        buttons_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.target_buttons = {}
        for i, target in enumerate(self.targets):
            row, col = divmod(i, 3)  # 3列布局

            btn = ttk.Button(
                buttons_frame,
                text=target,
                width=20,
                command=lambda t=target: self.toggle_target(t)
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

            self.target_buttons[target] = btn

        # 显示已选目标数量
        self.selected_count_var = tk.StringVar(value=f"已选择: 0/18")
        count_label = ttk.Label(
            page,
            textvariable=self.selected_count_var,
            font=("SimHei", 10)
        )
        count_label.pack(pady=5)

        # 保存按钮
        save_btn = ttk.Button(
            page,
            text="保存全部设置",
            command=self.save_all_settings
        )
        save_btn.pack(pady=10)

    def prev_page(self):
        if self.current_page > 0:
            self.show_page(self.current_page - 1)

    def next_page(self):
        if self.current_page < len(self.pages) - 1:
            self.show_page(self.current_page + 1)

    def show_page(self, page_num):
        # 隐藏所有页面
        for page in self.pages:
            page.pack_forget()

        # 显示指定页面
        self.pages[page_num].pack(fill=tk.BOTH, expand=True)
        self.current_page = page_num

        # 更新导航按钮状态
        if page_num == 0:
            self.prev_btn.pack_forget()  # 第一页隐藏上一页按钮
            self.next_btn.pack(side=tk.RIGHT, padx=5)  # 显示下一页按钮
        elif page_num == len(self.pages) - 1:
            self.prev_btn.pack(side=tk.LEFT, padx=5)  # 显示上一页按钮
            self.next_btn.pack_forget()  # 最后一页隐藏下一页按钮
        else:
            self.prev_btn.pack(side=tk.LEFT, padx=5)  # 显示上一页按钮
            self.next_btn.pack(side=tk.RIGHT, padx=5)  # 显示下一页按钮

    def validate_number(self, value):
        # 验证输入是否为正整数（用于第一页）
        if value == "":
            return True
        return value.isdigit() and int(value) > 0

    def validate_float(self, value):
        # 验证输入是否为正数（允许小数，用于第二页）
        if value == "":
            return True
        try:
            num = float(value)
            return num > 0
        except ValueError:
            return False

    def toggle_target(self, target):
        # 切换目标选择状态
        if target in self.selected_targets:
            self.selected_targets.remove(target)
            self.target_buttons[target].configure(style="TButton")  # 恢复默认样式
        else:
            self.selected_targets.add(target)
            self.target_buttons[target].configure(style="Accent.TButton")  # 设置选中样式

        # 更新已选数量显示
        self.selected_count_var.set(f"已选择: {len(self.selected_targets)}/18")

    def save_all_settings(self):
        # 保存第一页设置
        for key, entry in self.entries1.items():
            value = entry.get().strip()
            if not value:
                messagebox.showwarning("提示", f"请输入{key}")
                return
            self.settings[key] = int(value)

        # 保存第二页设置
        for key, entry in self.entries2.items():
            value = entry.get().strip()
            if not value:
                messagebox.showwarning("提示", f"请输入{key}")
                return
            self.settings[key] = float(value)

        # 检查是否选择了至少一个目标
        if not self.selected_targets:
            messagebox.showwarning("提示", "请至少选择一个目标")
            return

        # 保存第三页设置（目标选择）
        self.settings["选中目标"] = list(self.selected_targets)

        # 显示保存成功信息
        settings_text = "\n".join([
            f"{key}={value}"
            for key, value in self.settings.items()
            if key != "选中目标"
        ])
        targets_text = "\n".join([f"- {target}" for target in self.selected_targets])

        messagebox.showinfo(
            "成功",
            f"已保存所有设置：\n\n{settings_text}\n\n选中目标：\n{targets_text}"
        )

        # 保存到文件
        self.save_to_file()

    def save_to_file(self):
        # 保存所有设置到JSON文件
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=4)
            print("设置已保存到 config.json")
        except Exception as e:
            messagebox.showerror("错误", f"保存文件失败：{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()

    # 设置选中按钮的样式
    style = ttk.Style()
    style.configure("Accent.TButton", foreground="white", background="#4a86e8")

    app = FunctionSettingUI(root)
    root.mainloop()