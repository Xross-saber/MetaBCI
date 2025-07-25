import numpy as np
from math import pi
from target_config import stim_config
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap  # 导入颜色映射函数
import os
import math

class High_Frequency_SSVEP():
    def __init__(self):
        # 定义背景大小
        self.width, self.height = stim_config.BGW, stim_config.BGH
        self.win_size = np.array([self.width, self.height])
        # 创建黑色背景
        self.background = np.zeros((self.height, self.width), dtype=np.uint8)
        # 生成码元位数
        self.bit_length = stim_config.BIT_LENGTH

        # 设置matplotlib参数
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

    def target_position(self, n_elements=4, rows=2, columns=2, stim_length=150, stim_width=150):
        # 生成刺激目标位置矩阵
        # 调整位置分布范围，使目标集中
        win_size = self.win_size
        # win_size = np.array([int(self.win_size[0]/3), int(self.win_size[1]/2)])
        stim_length = stim_length
        stim_width = stim_width

        # 生成对应目标数的位置矩阵
        n_elements = n_elements
        if rows * columns >= n_elements:
            # 根据给定的列行，将自动生成刺激坐标
            stim_pos = np.zeros((n_elements, 2))
            # 将整个屏幕划分为行*列的刺激块，并确定首个刺激块的中心坐标
            first_pos = (
                    np.array([win_size[0] / columns, win_size[1] / rows]) / 2
            )
            # 判断首目标位置是否合适，因为刺激目标位置对称所以只需判断一个
            if (first_pos[0] < stim_length / 2) or (first_pos[1] < stim_width / 2):
                raise Exception("目标数过多或目标大小不合适")
            # 依次生成各刺激目标坐标
            # 横向排列
            for i in range(rows):
                for j in range(columns):
                    stim_pos[i * columns + j] = first_pos + [j, i] * first_pos * 2
            # 纵向排列
            # for i in range(columns):
            #     for j in range(rows):
            #         stim_pos[i * rows + j] = first_pos + [i, j] * first_pos * 2

            # 将刺激目标移至屏幕中央
            stim_pos -= win_size / 2
            return stim_pos
        else:
            raise Exception("设定目标数与目标分布情况不匹配")


    def sinusoidal_sample(self, freqs, phases, srate, frames, stim_color):
        # 生成刺激目标亮度序列
        # 判断单码元还是多位码元
        if self.bit_length == 1:
            # 单位码元
            # 每一帧抽样一次，参数为起始位置，帧时长，绘制帧数
            time = np.linspace(0, (frames - 1) / srate, frames)
            # 亮度矩阵初始化，参数为绘制帧数、目标数、色彩空间维度
            color = np.zeros((frames, len(freqs), 3))
            for ne, (freq, phase) in enumerate(zip(freqs, phases)):
                print(freq)
                print(phase)
                # 设定刺激目标亮度变化函数为正弦函数
                # sinw = (np.sin(2 * pi * freq * time + pi * phase) + 1) / 2
                cosw = (np.cos(2 * pi * freq * time + pi * phase) + 1) / 2
                color[:, ne, :] = np.vstack(
                    (cosw * stim_color[0], cosw * stim_color[1], cosw * stim_color[2])
                ).T
            return color

        else:
            # 多位码元
            color_set = [0]
            for num in range(self.bit_length):
                time = np.linspace(0, (frames[num] - 1) / srate, frames[num])
                color = np.zeros((frames[num], len(freqs[num]), 3))
                for ne, (freq, phase) in enumerate(zip(freqs[num], phases[num])):
                    print(freq)
                    print(phase)
                    # 设定刺激目标亮度变化函数为正弦函数
                    sinw = (np.sin(2 * pi * freq * time + pi * phase) + 1) / 2
                    color[:, ne, :] = np.vstack(
                        (sinw * stim_color[0], sinw * stim_color[1], sinw * stim_color[2])
                    ).T
                color_set.extend(color)
            color_set = np.array(color_set[1:])
            return color_set



    def plot_stim_target(self, data):
        print(data)
        # x = range(int(len(data[0])/16))
        for i in range(len(data)):
            data[i] = data[i]
            # data[i] = data[i]
        x = range(int(len(data[0])))
        '''绘制波形'''
        # 设置颜色映射和生成颜色列表
        cmap = get_cmap('gist_rainbow')  # 选择一个颜色映射，例如'viridis'
        num_colors = len(data)  # 波形的数量，也是颜色的数量
        colors = [cmap(i / num_colors) for i in range(num_colors)]  # 生成颜色列表

        fig, axs = plt.subplots(nrows=len(data), figsize=(10, 15))  # figsize调整整体图形大小

        # 绘制每个子图
        if len(data) == 1:
            axs.step(x, data[0][:len(x)], color=colors[0])
            axs.set_xticks([])
        else:
            for i in range(len(data)):
                axs[i].step(x, data[i][:len(x)], color=colors[i])
                axs[i].set_xticks([])

        # 调整布局以避免重叠
        plt.tight_layout()

        # 设置图表的标题和坐标轴标签
        # plt.title('编码波形')
        # 显示图表
        # plt.grid(True)  # 显示网格线
        # plt.xlim(1, 7)
        # plt.ylim(0, 1)
        plt.show()


