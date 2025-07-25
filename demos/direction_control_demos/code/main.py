import os
import numpy as np
from math import pi
from PIL import Image, ImageEnhance

import functions
from functions import High_Frequency_SSVEP
from target_config import stim_config

def creat_stim_target_order():
    '''生成目录界面待刺激序列'''
    for sf in range(frames):
    # for sf in range(sum(frames)):
        for i in range(n_elements):
            # 读取刺激目标图片
            # foreground_image = Image.open(symbols[i])
            foreground_image = Image.open(output_images)
            # 调节目标图片大小
            resized_image = foreground_image.resize((width, height), 3)

            # 创建亮度增强对象
            enhancer = ImageEnhance.Brightness(resized_image)

            # 调节目标亮度
            brightened_image = enhancer.enhance(brightness_values[sf, i, 0])

            # 将刺激目标图片粘贴到背景图片的指定位置
            foreground_width, foreground_height = brightened_image.size
            center_x = stim_pos[i][0]
            center_y = stim_pos[i][1]
            left = center_x - foreground_width // 2 + background_width//2
            top = center_y - foreground_height // 2 + background_height//2
            background_image.paste(brightened_image, (int(left), int(top)))

        # 图片存储至指定位置
        output_path = os.path.join(current_directory, 'stim_generate', 'image_folder', f'{sf}.jpg')
        background_image.save(output_path)

def creat_stim_init_pic():
    '''生成目录界面初始化帧'''
    for i in range(n_elements):
        foreground_image = Image.open(output_images)
        # foreground_image = Image.open(symbols[i])
        resized_image = foreground_image.resize((width, height), 3)

        foreground_width, foreground_height = resized_image.size
        center_x = stim_pos[i][0]
        center_y = stim_pos[i][1]
        left = center_x - foreground_width // 2 + background_width // 2
        top = center_y - foreground_height // 2 + background_height // 2
        background_image.paste(resized_image, (int(left), int(top)))

    output_path = os.path.join(current_directory, 'stim_generate', 'init.jpg')
    background_image.save(output_path)

def creat_stimulus_target_waveform(data):
    '''生成刺激目标波形'''
    data = data
    tar_list = []
    for j in range(len(data[0])):
        tar_bright = []
        for i in range(len(data)):
            tar_bright.append(data[i][j][0])
        tar_list.append(tar_bright)
    # for num in range(0, len(data[0]), 2):
    #     paradigm.plot_stim_target(tar_list[num], tar_list[num + 1])
    paradigm.plot_stim_target(tar_list)




if __name__ == "__main__":

    # 背景参数设置
    background_width = stim_config.BGW
    background_height = stim_config.BGH
    background_image = Image.new('RGB', (background_width, background_height), (0, 0, 0))

    # 图片路径读取
    current_directory = os.getcwd()
    output_images = os.path.join(current_directory, 'target_image', 'white.jpg')

    # 刺激目标配置初始化
    n_elements, rows, columns = stim_config.N_ELEMENTS, stim_config.ROW, stim_config.COLUMNS
    freqs = stim_config.FERQS
    # print(freqs)
    phases = stim_config.PHASES
    # print(phases)
    srate = stim_config.SRATE
    frames = stim_config.FRAMES
    stim_color = stim_config.COLOR
    width, height = stim_config.WIDTH, stim_config.HEIGHT

    # 刺激界面初始化
    paradigm = High_Frequency_SSVEP()
    stim_pos = paradigm.target_position(n_elements, rows, columns, width, height)
    print(stim_pos)
    brightness_values = paradigm.sinusoidal_sample(freqs, phases, srate, frames, stim_color)

    # 生成单目标训练刺激
    # 生成刺激目标波形
    creat_stimulus_target_waveform(brightness_values)

    # 生成目录界面待刺激序列
    creat_stim_target_order()

    # 生成目录界面初始化帧
    creat_stim_init_pic()

