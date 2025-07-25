import numpy as np

class stim_config():
    # 背景参数配置
    # BGW, BGH = 1920, 1080
    BGW, BGH = 1280, 720
    # 屏幕刷新率
    SRATE = 60
    # 颜色空间
    COLOR = [1, 1, 1]

    # 码元位数设置
    BIT_LENGTH = 1

    # 刺激目标配置
    # 刺激目标个数
    N_ELEMENTS = 1200
    # N_ELEMENTS = 27
    # 刺激目标分布
    ROW, COLUMNS = 30, 40
    # ROW, COLUMNS = 1, 1
    # 刺激目标大小
    WIDTH, HEIGHT = 10, 10

    FERQS = np.arange(8, (8 + N_ELEMENTS * 0.01), 0.01)
    PHASES = np.array([i * 0.35 % 2 for i in range(N_ELEMENTS)])

    # 刺激帧数
    FRAMES = 240  #单码元
    # FRAMES = [180, 180, 180] #多位码元,数组内元素为每一位码元持续时长

