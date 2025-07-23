import numpy as np
import glob
import pickle
from pylsl import StreamInfo, StreamOutlet
import time

# 配置参数
pick_chs = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
ch_num = len(pick_chs)

# 1. 读取所有pkl文件
pkl_files = sorted(glob.glob("*.pkl"), key=lambda x: int(x.split('.')[0]))
print(f"检测到{len(pkl_files)}个pkl文件: {pkl_files}")

# 2. 创建LSL流
stream_name = "meta_feedback"
stream_source_id = "meta_online_worker"
info = StreamInfo(stream_name, 'Markers', 1, 0, 'int32', stream_source_id)
outlet = StreamOutlet(info)
print(f"LSL流已创建: {stream_name}")

# 3. 依次处理每个pkl文件
for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)  # data: numpy.ndarray, shape=(n_chan, n_samples)
    print(f"读取{pkl_file}，shape={data.shape}")
    # 数据预处理（如需）
    # 这里只做简单归一化和通道选择
    if data.shape[0] > ch_num:
        data = data[:ch_num, :]
    X = data[np.newaxis, :, :]  # shape=(1, n_chan, n_samples)
    # 预测标签：某个通道均值最大就是标签
    mean_per_channel = np.mean(data, axis=1)
    pred_label = int(np.argmax(mean_per_channel)) + 1  # 1-based label
    print(f"预测标签: {pred_label}")
    # 4. 通过LSL发送到Unity
    outlet.push_sample([pred_label])
    print(f"已发送到LSL: {pred_label}")
    time.sleep(1)  # 每条间隔1秒，便于Unity端接收
print("全部pkl文件已处理并发送完毕。") 