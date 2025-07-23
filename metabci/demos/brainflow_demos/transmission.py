# -*- coding: utf-8 -*-
"""
Retention Transmission for Real-time SSVEP Analysis
Auxiliary script to support online analysis with FBDSP.
Authors: [Your Name]
Last update date: 2025-07-07
License: MIT License
"""

import numpy as np
from mne.filter import resample
import pickle
import time
import socket
from typing import Optional, List
from metabci.brainda.algorithms.decomposition import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.decomposition import FBDSP
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str = 'trained_fbdsp.pkl') -> FBDSP:
    """
    Load pre-trained FBDSP model from file.

    Parameters
    ----------
    model_path : str
        Path to the pre-trained model file (default: 'trained_fbdsp.pkl')

    Returns
    -------
    model : FBDSP
        Loaded FBDSP model
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def preprocess_data(X: np.ndarray, srate: int = 1000) -> np.ndarray:
    """
    Preprocess real-time EEG data: resample and normalize.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_channels, n_samples)
    srate : int
        Original sampling rate (default: 1000 Hz)

    Returns
    -------
    X_processed : np.ndarray
        Processed EEG data, shape (n_channels, n_samples_resampled)
    """
    # 降采样到 256 Hz
    X = resample(X, up=256, down=srate)
    # 零均值单位方差归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=-1, keepdims=True)
    return X


def retention_transmission(
        data_stream: np.ndarray,
        model: FBDSP,
        srate: int = 1000,
        window_length: float = 0.5,
        optimize_weights: bool = False,
        n_splits: int = 5,
        udp_port: Optional[int] = None,
        output_file: str = 'retention_results.txt'
) -> dict:
    """
    Process and transmit real-time SSVEP data using FBDSP.

    Parameters
    ----------
    data_stream : np.ndarray
        Real-time EEG data stream, shape (n_channels, n_samples)
    model : FBDSP
        Pre-trained FBDSP model
    srate : int
        Sampling rate of the data stream (default: 1000 Hz)
    window_length : float
        Length of the data window in seconds (default: 0.5)
    optimize_weights : bool
        Whether to optimize filter weights online (default: False)
    n_splits : int
        Number of folds for weight optimization (default: 5)
    udp_port : Optional[int]
        UDP port for transmitting results (default: None)
    output_file : str
        File to save results (default: 'retention_results.txt')

    Returns
    -------
    result : dict
        Dictionary containing prediction results and optional optimized weights
    """
    result = {}

    # 检查数据形状
    if data_stream.ndim != 2:
        logger.error("Data stream must be 2D array (n_channels, n_samples)")
        raise ValueError("Invalid data stream shape")

    # 预处理数据
    X = preprocess_data(data_stream, srate)

    # 确保数据长度符合窗口要求
    expected_samples = int(window_length * 256)  # 降采样后为 256 Hz
    if X.shape[-1] < expected_samples:
        logger.error(f"Data length {X.shape[-1]} is less than required {expected_samples}")
        raise ValueError("Insufficient data length")

    # 截取时间窗
    X = X[:, :expected_samples]
    X = X[np.newaxis, :, :]  # 转换为 (1, n_channels, n_samples)

    # 进行预测
    pred_label = model.predict(X)[0]
    result['predicted_label'] = pred_label

    # 可选：在线权重优化
    if optimize_weights:
        logger.info("Performing online weight optimization")
        # 假设使用预测标签（实际应用需提供真实标签）
        y = np.array([pred_label])  # 临时标签，需替换
        new_weights = model.optimize_weights(X, y, n_splits=n_splits)
        result['optimized_weights'] = new_weights
        logger.info(f"Optimized weights: {new_weights}")

    # 保存结果到文件
    with open(output_file, 'a') as f:
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Predicted label: {pred_label}\n")
        if 'optimized_weights' in result:
            f.write(f"Optimized weights: {result['optimized_weights']}\n")
        f.write("\n")

    # 可选：通过 UDP 传输结果
    if udp_port is not None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                message = f"Predicted label: {pred_label}"
                if 'optimized_weights' in result:
                    message += f", Optimized weights: {result['optimized_weights']}"
                sock.sendto(message.encode(), ('localhost', udp_port))
                logger.info(f"Sent result to UDP port {udp_port}: {message}")
        except Exception as e:
            logger.error(f"Failed to send UDP message: {e}")

    return result


def main():
    """
    Main function for retention transmission.
    Simulates real-time EEG data stream for demonstration.
    """
    # 参数设置
    srate = 1000
    window_length = 0.5
    n_channels = 9
    n_samples = int(srate * window_length)
    udp_port = 12345  # 可选 UDP 端口

    # 加载模型
    model = load_model('trained_fbdsp.pkl')

    # 模拟数据流（实际应用中替换为 LSL 或硬件输入）
    logger.info("Simulating EEG data stream")
    data_stream = np.random.randn(n_channels, n_samples)

    # 运行留置传输
    result = retention_transmission(
        data_stream,
        model,
        srate=srate,
        window_length=window_length,
        optimize_weights=True,  # 开启在线权重优化
        n_splits=5,
        udp_port=udp_port,
        output_file='retention_results.txt'
    )

    # 输出结果
    logger.info(f"Predicted frequency label: {result['predicted_label']}")
    if 'optimized_weights' in result:
        logger.info(f"Optimized filter weights: {result['optimized_weights']}")


if __name__ == "__main__":
    main()