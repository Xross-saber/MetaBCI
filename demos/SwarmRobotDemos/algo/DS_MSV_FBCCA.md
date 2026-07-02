# DS-MSV-FBCCA

算法全称为 **Dynamic-Stopping Multi-Singular-Value Filter-Bank CCA**，中文名称为“动态停止多奇异值滤波器组CCA”。

## 来源与复现内容

该实现根据 `127.82.tar` 中的 `CCAClass.py` 和 `AlgorithmImplement_SSVEP.py` 重构，保留以下核心设计：

- 六个Chebyshev-I型滤波子带；
- 子带权重 `(m + 1)^(-1.25) + 0.24`；
- QR分解后计算EEG子空间和参考子空间的奇异值；
- 奇异值权重 `[1.4, 0.37, 0.10, 0.3, -0.1]`；
- 单子带相关分数取四次方后进行滤波器组融合；
- 使用第一名和第二名分数差作为动态停止置信度；
- 动态时间窗为 `0.92、1.08、1.16、1.24、1.32、1.48秒`。

原文件针对40类赛事数据硬编码。本实现将类别数、频率、相位、通道和参考模板参数化，可以直接读取 `NewFunc2/config.json` 的20目标配置。

## 文件

- `ds_msv_fbcca.py`：MetaBCI风格算法类，提供 `fit/transform/predict`。
- `online_ssvep_neuracle_ds_msv_fbcca.py`：博睿康动态停止在线示例。

## 算法接口

```python
model = DSMSVFBCCA(filterbank=filterbank)
model.fit(Yf=references)
scores = model.transform(X)
labels = model.predict(X)
labels, confidence = model.predict_with_confidence(X)
```

输入 `X` 的形状为：

```text
试次数 × 通道数 × 采样点数
```

`predict()` 与MetaBCI的CCA类一致，返回从0开始的类别编号。

## 启动在线示例

```powershell
python demos\SwarmRobotDemos\algo\online_ssvep_neuracle_ds_msv_fbcca.py
```

默认接口与原在线示例一致：

- 博睿康DataService：`127.0.0.1:8712`
- 默认脑电通道：`0-7`（忠实保留原算法的8通道选择）
- LSL source ID：`meta_online_worker`
- 正常判决：`1-20`
- 失败标签：`21`
- 最新结果文件：`result.txt`

## 注意

压缩包中的动态停止阈值是针对其原始40类数据标定的。本实现为忠实复现而保留这些阈值，但应用到20类刺激和新受试者前，应使用实际数据重新标定阈值。

当前刺激端仍按固定刺激时长显示。算法可以提前得到并回传结果，但若要真正缩短每轮视觉刺激时间，还需要刺激端支持收到提前判决后结束当前闪烁。
