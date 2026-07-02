# 博睿康模拟板在线实验操作文档

## 1. 功能说明

本模拟系统用于在没有真实博睿康放大器和 Trigger Box 的情况下，联调以下三个模块：

- 模拟采集端：`brainco_mock_board.py`
- 在线算法端：`algo/online_ssvep_neuracle_fbscca.py`
- 刺激端：`stim/NewFunc2/stim_customized_service.py`（自动识别真实或虚拟 COM4）

模拟板默认提供 64 路 EEG 通道和 1 路 Trigger 通道，与在线算法的 65 通道默认配置一致。

每轮实验的处理逻辑如下：

1. 刺激端通过虚拟 COM4 发送 Trigger `240`。
2. 模拟板开始缓存本轮 EEG 数据，并从 `4、4、1、2` 序列中选择当前指令。
3. 模拟板生成该指令对应频率的 SSVEP 信号。
4. 刺激结束时，刺激端通过虚拟 COM4 发送 Trigger `241`。
5. 模拟板停止缓存，将完整试次按 Neuracle TCP float32 格式发送给算法端。
6. 发给算法的 Trigger 通道写入当前指令编号，例如 `4`，而不是控制触发值 `240/241`。
7. FBSCCA 完成识别，并通过 LSL 和 `algo/result.txt` 输出结果。

默认指令与频率的关系为：

| 试次顺序 | 指令编号 | SSVEP 频率 |
| --- | ---: | ---: |
| 1 | 4 | 8.6 Hz |
| 2 | 4 | 8.6 Hz |
| 3 | 1 | 8.0 Hz |
| 4 | 2 | 8.2 Hz |

第四轮结束后，序列会从第一项开始循环。

## 2. 文件说明

| 文件 | 用途 |
| --- | --- |
| `brainco_mock_board.py` | 模拟博睿康采集设备、缓存试次并提供 TCP 数据服务 |
| `algo/online_ssvep_neuracle_fbscca.py` | 接收 65 通道数据并执行在线 FBSCCA |
| `stim/NewFunc2/stim_customized_service.py` | 原始刺激程序；模拟板运行时自动使用虚拟 COM4，否则使用真实 COM4 |
| `stim/online_ssvep_paradigm.py` | 自动检测模拟板虚拟 COM4，并在虚拟与真实串口之间切换 |
| `stim/NewFunc2/stim_customized_service_mock.py` | 保留的显式模拟入口，通常不再需要使用 |
| `stim/NewFunc2/virtual_com4_online_ssvep.py` | 将刺激端事件转换为 240/241，并发送到模拟板 |

运行期间，模拟板会动态创建：

```text
demos/SwarmRobotDemos/brainco_mock_com4.json
```

该文件表示虚拟 COM4 的占用状态和 UDP 地址。模拟板正常关闭后，状态会变为 `available`。它是运行时状态文件，不需要手工创建。

## 3. 环境准备

请使用已经安装 MetaBCI 在线实验依赖的 Python 环境。至少需要以下包：

- `numpy`
- `scipy`
- `scikit-learn`
- `pylsl`
- `mne`
- `psychopy`
- `pyserial`

进入项目根目录：

```powershell
Set-Location "F:\work\脑机接口\2026wrcc\MetaBCI-master"
```

检查关键依赖：

```powershell
python -c "import numpy, scipy, pylsl, mne, psychopy, serial; print('依赖检查通过')"
```

如果出现 `ModuleNotFoundError`，请先切换到平时运行 MetaBCI/PsychoPy 的 Python 环境。不要在缺少 `pylsl`、`mne` 或 `psychopy` 的基础环境中启动在线实验。

## 4. 启动流程

需要打开三个 PowerShell 终端，并严格按照以下顺序启动。

### 4.1 启动模拟板

在第一个终端运行：

```powershell
python demos/SwarmRobotDemos/brainco_mock_board.py
```

正常启动时会看到类似输出：

```text
Mock BrainCo board online: data=tcp://127.0.0.1:8712, channels=65 (64 EEG + Trigger)
virtual COM4 occupied: udp://127.0.0.1:8713; trigger protocol 240=start, 241=stop
synthetic command sequence: 4 -> 4 -> 1 -> 2
```

此时模拟板已经占用：

- TCP `127.0.0.1:8712`：向在线算法发送 EEG 数据。
- UDP `127.0.0.1:8713`：模拟 COM4 Trigger Box。

保持该终端运行，不要关闭。

### 4.2 启动在线算法

在第二个终端运行：

```powershell
python demos/SwarmRobotDemos/algo/online_ssvep_neuracle_fbscca.py
```

算法会连接模拟板的 TCP 8712 端口，并创建：

```text
LSL source_id=meta_online_worker
```

模拟板终端应出现类似输出：

```text
MetaBCI Neuracle client connected from 127.0.0.1:xxxxx
```

算法端提示等待刺激程序连接 LSL 时属于正常状态。保持该终端运行。

### 4.3 启动原始刺激程序

在第三个终端运行：

```powershell
python demos/SwarmRobotDemos/stim/NewFunc2/stim_customized_service.py
```

刺激范式会先检查模拟板发布的 COM4 状态，并通过 UDP 握手确认模拟板确实在线。握手成功后会显示：

```text
[刺激端][触发] 检测到模拟板，COM4 已由虚拟 Trigger Box 占用
```

此时不会调用 pyserial 打开物理串口，而是自动通过虚拟 COM4 发送 `240/241`。因此模拟实验和真实实验可以使用同一个 `stim_customized_service.py` 启动文件。

原有的 `stim_customized_service_mock.py` 仍然保留，可用于显式模拟，但正常联调不再需要使用。

进入刺激界面后，按照界面提示开始实验。

### 4.4 真实博睿康硬件启动路径

真实实验与模拟实验是两条独立路径。使用真实博睿康设备时：

- 不运行 `brainco_mock_board.py`。
- 不运行 `stim_customized_service_mock.py`。
- 不使用 UDP 8713 和应用级虚拟 COM4。
- 使用博睿康 Neusen W/DataService、真实 Trigger Box 和原始刺激程序。

真实硬件的数据路径如下：

```text
真实脑电帽/博睿康放大器
  -> Neusen W 采集软件
  -> DataService TCP 8712
  -> online_ssvep_neuracle_fbscca.py
  -> FBSCCA 预测
  -> LSL source_id=meta_online_worker
  -> stim_customized_service.py

stim_customized_service.py
  -> 真实 COM4
  -> 博睿康 Trigger Box
  -> Trigger 写入真实 EEG 最后一个通道
```

真实硬件建议按照以下顺序启动。

#### 第一步：连接和检查真实硬件

1. 连接脑电帽、博睿康放大器和采集主机。
2. 将博睿康 Trigger Box 连接到刺激计算机。
3. 在 Windows 设备管理器中确认 Trigger Box 的串口号为 `COM4`。
4. 如果实际串口号不是 COM4，修改原始刺激文件中的 `port_addr`：

```text
demos/SwarmRobotDemos/stim/NewFunc2/stim_customized_service.py
```

对应配置位置：

```python
port_addr="COM4"
```

#### 第二步：启动 Neusen W 和 DataService

在博睿康采集主机上启动 Neusen W，确认 EEG 波形可以正常显示，然后打开软件中的 DataService。

DataService 参数应与算法一致：

| 参数 | 建议值 |
| --- | --- |
| TCP 端口 | `8712` |
| 采样率 | `1000 Hz` |
| 数据通道 | `64` 路 EEG |
| Trigger 通道 | 最后 1 路 |
| 算法总通道数 | `65` |

如果 DataService 和算法运行在同一台计算机上，设备地址使用：

```text
127.0.0.1:8712
```

如果 DataService 位于另一台采集主机上，需要记录采集主机的局域网 IPv4 地址，例如：

```text
192.168.1.100:8712
```

同时确认两台计算机网络互通，并允许 TCP 8712 通过防火墙。

#### 第三步：启动真实在线算法

进入项目根目录：

```powershell
Set-Location "F:\work\脑机接口\2026wrcc\MetaBCI-master"
```

DataService 与算法位于同一台计算机时运行：

```powershell
python demos/SwarmRobotDemos/algo/online_ssvep_neuracle_fbscca.py `
  --host 127.0.0.1 `
  --port 8712 `
  --sample-rate 1000 `
  --num-channels 65
```

DataService 位于其他采集主机时，将 `--host` 改为真实采集主机 IP：

```powershell
python demos/SwarmRobotDemos/algo/online_ssvep_neuracle_fbscca.py `
  --host 192.168.1.100 `
  --port 8712 `
  --sample-rate 1000 `
  --num-channels 65
```

请将示例中的 `192.168.1.100` 替换为实际 IP。

算法连接成功后，应看到博睿康 TCP 连接成功、FBSCCA Worker 启动以及 LSL 流创建等日志。

#### 第四步：启动真实刺激程序

在刺激计算机上使用原始文件启动：

```powershell
python demos/SwarmRobotDemos/stim/NewFunc2/stim_customized_service.py
```

不要使用以下模拟入口：

```text
stim/NewFunc2/stim_customized_service_mock.py
```

刺激程序将打开真实 COM4，通过博睿康 Trigger Box 发送刺激编号，并通过 `source_id=meta_online_worker` 接收算法预测。

#### 第五步：开始实验

确认以下状态后再开始刺激：

- Neusen W 正常显示 EEG。
- DataService 已开启 TCP 8712。
- 算法端已连接 DataService。
- 算法端已创建 `meta_online_worker` LSL 流。
- 刺激端成功打开真实 COM4。
- 刺激端成功连接算法的 LSL 流。

真实实验的完整启动命令顺序可简写为：

```text
1. 启动 Neusen W
2. 在 Neusen W 中启动 DataService（TCP 8712）
3. python demos/SwarmRobotDemos/algo/online_ssvep_neuracle_fbscca.py
4. python demos/SwarmRobotDemos/stim/NewFunc2/stim_customized_service.py
```

真实模式下，算法直接读取博睿康数据流最后一个通道中的 `1-20` 刺激编号。模拟模式使用的 `240/241` 是模拟板的试次控制协议，模拟板会在发送数据前将其转换成当前 `1-20` 指令编号；真实硬件路径不经过这层转换。

## 5. 单轮数据流程

单轮刺激期间，各模块的执行顺序如下：

```text
刺激首帧
  -> 虚拟 COM4 发送 240
  -> 模拟板选择下一条指令
  -> 模拟板实时生成并缓存 SSVEP
  -> 刺激持续约 2 秒
  -> 虚拟 COM4 发送 241
  -> 模拟板封装并通过 TCP 发送完整试次
  -> 算法从 Trigger 通道检测 1-20 指令号
  -> 截取 0.14-1.14 秒数据窗
  -> FBSCCA 预测
  -> LSL 返回刺激端
  -> 覆盖写入 algo/result.txt
```

例如第一轮应看到：

```text
[刺激端][虚拟COM4] 已发送 Trigger=240
trial 1 started: command=4, frequency=8.6 Hz
[刺激端][虚拟COM4] 已发送 Trigger=241
trial 1 stopped ... queued for TCP
sending trial 1 to algorithm: command=4 ...
```

后续三轮的 `command` 应依次为：

```text
4, 1, 2
```

合并第一轮后，完整顺序为 `4, 4, 1, 2`。

## 6. 手工测试 Trigger

模拟板运行时，可以在其他终端手工发送控制 Trigger。

发送开始 Trigger：

```powershell
python demos/SwarmRobotDemos/brainco_mock_board.py --trigger 240 --binary-trigger
```

发送停止 Trigger：

```powershell
python demos/SwarmRobotDemos/brainco_mock_board.py --trigger 241 --binary-trigger
```

其中 `--binary-trigger` 表示使用真实 Neuracle Trigger Box 的五字节格式：

```text
01 E1 01 00 XX
```

也可以省略 `--binary-trigger`，此时会发送 ASCII 数字，适合简单调试。

## 7. 参数配置

查看模拟板所有参数：

```powershell
python demos/SwarmRobotDemos/brainco_mock_board.py --help
```

常用参数如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--host` | `127.0.0.1` | TCP 和虚拟 Trigger 监听地址 |
| `--data-port` | `8712` | Neuracle TCP 数据端口 |
| `--trigger-port` | `8713` | 虚拟 COM4 UDP 端口 |
| `--virtual-com-name` | `COM4` | 虚拟串口显示名称 |
| `--srate` | `1000` | 模拟采样率 |
| `--eeg-chans` | `64` | EEG 通道数，不包括 Trigger |
| `--packet-samples` | `40` | 每个 TCP 数据块的采样点数 |
| `--sequence` | `4,4,1,2` | 循环生成的指令序列 |
| `--minimum-trial-seconds` | `1.2` | 过短试次的最小补齐时长 |
| `--verbose-stream` | 关闭 | 每秒显示发送进度 |

例如，将测试序列改为 `1、2、3、4`：

```powershell
python demos/SwarmRobotDemos/brainco_mock_board.py --sequence 1,2,3,4
```

如果修改 TCP 端口，需要同步修改算法端：

```powershell
python demos/SwarmRobotDemos/brainco_mock_board.py --data-port 9000
python demos/SwarmRobotDemos/algo/online_ssvep_neuracle_fbscca.py --port 9000
```

如果修改虚拟 Trigger UDP 端口，不需要修改原始刺激程序；刺激端会读取模拟板运行时状态文件中公布的端口。

## 8. 正常停止

建议按照以下顺序停止：

1. 退出刺激程序。
2. 在算法终端按 Enter，等待算法关闭 Worker 和 TCP 连接。
3. 在模拟板终端输入 `q` 并按 Enter。

模拟板正常退出后会释放 TCP、UDP 端口，并将虚拟 COM4 状态写为 `available`。

如果刺激正在进行时直接关闭程序，虚拟适配器会尽量补发 Trigger `241`，避免模拟板长时间停留在采集状态。

## 9. 常见问题

### 9.1 算法提示连接被拒绝

可能原因：模拟板未启动，或者算法与模拟板的 TCP 端口不一致。

处理方法：

1. 先确认模拟板终端存在 `data=tcp://127.0.0.1:8712`。
2. 再启动算法。
3. 检查算法参数 `--host` 和 `--port`。

### 9.2 提示虚拟 COM4 不可用

如果模拟板没有运行，原始刺激程序会回退到真实 COM4。无真实硬件时可能出现类似提示：

```text
SerialException: could not open port 'COM4'
```

处理方法：先启动模拟板，确认终端显示 `virtual COM4 occupied`，然后重新启动原始 `stim_customized_service.py`。

### 9.3 端口 8712 或 8713 已被占用

说明之前的模拟板仍在运行，或者其他程序占用了端口。

PowerShell 检查命令：

```powershell
Get-NetTCPConnection -LocalPort 8712 -ErrorAction SilentlyContinue
Get-NetUDPEndpoint -LocalPort 8713 -ErrorAction SilentlyContinue
```

关闭对应旧进程后重新启动。也可以使用其他端口，但必须同步修改相关模块参数。

### 9.4 刺激端找不到 LSL 流

确认：

- 在线算法已经启动。
- 算法端已经创建 `source_id=meta_online_worker` 的 LSL 流。
- 防火墙没有阻止本机 LSL 通信。
- 刺激端和算法端使用同一个 `lsl_source_id`。

### 9.5 算法没有检测到 Trigger

检查模拟板终端是否依次出现：

```text
received trigger 240
trial ... started
received trigger 241
trial ... queued for TCP
sending trial ... to algorithm
```

同时检查算法参数：

```text
--sample-rate 1000
--num-channels 65
--eeg-channels 0,1,...,63
```

最后一个通道必须作为 Trigger，不能被列入 EEG 通道。

### 9.6 预测结果不是 4、4、1、2

先观察模拟板日志中的 `command` 是否正确。如果模拟板日志正确但算法结果错误，请检查：

- 算法采样率是否为 1000 Hz。
- 通道数是否为 65。
- 算法数据窗是否仍为 0.14-1.14 秒。
- 是否同时运行了第二个算法进程或第二个模拟板。

当前合成信号已通过项目现有 FBSCCA 解码器验证，标准配置下的预测顺序为 `4、4、1、2`。

## 10. 使用限制

- 虚拟 COM4 是应用级模拟，不会在 Windows 设备管理器中创建真实 COM4 设备。
- 虚拟串口通信实际使用本机 UDP，默认端口为 8713。
- 如果必须让未经修改的第三方串口软件直接打开 COM4，需要另外安装虚拟串口驱动并建立端口对；当前方案不需要管理员权限或驱动安装。
- 真实和模拟实验均可使用原始 `stim_customized_service.py`：模拟板在线时自动走虚拟 COM4，模拟板离线时打开真实 COM4。
