# MetaBCI + Unity 小车控制系统

## 系统概述

本系统实现了通过脑机接口(BCI)控制Unity中的虚拟小车。系统由两部分组成：

1. **MetaBCI控制程序** (`MetaBCI/demos/direction_control_demos/online_analysis.py`)
2. **Unity小车控制界面** (`test/`)

## 预测标签分析

### 标签范围
- **预测标签范围**: 1-40
- **对应刺激布局**: 5行×8列 = 40个SSVEP刺激目标

### 标签映射方案（基于paradigm.py指令集）

根据paradigm.py中的指令集，标签映射如下：

```
标签 1-12:  无效指令 (AB)
标签 13:    左转30度 -> OnTurnLeftClick()
标签 14:    左行50cm -> OnLeftClick()
标签 15:    停止旋转 -> OnSpinStopButtonClicked()
标签 16:    无效指令 (AB)
标签 17:    前进50cm -> OnForwardClick()
标签 18:    旋转 -> OnSpinStartButtonClicked()
标签 19:    后退50cm -> OnBackwardClick()
标签 20:    无效指令 (AB)
标签 21:    右转30度 -> OnTurnRightClick()
标签 22:    右行50cm -> OnRightClick()
标签 23:    平视 -> OnGimbalForwardClick()
标签 24:    无效指令 (AB)
标签 25:    抬头 -> OnGimbalUpClick()
标签 26:    头左转30度 -> 云台左转（待实现）
标签 27:    头右转30度 -> 云台右转（待实现）
标签 28:    低头 -> OnGimbalDownClick()
标签 29:    无效指令 (AB)
标签 30:    开启发射 -> 开启发射（待实现）
标签 31:    攻击一次 -> 攻击（待实现）
标签 32:    关闭发射 -> 关闭发射（待实现）
标签 33-40: 无效指令 (AB)
```

### 有效控制指令
- **移动控制**: 前进、后退、左移、右移、左转、右转
- **旋转控制**: 开始旋转、停止旋转
- **视角控制**: 抬头、平视、低头
- **特殊功能**: 头左转、头右转、开启发射、攻击、关闭发射（待实现）

### 数据传输
- **传输协议**: LSL (Lab Streaming Layer)
- **流名称**: `meta_feedback`
- **数据类型**: int32
- **实时性**: 支持实时数据传输

## 文件说明

### Unity部分
- `Assets/CarController.cs`: 小车物理控制脚本
- `Assets/CarUIController.cs`: 小车UI控制脚本
- `Assets/GimbalController.cs`: 云台控制脚本
- `Assets/LSLReceiver.cs`: LSL数据接收和控制脚本

### 测试工具
- `lsl_test_sender.py`: Python LSL测试发送器

## 使用方法

### 1. 启动Unity项目
1. 打开Unity，加载`test`项目
2. 在场景中添加`LSLReceiver`组件到GameObject
3. 将`CarController`和`CarUIController`组件拖拽到`LSLReceiver`的相应字段

### 2. 测试模式
- 设置`LSLReceiver.simulateData = true`进行模拟测试
- 系统会每2秒随机发送一个有效控制指令

### 3. 实际BCI控制
1. 安装LSL Unity插件: https://github.com/labstreaminglayer/lsl-unity
2. 设置`LSLReceiver.simulateData = false`
3. 运行MetaBCI的`online_analysis.py`
4. Unity会自动接收并执行BCI指令

### 4. 测试LSL连接
```bash
# 安装pylsl
pip install pylsl

# 运行测试发送器
python lsl_test_sender.py
```

## 系统架构

```
MetaBCI (Python)
    ↓ LSL传输
预测标签 (1-40)
    ↓ Unity接收
LSLReceiver.cs
    ↓ 标签映射
CarUIController.cs
    ↓ 执行控制
CarController.cs / GimbalController.cs
    ↓ 物理更新
Unity小车/云台
```

## 自定义配置

### 修改标签映射
在`LSLReceiver.cs`的`InitializeLabelMapping()`方法中修改映射关系：

```csharp
// 示例：修改映射关系
labelToAction[13] = "turnLeft";      // 左转
labelToAction[14] = "left";          // 左移
// ... 其他映射
```

### 添加新功能
1. 在对应的Controller中添加新方法
2. 在`CarUIController.cs`中添加对应的UI方法
3. 在`LSLReceiver.cs`中更新标签映射和ExecuteAction方法

### 调整控制参数
- `controlInterval`: 控制间隔时间
- `moveStep`: 小车移动步长
- `turnStep`: 小车转向角度

## 待实现功能

以下功能需要进一步开发：

1. **云台水平旋转**:
   - 头左转30度 (标签26)
   - 头右转30度 (标签27)

2. **武器系统**:
   - 开启发射 (标签30)
   - 攻击一次 (标签31)
   - 关闭发射 (标签32)

## 故障排除

### 1. LSL连接问题
- 确保MetaBCI程序正在运行
- 检查防火墙设置
- 验证LSL插件是否正确安装

### 2. 小车不响应
- 检查`LSLReceiver`组件是否正确配置
- 验证`CarController`和`CarUIController`引用
- 查看Unity Console中的调试信息

### 3. 标签映射错误
- 确认标签范围在1-40之间
- 检查映射字典是否正确初始化
- 验证动作名称与`CarUIController`方法名一致

## 扩展功能

### 添加新的控制动作
1. 在`CarController.cs`或`GimbalController.cs`中添加新方法
2. 在`CarUIController.cs`中添加对应的UI方法
3. 在`LSLReceiver.cs`中更新标签映射

### 支持更多标签
- 修改SSVEP刺激布局
- 更新标签映射逻辑
- 扩展控制指令集

## 技术细节

### LSL数据格式
```python
# MetaBCI发送的数据格式
p_labels = np.array([int(p_labels + 1)])  # 范围：1-40
outlet.push_sample(p_labels.tolist())
```

### Unity接收格式
```csharp
// Unity接收的数据格式
int label = Mathf.RoundToInt(sample[0]);  // 范围：1-40
```

### 实时性要求
- LSL传输延迟: < 10ms
- Unity处理延迟: < 50ms
- 总体响应时间: < 100ms 