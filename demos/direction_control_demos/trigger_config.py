
class TriggerConfig:
    # trigger类型
    EEG_TYPE = 'neuracle'
    # EEG_TYPE = 'neuroscan'

    # trigger串/并口
    # TRIGGER_HANDLE = 'serial'
    TRIGGER_HANDLE = 'parallel'

    # 采用并口发送trigger(Neuracle/Neuroscan都支持)
    TRIGGER_PORT = 24568
    # TRIGGER_PORT='COM3'

    # ======================================事件及trigger定义======================================
    # trial开始trigger
    TRIAL_START_TRIGGER = 240

    # trial结束trigger
    TRIAL_END_TRIGGER = 241

    # block开始trigger
    BLOCK_START_TRIGGER = 242

    # block结束trigger
    BLOCK_END_TRIGGER = 243

    # 数据记录开始trigger
    RECORD_START_TRIGGER = 250

    # 数据记录结束trigger
    RECORD_END_TRIGGER = 251