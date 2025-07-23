using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using LSL;

public class LSLReceiver : MonoBehaviour
{
    public CarController car;
    public CarUIController carUI;

    public float controlInterval = 0.1f;
    private float lastControlTime = 0f;

    private Dictionary<int, string> labelToAction = new Dictionary<int, string>();
    private int lastReceivedLabel = 0;

    public bool simulateData = false; // 默认关闭模拟，使用LSL

    // LSL相关
    private StreamInlet inlet;
    private const string streamName = "meta_feedback"; // Python端推送的流名
    private const string streamSourceId = "meta_online_worker"; // Python端source_id

    void Start()
    {
        InitializeLabelMapping();
        if (simulateData)
        {
            StartCoroutine(SimulateData());
        }
        else
        {
            StartCoroutine(ConnectToLSL());
        }
    }

    void InitializeLabelMapping()
    {
        for (int i = 1; i <= 40; i++)
        {
            labelToAction[i] = "none";
        }
        // 范式指令与方法映射
        labelToAction[1] = "turnLeft";         // "左转\n30度"
        labelToAction[2] = "left";             // "左行\n50cm"
        labelToAction[3] = "stopSpin";         // "停止\n旋转"
        labelToAction[4] = "forward";          // "前进\n50cm"
        labelToAction[5] = "startSpin";        // "旋转"
        labelToAction[6] = "backward";         // "后退\n50cm"
        labelToAction[7] = "turnRight";        // "右转\n30度"
        labelToAction[8] = "right";            // "右行\n50cm"
        labelToAction[9] = "lookUp";           // "抬头"
        labelToAction[10] = "lookForward";     // "低头"
        // 其余按需补充
    }

    IEnumerator SimulateData()
    {
        while (true)
        {
            yield return new WaitForSeconds(2f);
            // 只模拟1~10号标签，对应范式指令
            int randomLabel = Random.Range(1, 11);
            ProcessReceivedLabel(randomLabel);
        }
    }

    IEnumerator ConnectToLSL()
    {
        yield return new WaitForSeconds(1f); // 等待MetaBCI端启动
        Debug.Log("尝试连接LSL流...");

        // 查找流
        var results = LSL.LSL.resolve_stream("name", streamName, 1, 5.0);
        if (results.Length == 0)
        {
            Debug.LogError("未找到名为 " + streamName + " 的LSL流！");
            yield break;
        }
        inlet = new StreamInlet(results[0]);
        Debug.Log("已连接到LSL流: " + streamName);

        float[] sample = new float[1];
        while (true)
        {
            double timestamp = inlet.pull_sample(sample, 0.0f);
            if (timestamp > 0)
            {
                int label = (int)sample[0];
                OnDataReceived(label);
            }
            yield return null;
        }
    }

    void Update()
    {
        if (Time.time - lastControlTime > controlInterval)
        {
            if (lastReceivedLabel > 0)
            {
                ProcessReceivedLabel(lastReceivedLabel);
                lastReceivedLabel = 0;
            }
            lastControlTime = Time.time;
        }
    }

    public void ProcessReceivedLabel(int label)
    {
        Debug.Log($"接收到标签: {label}");
        if (labelToAction.ContainsKey(label) && labelToAction[label] != "none")
        {
            ExecuteAction(labelToAction[label]);
        }
        else
        {
            Debug.Log($"标签 {label} 无效或无对应动作");
        }
    }

    // LSL数据到达时调用
    public void OnDataReceived(int label)
    {
        lastReceivedLabel = label;
    }

    void ExecuteAction(string action)
    {
        switch (action)
        {
            case "forward":
                carUI.OnForwardClick();
                Debug.Log("执行: 前进");
                break;
            case "backward":
                carUI.OnBackwardClick();
                Debug.Log("执行: 后退");
                break;
            case "left":
                carUI.OnLeftClick();
                Debug.Log("执行: 左行");
                break;
            case "right":
                carUI.OnRightClick();
                Debug.Log("执行: 右行");
                break;
            case "turnLeft":
                carUI.OnTurnLeftClick();
                Debug.Log("执行: 左转");
                break;
            case "turnRight":
                carUI.OnTurnRightClick();
                Debug.Log("执行: 右转");
                break;
            case "startSpin":
                carUI.OnSpinStartButtonClicked();
                Debug.Log("执行: 旋转");
                break;
            case "stopSpin":
                carUI.OnSpinStopButtonClicked();
                Debug.Log("执行: 停止旋转");
                break;
            case "lookUp":
                carUI.OnGimbalUpClick();
                Debug.Log("执行: 抬头");
                break;
            case "lookForward":
                carUI.OnGimbalForwardClick();
                Debug.Log("执行: 低头");
                break;
            default:
                Debug.Log($"未知动作: {action}");
                break;
        }
    }

    // 手动测试按钮（只保留1~10号标签，对应范式指令）
    public void TestTurnLeft() { ProcessReceivedLabel(1); }      // 左转30度
    public void TestLeft() { ProcessReceivedLabel(2); }          // 左行50cm
    public void TestStopSpin() { ProcessReceivedLabel(3); }      // 停止旋转
    public void TestForward() { ProcessReceivedLabel(4); }       // 前进50cm
    public void TestSpin() { ProcessReceivedLabel(5); }          // 旋转
    public void TestBackward() { ProcessReceivedLabel(6); }      // 后退50cm
    public void TestTurnRight() { ProcessReceivedLabel(7); }     // 右转30度
    public void TestRight() { ProcessReceivedLabel(8); }         // 右行50cm
    public void TestLookUp() { ProcessReceivedLabel(9); }        // 抬头
    public void TestLookForward() { ProcessReceivedLabel(10); }  // 低头
} 