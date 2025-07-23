using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine;

public class CarController : MonoBehaviour
{
    public float moveStep = 0.5f;   // 50cm
    public float turnStep = 30f;    // 30度

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    // 前后左右移动
    public void Move(Vector3 direction)
    {
        Vector3 move = direction.normalized * moveStep;
        rb.MovePosition(rb.position + transform.TransformDirection(move));
    }

    // 左右转向
    public void Turn(float angle)
    {
        Quaternion turnRotation = Quaternion.Euler(0f, angle, 0f);
        rb.MoveRotation(rb.rotation * turnRotation);
    }

    private bool isSpinning = false;
    public float spinSpeed = 180f; // 每秒旋转角度

    void Update()
    {
        if (isSpinning)
        {
            float angle = spinSpeed * Time.deltaTime;
            Turn(angle);
        }
    }

    // ... existing code ...

    // 开始自旋
    public void StartSpin()
    {
        isSpinning = true;
    }

    // 停止自旋
    public void StopSpin()
    {
        isSpinning = false;
    }
}