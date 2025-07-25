using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GimbalController : MonoBehaviour
{
    // 云台的三个预设角度
    private float upAngle = 45f;    // 向上45度
    private float midAngle = 90f;   // 平视90度
    private float downAngle = 120f; // 向下120度

    private float targetAngle = 90f;

    void Update()
    {
        Vector3 euler = transform.localEulerAngles;
        float currentAngle = euler.x;
        if (currentAngle > 180) currentAngle -= 360;
        float newAngle = Mathf.Lerp(currentAngle, targetAngle, Time.deltaTime * 10f);
        transform.localEulerAngles = new Vector3(newAngle, euler.y, euler.z);
    }

    public void LookUp()
    {
        targetAngle = upAngle;
    }

    public void LookForward()
    {
        targetAngle = midAngle;
    }

    public void LookDown()
    {
        targetAngle = downAngle;
    }
}