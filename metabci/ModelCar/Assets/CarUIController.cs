using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarUIController : MonoBehaviour
{
    public CarController car;
    public GimbalController gimbal;

    // 小车移动
    public void OnForwardClick()  { car.Move(Vector3.forward); }
    public void OnBackwardClick() { car.Move(Vector3.back); }
    public void OnLeftClick()     { car.Move(Vector3.left); }
    public void OnRightClick()    { car.Move(Vector3.right); }
    public void OnTurnLeftClick() { car.Turn(-car.turnStep); }
    public void OnTurnRightClick(){ car.Turn(car.turnStep); }
    public void OnSpinStartButtonClicked(){car.StartSpin();}
    public void OnSpinStopButtonClicked(){car.StopSpin();}
    // 云台控制
    public void OnGimbalUpClick()     { gimbal.LookUp(); }
    public void OnGimbalForwardClick(){ gimbal.LookForward(); }
    public void OnGimbalDownClick()   { gimbal.LookDown(); }
}
