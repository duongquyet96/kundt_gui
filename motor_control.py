# motor_control.py
import struct
import time
from serial_comm import send_command
from commands import CMD_HOME, CMD_STEPPER_MOVE, STS_ACK

def wait_until_home_complete(ser, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        status, _ = send_command(ser, 0x32)   # CMD_HOME_STATUS
        if status == STS_ACK:
            return True
        time.sleep(0.05)
    return False

def wait_until_move_complete(ser, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        status, _ = send_command(ser, 0x33)   # CMD_MOVE_STATUS
        if status == STS_ACK:
            return True
        time.sleep(0.02)
    return False

def home_motor(ser):
    status, _ = send_command(ser, CMD_HOME)
    return status

def move_to_mm(ser, pos_mm):
    payload = struct.pack('<f', float(pos_mm))
    status, _ = send_command(ser, CMD_STEPPER_MOVE, payload)
    return status
