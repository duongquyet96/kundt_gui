import struct
import time
from serial_comm import send_command
from commands import *

def wait_until_home_complete(ser, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        status, _ = send_command(ser, 0x32)
        if status == STS_ACK:
            return True
        time.sleep(0.05)
    return False

def wait_until_move_complete(ser, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        status, _ = send_command(ser, 0x33)
        if status == STS_ACK:
            return True
        time.sleep(0.02)
    return False

def move_mm(ser, mm):
    status, _ = send_command(ser, CMD_STEPPER_MOVE, struct.pack("<f", mm))
    return status
