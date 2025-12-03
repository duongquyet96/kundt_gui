from commands import *
from serial_comm import send_command

def digipot_set(ser, value):
    v = max(0, min(99, value))
    print(f"Setting digipot to {v}/99")
    status, _ = send_command(ser, CMD_DIGIPOT_SET, bytes([v]))
    print("OK" if status == STS_ACK else "ERROR")
