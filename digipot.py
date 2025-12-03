# digipot.py
from serial_comm import send_command
from commands import CMD_DIGIPOT_SET, STS_ACK

def digipot_set(ser, value):
    v = max(0, min(99, int(value)))
    status, _ = send_command(ser, CMD_DIGIPOT_SET, bytes([v]))
    if status != STS_ACK:
        print("Digipot set failed, status:", status)
