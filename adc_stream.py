import numpy as np
from commands import SAMPLES_PER_FRAME

def read_exact(ser, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            continue
        buf.extend(chunk)
    return bytes(buf)

def read_adc_frame(ser):
    # Wait for header
    while True:
        b = ser.read(1)
        if b != b'\xAA':
            continue
        if ser.read(1) == b'\x55':
            break

    # Read full frame safely
    data = read_exact(ser, SAMPLES_PER_FRAME * 2)
    return np.frombuffer(data, dtype=np.uint16)
