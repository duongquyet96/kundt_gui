import struct, time, serial
from commands import *

def crc16_arc(data: bytes) -> int:
    crc = 0x0000
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else crc >> 1
    return crc & 0xFFFF

def build_frame(cmd: int, payload: bytes = b'') -> bytes:
    header = b'\xAA\x55'
    body = bytes([cmd & 0xFF, len(payload) & 0xFF]) + payload
    crc = crc16_arc(body)
    return header + body + struct.pack("<H", crc)

def read_frame(ser, timeout=1.0):
    start = time.time()
    buf = bytearray()

    while time.time() - start < timeout:
        b = ser.read(1)
        if not b:
            continue
        buf += b

        idx = buf.find(b"\xAA\x55")
        if idx >= 0:
            if idx > 0:
                del buf[:idx]
            if len(buf) >= 4:
                plen = buf[3]
                total = 6 + plen
                if len(buf) < total:
                    buf += ser.read(total - len(buf))
                if len(buf) >= total:
                    return bytes(buf[:total])

    raise TimeoutError("Frame timeout")

def parse_frame(resp):
    if resp[0:2] != b"\xAA\x55":
        raise ValueError("Bad header")
    status = resp[2]
    plen = resp[3]
    payload = resp[4:4+plen]
    crc_rx = struct.unpack("<H", resp[4+plen:6+plen])[0]
    crc_calc = crc16_arc(resp[2:4+plen])
    if crc_rx != crc_calc:
        raise ValueError("CRC mismatch")
    return status, payload

def send_command(ser, cmd, payload=b""):
    ser.reset_input_buffer()
    ser.write(build_frame(cmd, payload))

    try:
        resp = read_frame(ser)
        return parse_frame(resp)
    except Exception as e:
        print("Command error:", e)
        return None, None
