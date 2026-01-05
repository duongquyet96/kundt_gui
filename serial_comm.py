import struct, time
from commands import *


MAX_PLEN = 64

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

def read_frame(ser, timeout=2.0, rx_buf=None, accept=None):
    """
    Robust framed reader:
      - tolerates garbage/other packet types in stream
      - requires CRC to match before returning
      - keeps leftover bytes in rx_buf across calls
    """
    if rx_buf is None:
        rx_buf = bytearray()

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        # Read as much as is available (faster + less timing sensitive than 1 byte)
        rx_buf.extend(ser.read(ser.in_waiting or 1))

        while True:
            idx = rx_buf.find(b"\xAA\x55")
            if idx < 0:
                # prevent unbounded growth if no header appears
                if len(rx_buf) > 2048:
                    del rx_buf[:-2]
                break

            if idx > 0:
                del rx_buf[:idx]

            if len(rx_buf) < 4:
                break  # need cmd/status + len

            cmd_or_status = rx_buf[2]
            plen = rx_buf[3]

            # Length sanity check: reject impossible frames quickly
            if plen > MAX_PLEN:
                del rx_buf[:2]      # drop this header and keep searching
                continue

            total = 6 + plen
            if len(rx_buf) < total:
                break  # wait for more bytes

            frame = bytes(rx_buf[:total])
            del rx_buf[:total]

            # CRC check here (critical): ignore false headers / other packet types
            crc_rx = struct.unpack("<H", frame[4+plen:6+plen])[0]
            crc_calc = crc16_arc(frame[2:4+plen])
            if crc_rx != crc_calc:
                continue

            payload = frame[4:4+plen]
            if accept and not accept(cmd_or_status, payload):
                continue

            return frame, rx_buf

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

_rx_buf = bytearray()

def send_command(ser, cmd, payload=b"", timeout=2.0):
    # Avoid reset_input_buffer() during scans/streaming; it can desync you.
    ser.write(build_frame(cmd, payload))
    ser.flush()

    try:
        # If you know your reply status range, filter it here to skip non-replies.
        # Example: accept only ack/error codes:
        frame, _ = read_frame(
            ser, timeout=timeout, rx_buf=_rx_buf,
            accept=lambda status, pl: status in (0xFE, 0xFF)  # adapt to your protocol
        )
        return parse_frame(frame)
    except Exception as e:
        print("Command error:", e)
        return None, None
