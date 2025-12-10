import numpy as np
from commands import SAMPLES_PER_FRAME, USB_SAMPLES_PER_PACKET

def read_exact(ser, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            continue
        buf.extend(chunk)
    return bytes(buf)

def read_packet(ser, packet_samples):
    # sync to AB CD
    while True:
        if ser.read(1) != b'\xAB':
            continue
        if ser.read(1) != b'\xCD':
            continue
        break

    # read sequence number
    seq_lo = ser.read(1)[0]
    seq_hi = ser.read(1)[0]
    seq = seq_lo | (seq_hi << 8)

    # read samples
    payload = read_exact(ser, packet_samples * 2)
    samples = np.frombuffer(payload, dtype=np.uint16)

    return seq, samples


def read_adc_frame(ser):
    total = SAMPLES_PER_FRAME
    ps = USB_SAMPLES_PER_PACKET
    packets = total // ps

    out = np.zeros(total, dtype=np.uint16)
    expected_seq = None
    idx = 0

    for _ in range(packets):
        seq, samples = read_packet(ser, ps)

        if expected_seq is None:
            expected_seq = seq
        else:
            if seq != expected_seq:
                print("WARNING: Packet lost or misaligned (got", seq, "expected", expected_seq, ")")

        out[idx:idx+ps] = samples
        idx += ps
        expected_seq = (expected_seq + 1) & 0xFFFF

    return out
