# scan_kundt.py
import struct
import numpy as np
from serial_comm import send_command
from adc_stream import read_adc_frame
from motor_control import home_motor, wait_until_home_complete, move_to_mm, wait_until_move_complete
from fft_tools import fft_mag_at_freq
from commands import CMD_START_SAMPLING, CMD_STOP_SAMPLING, FS

def scan_kundt(ser, f_hz, start_mm, end_mm, step_mm):
    """
    Perform Kundt tube scan.
    Returns: positions (list), magnitudes (numpy array), results dict
    results = { 'p_max', 'p_min', 'idx_max', 'idx_min', 'SWR', 'R_mag' }
    """

    if step_mm <= 0:
        raise ValueError("step_mm must be > 0")
    if end_mm <= start_mm:
        raise ValueError("end_mm must be > start_mm")

    # --- Homing ---
    home_motor(ser)
    if not wait_until_home_complete(ser):
        raise RuntimeError("Homing timeout")

    # --- Move to start ---
    move_to_mm(ser, start_mm)
    if not wait_until_move_complete(ser):
        raise RuntimeError("Move-to-start timeout")

    positions = []
    magnitudes = []

    x = start_mm
    while x <= end_mm + 1e-9:
        positions.append(x)

        # Capture ADC frame: discard first frame to avoid misalignment
        send_command(ser, CMD_START_SAMPLING)
        _ = read_adc_frame(ser)        # discard one frame
        samples = read_adc_frame(ser)  # use second frame
        send_command(ser, CMD_STOP_SAMPLING)

        if samples is None:
            raise RuntimeError("ADC error during scan")

        mag, f_bin = fft_mag_at_freq(samples, FS, f_hz)
        magnitudes.append(mag)

        x += step_mm
        if x > end_mm:
            break
        move_to_mm(ser, x)
        if not wait_until_move_complete(ser):
            raise RuntimeError("Move timeout during scan")

    mag_arr = np.array(magnitudes)
    idx_max = int(np.argmax(mag_arr))
    idx_min = int(np.argmin(mag_arr))
    p_max = float(mag_arr[idx_max])
    p_min = float(mag_arr[idx_min])
    SWR = p_max / p_min
    R_mag = (SWR - 1.0) / (SWR + 1.0)

    results = {
        "p_max": p_max,
        "p_min": p_min,
        "idx_max": idx_max,
        "idx_min": idx_min,
        "SWR": SWR,
        "R_mag": R_mag,
    }

    return positions, mag_arr, results
