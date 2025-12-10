# scan_kundt.py
import struct
import numpy as np
from serial_comm import send_command
from adc_stream import read_adc_frame
from motor_control import home_motor, wait_until_home_complete, move_to_mm, wait_until_move_complete
from fft_tools import fft_mag_at_freq
from commands import CMD_START_SAMPLING, CMD_STOP_SAMPLING, FS

def scan_kundt(ser, f_hz, start_mm, end_mm, step_mm, fs_hz):
    """
    Kundt tube scan (fast + accurate version).
    fs_hz comes from GUI (self.FS).
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

    # --- Start continuous ADC stream ---
    send_command(ser, CMD_START_SAMPLING)
    # Discard first frame (DMA alignment)
    _ = read_adc_frame(ser)

    x = start_mm
    while x <= end_mm + 1e-9:

        positions.append(x)

        # Read one frame only
        samples = read_adc_frame(ser)
        if samples is None:
            send_command(ser, CMD_STOP_SAMPLING)
            raise RuntimeError("ADC error during scan")

        # Compute FFT magnitude at frequency
        mag, f_bin = fft_mag_at_freq(samples, fs_hz, f_hz)
        magnitudes.append(mag)

        # Move to next point
        x += step_mm
        if x > end_mm:
            break

        move_to_mm(ser, x)
        if not wait_until_move_complete(ser):
            send_command(ser, CMD_STOP_SAMPLING)
            raise RuntimeError("Move timeout during scan")

    # Stop streaming
    send_command(ser, CMD_STOP_SAMPLING)

    # Convert to array
    mag_arr = np.array(magnitudes)

    # Find extrema
    idx_max = int(np.argmax(mag_arr))
    idx_min = int(np.argmin(mag_arr))
    p_max = float(mag_arr[idx_max])
    p_min = float(max(mag_arr[idx_min], 1e-9))   # avoid division by zero

    # Standing-wave ratio
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

def scan_kundt_no_home(ser, f_hz, start_mm, end_mm, step_mm, fs_hz):
    """
    SAME AS scan_kundt BUT WITHOUT HOMING or initial move-to-start.
    Assumes the mic is already at start_mm when the function is called.
    """

    if step_mm <= 0:
        raise ValueError("step_mm must be > 0")
    if end_mm <= start_mm:
        raise ValueError("end_mm must be > start_mm")

    positions = []
    magnitudes = []

    # Start sampling ONCE
    send_command(ser, CMD_START_SAMPLING)
    _ = read_adc_frame(ser)  # alignment discard

    x = start_mm
    while x <= end_mm + 1e-9:
        positions.append(x)

        # read frame
        samples = read_adc_frame(ser)
        if samples is None:
            send_command(ser, CMD_STOP_SAMPLING)
            raise RuntimeError("ADC error during scan")

        # FFT magnitude at f
        mag, f_bin = fft_mag_at_freq(samples, fs_hz, f_hz)
        magnitudes.append(mag)

        # move to next point
        x += step_mm
        if x > end_mm:
            break

        move_to_mm(ser, x)
        if not wait_until_move_complete(ser):
            send_command(ser, CMD_STOP_SAMPLING)
            raise RuntimeError("Move timeout")

    send_command(ser, CMD_STOP_SAMPLING)

    mag_arr = np.array(magnitudes)

    # coarse extrema
    idx_max = int(np.argmax(mag_arr))
    idx_min = int(np.argmin(mag_arr))

    p_max = float(mag_arr[idx_max])
    p_min = float(max(mag_arr[idx_min], 1e-9))

    SWR = p_max / p_min
    R_mag = (SWR - 1) / (SWR + 1)

    return positions, mag_arr, {
        "p_max": p_max,
        "p_min": p_min,
        "idx_max": idx_max,
        "idx_min": idx_min,
        "SWR": SWR,
        "R_mag": R_mag,
    }

def scan_kundt_two_stage(
    ser,
    f_hz,
    start_mm,
    end_mm,
    coarse_step_mm,
    fine_step_mm,
    fine_window_mm,
    fs_hz
):
    """
    Performs:
      1) Coarse scan across the whole tube
      2) Fine scan around coarse Pmax and Pmin

    Returns a dictionary with:
        coarse_positions, coarse_mag, coarse_res
        fine_max_positions, fine_max_mag, fine_max_res
        fine_min_positions, fine_min_mag, fine_min_res
    """

    # ----------------------------
    # 1. COARSE SCAN
    # ----------------------------
    coarse_positions, coarse_mag, coarse_res = scan_kundt(
        ser, f_hz, start_mm, end_mm, coarse_step_mm, fs_hz
    )

    pos_arr = np.array(coarse_positions, float)
    mag_arr = np.array(coarse_mag, float)

    # Find coarse maxima and minima
    idx_max_coarse = int(np.argmax(mag_arr))
    idx_min_coarse = int(np.argmin(mag_arr))

    x_max_coarse = pos_arr[idx_max_coarse]
    x_min_coarse = pos_arr[idx_min_coarse]

    # ----------------------------
    # 2. FINE SCAN AROUND Pmax
    # ----------------------------

    fine_max_start = max(start_mm, x_max_coarse - fine_window_mm)
    fine_max_end   = min(end_mm,   x_max_coarse + fine_window_mm)

    fine_max_positions, fine_max_mag, fine_max_res = scan_kundt_no_home(
        ser, f_hz, fine_max_start, fine_max_end, fine_step_mm, fs_hz
    )

    # ----------------------------
    # 3. FINE SCAN AROUND Pmin
    # ----------------------------

    fine_min_start = max(start_mm, x_min_coarse - fine_window_mm)
    fine_min_end   = min(end_mm,   x_min_coarse + fine_window_mm)

    fine_min_positions, fine_min_mag, fine_min_res = scan_kundt_no_home(
        ser, f_hz, fine_min_start, fine_min_end, fine_step_mm, fs_hz
    )

    return {
        "coarse_positions": coarse_positions,
        "coarse_mag": coarse_mag,
        "coarse_res": coarse_res,

        "fine_max_positions": fine_max_positions,
        "fine_max_mag": fine_max_mag,
        "fine_max_res": fine_max_res,

        "fine_min_positions": fine_min_positions,
        "fine_min_mag": fine_min_mag,
        "fine_min_res": fine_min_res,
    }