import struct
import numpy as np
import matplotlib.pyplot as plt

from serial_comm import send_command
from adc_stream import read_adc_frame
from motor_control import *
from fft_tools import fft_bin_at_freq
from commands import *

def scan_standing_wave_pmax_pmin(ser):
    FS = 100000

    try:
        f_hz = float(input("Excitation frequency (Hz): "))
        start_mm = float(input("Start position (mm): "))
        end_mm = float(input("End position (mm): "))
        step_mm = float(input("Step size (mm): "))
    except:
        print("Invalid input")
        return

    print("Homing…")
    send_command(ser, CMD_HOME)
    if not wait_until_home_complete(ser):
        print("Homing timeout")
        return

    print(f"Moving to {start_mm} mm")
    send_command(ser, CMD_STEPPER_MOVE, struct.pack("<f", start_mm))
    wait_until_move_complete(ser)

    positions, magnitudes = [], []

    x = start_mm
    while x <= end_mm:
        print(f"At {x:.2f} mm")

        send_command(ser, CMD_START_SAMPLING)
        samples = read_adc_frame(ser)
        send_command(ser, CMD_STOP_SAMPLING)

        fft_val, f_bin = fft_bin_at_freq(samples, FS, f_hz)
        positions.append(x)
        magnitudes.append(abs(fft_val))

        x += step_mm
        send_command(ser, CMD_STEPPER_MOVE, struct.pack("<f", x))
        wait_until_move_complete(ser)

    mag = np.array(magnitudes)
    idx_max = np.argmax(mag)
    idx_min = np.argmin(mag)

    SWR = mag[idx_max] / mag[idx_min]
    R = (SWR - 1) / (SWR + 1)

    print("max:", mag[idx_max])
    print("min:", mag[idx_min])
    print("SWR:", SWR)
    print("|R|:", R)

    plt.plot(positions, mag)
    plt.grid(True)
    plt.show()
