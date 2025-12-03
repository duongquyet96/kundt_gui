import serial
import struct

from commands import *
from serial_comm import send_command
from adc_stream import read_adc_frame
from oscilloscope import run_oscilloscope
from fft_tools import plot_fft
from digipot import digipot_set
from scan_kundt import scan_standing_wave_pmax_pmin
from motor_control import wait_until_move_complete, wait_until_home_complete

PORT = "COM5"
BAUD = 115200


# -----------------------------
# Menu
# -----------------------------
def print_menu():
    print("\nMenu:")
    print("1: Toggle direction pin")
    print("2: Set direction pin (0=RESET, 1=SET)")
    print("3: Move stepper motor (mm)")
    print("4: Read end switch state")
    print("5: Home motor until end switch LOW")
    print("6: Set AD9833 sine frequency (Hz)")
    print("7: Oscilloscope Mode")
    print("8: Stop sampling")
    print("9: Show FFT of one ADC frame")
    print("10: Scan tube (pmax/pmin, |R|)")
    print("11: Set amplifier gain (0–99)")
    print("12: Exit")


# -----------------------------
# Main
# -----------------------------
def main():
    with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
        print(f"Connected to {PORT} at {BAUD} baud")

        while True:
            print_menu()
            choice = input("Select option: ").strip()

            # 1 ------------------------------------------
            if choice == '1':
                status, _ = send_command(ser, CMD_TOGGLE_DIR)
                print(f"Toggle response: {status}")

            # 2 ------------------------------------------
            elif choice == '2':
                val = input("0 or 1: ").strip()
                if val not in ('0', '1'):
                    print("Invalid")
                    continue
                status, _ = send_command(ser, CMD_SET_DIR, bytes([int(val)]))
                print(f"Set response: {status}")

            # 3 ------------------------------------------
            elif choice == '3':
                try:
                    dist = float(input("Enter distance in mm: ").strip())
                except ValueError:
                    print("Invalid number")
                    continue

                payload = struct.pack('<f', dist)
                status, _ = send_command(ser, CMD_STEPPER_MOVE, payload)
                print(f"Move response: {status}")

                if not wait_until_move_complete(ser):
                    print("Move timeout!")

            # 4 ------------------------------------------
            elif choice == '4':
                status, pl = send_command(ser, CMD_READ_SWITCH)
                if status == STS_ACK and pl:
                    print("End switch:", "HIGH" if pl[0] else "LOW")
                else:
                    print(f"Switch read failed: status={status}")

            # 5 ------------------------------------------
            elif choice == '5':
                status, _ = send_command(ser, CMD_HOME)
                print("Homing started…")
                if not wait_until_home_complete(ser):
                    print("Error: homing timeout.")
                else:
                    print("Home complete.")

            # 6 ------------------------------------------
            elif choice == '6':
                try:
                    f_hz = float(input("Frequency in Hz: ").strip())
                except ValueError:
                    print("Invalid number")
                    continue

                f_hz = max(0.0, min(f_hz, 12_000_000.0))
                payload = struct.pack('<f', f_hz)

                status, _ = send_command(ser, CMD_AD9833_SINE_FREQ, payload)
                print("AD9833:", "OK" if status == STS_ACK else "FAILED")

            # 7 ------------------------------------------
            elif choice == '7':
                run_oscilloscope(ser)

            # 8 ------------------------------------------
            elif choice == '8':
                status, _ = send_command(ser, CMD_STOP_SAMPLING)
                print(f"Stop sampling: status={status}")

            # 9 ------------------------------------------
            elif choice == '9':
                print("Reading frame...")
                send_command(ser, CMD_START_SAMPLING)
                samples = read_adc_frame(ser)
                send_command(ser, CMD_STOP_SAMPLING)

                if samples is None:
                    print("ADC error")
                    continue

                print("FFT:")
                plot_fft(samples, FS)

            # 10 ------------------------------------------
            elif choice == '10':
                scan_standing_wave_pmax_pmin(ser)

            # 11 ------------------------------------------
            elif choice == '11':
                try:
                    v = int(input("Gain (0–99): ").strip())
                    digipot_set(ser, v)
                except:
                    print("Invalid number")

            # 12 ------------------------------------------
            elif choice == '12':
                print("Goodbye!")
                break

            else:
                print("Invalid option.")


# -----------------------------
if __name__ == "__main__":
    main()
