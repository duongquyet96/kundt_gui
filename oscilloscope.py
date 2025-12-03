import matplotlib.pyplot as plt
from adc_stream import read_adc_frame
from serial_comm import send_command
from commands import *

def run_oscilloscope(ser, trigger_level=500, trigger_mode="rising", fs=FS):
    print("Starting oscilloscope...")
    status, _ = send_command(ser, CMD_START_SAMPLING)

    if status != STS_ACK:
        print("Failed to start sampling")
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(9,4))
    line, = ax.plot([0]*SAMPLES_PER_FRAME)

    ax.set_ylim(0, 4095)
    ax.set_xlim(0, SAMPLES_PER_FRAME - 1)
    ax.grid(True)

    def triggered(x):
        for i in range(1, len(x)):
            prev, curr = x[i-1], x[i]
            if trigger_mode == "rising" and prev < trigger_level <= curr:
                return True
            if trigger_mode == "falling" and prev > trigger_level >= curr:
                return True
        return False

    first = True

    try:
        while True:
            samples = read_adc_frame(ser)
            if samples is None:
                continue

            if first:
                line.set_ydata(samples)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                first = False
                continue

            if not triggered(samples):
                continue

            line.set_ydata(samples)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        pass

    finally:
        send_command(ser, CMD_STOP_SAMPLING)
        plt.close(fig)
