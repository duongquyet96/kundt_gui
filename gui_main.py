# gui_main.py
import sys
import struct
import numpy as np
import serial

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QGroupBox, QGridLayout, QMessageBox, QTabWidget, QFormLayout
)
from PyQt5.QtCore import Qt

from commands import *               # FS, command IDs
from serial_comm import send_command
from adc_stream import read_adc_frame
from motor_control import wait_until_move_complete, wait_until_home_complete
from digipot import digipot_set
from scan_kundt import scan_kundt    # <-- use shared backend scan


class SerialManager:
    def __init__(self):
        self.ser = None

    def connect(self, port, baud):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = serial.Serial(port, baud, timeout=0.1)

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None

    def is_connected(self):
        return self.ser is not None and self.ser.is_open


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Kundt Tube Controller GUI")
        self.serial_mgr = SerialManager()
        self.last_samples = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        # Connection
        conn_group = QGroupBox("Connection")
        conn_layout = QHBoxLayout()
        self.port_edit = QLineEdit("COM5")
        self.baud_edit = QLineEdit("115200")
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")

        self.connect_btn.clicked.connect(self.on_connect)
        self.disconnect_btn.clicked.connect(self.on_disconnect)

        conn_layout.addWidget(QLabel("Port:"))
        conn_layout.addWidget(self.port_edit)
        conn_layout.addWidget(QLabel("Baud:"))
        conn_layout.addWidget(self.baud_edit)
        conn_layout.addWidget(self.connect_btn)
        conn_layout.addWidget(self.disconnect_btn)
        conn_group.setLayout(conn_layout)
        main_layout.addWidget(conn_group)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_motor_tab(), "Position")
        tabs.addTab(self._build_signal_tab(), "Signal / ADC / FFT")
        tabs.addTab(self._build_kundt_tab(), "Kundt Scan")
        tabs.addTab(self._build_gain_tab(), "Gain (Digipot)")
        main_layout.addWidget(tabs)

        self.setCentralWidget(central)

    # ---------------- Motor tab ----------------
    def _build_motor_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        grid = QGridLayout()

        
        self.move_spin = QDoubleSpinBox()
        self.move_spin.setRange(-1000.0, 1000.0)
        self.move_spin.setDecimals(3)
        self.move_spin.setValue(10.0)
        btn_move = QPushButton("Move (mm)")
        btn_move.clicked.connect(self.on_move_mm)
        grid.addWidget(QLabel("Distance (mm):"), 1, 0)
        grid.addWidget(self.move_spin, 1, 1)
        grid.addWidget(btn_move, 1, 2)

    

        btn_home = QPushButton("Home Motor")
        btn_home.clicked.connect(self.on_home)
        grid.addWidget(btn_home, 3, 0)

        layout.addLayout(grid)
        layout.addStretch()
        return w

    # ---------------- Signal / ADC / FFT tab ----------------
    def _build_signal_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        freq_group = QGroupBox("Signal Generator (AD9833)")
        fg_layout = QHBoxLayout()
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.0, 12_000_000.0)
        self.freq_spin.setDecimals(2)
        self.freq_spin.setValue(1000.0)

        btn_set_freq = QPushButton("Set Frequency (Hz)")
        btn_set_freq.clicked.connect(self.on_set_freq)

        btn_mute = QPushButton("Mute")
        btn_mute.clicked.connect(self.on_mute)

        fg_layout.addWidget(QLabel("Frequency (Hz):"))
        fg_layout.addWidget(self.freq_spin)
        fg_layout.addWidget(btn_set_freq)
        fg_layout.addWidget(btn_mute)
        
        freq_group.setLayout(fg_layout)
        layout.addWidget(freq_group)

        adc_group = QGroupBox("ADC / Capture")
        adc_layout = QHBoxLayout()
        btn_capture = QPushButton("Capture Frame")
        btn_capture.clicked.connect(self.on_capture_frame)
        adc_layout.addWidget(btn_capture)
        adc_group.setLayout(adc_layout)
        layout.addWidget(adc_group)

        self.time_canvas = MplCanvas(self, width=5, height=3)
        self.fft_canvas = MplCanvas(self, width=5, height=3)

        layout.addWidget(QLabel("Time-domain signal"))
        layout.addWidget(self.time_canvas)
        layout.addWidget(QLabel("FFT magnitude"))
        layout.addWidget(self.fft_canvas)
        layout.addStretch()
        return w

    # ---------------- Kundt scan tab ----------------
    def _build_kundt_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        form = QFormLayout()
        self.scan_freq_spin = QDoubleSpinBox()
        self.scan_freq_spin.setRange(1.0, 20_000.0)
        self.scan_freq_spin.setValue(1000.0)

        self.scan_start_spin = QDoubleSpinBox()
        self.scan_start_spin.setRange(0.0, 500.0)
        self.scan_start_spin.setValue(5.0)

        self.scan_end_spin = QDoubleSpinBox()
        self.scan_end_spin.setRange(0.0, 500.0)
        self.scan_end_spin.setValue(250.0)

        self.scan_step_spin = QDoubleSpinBox()
        self.scan_step_spin.setRange(0.1, 50.0)
        self.scan_step_spin.setValue(5.0)

        form.addRow("Frequency (Hz):", self.scan_freq_spin)
        form.addRow("Start (mm):", self.scan_start_spin)
        form.addRow("End (mm):", self.scan_end_spin)
        form.addRow("Step (mm):", self.scan_step_spin)

        layout.addLayout(form)

        btn_run_scan = QPushButton("Run Scan")
        btn_run_scan.clicked.connect(self.on_run_scan)
        layout.addWidget(btn_run_scan)

        self.kundt_canvas = MplCanvas(self, width=5, height=3)
        layout.addWidget(self.kundt_canvas)

        self.scan_result_label = QLabel("Results: -")
        layout.addWidget(self.scan_result_label)

        layout.addStretch()
        return w

    # ---------------- Gain tab ----------------
    def _build_gain_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(0, 99)
        self.gain_spin.setValue(50)
        btn_set_gain = QPushButton("Set Gain (Digipot)")
        btn_set_gain.clicked.connect(self.on_set_gain)

        layout.addWidget(QLabel("Gain setting (0–99):"))
        layout.addWidget(self.gain_spin)
        layout.addWidget(btn_set_gain)
        layout.addStretch()
        return w

    # ---------------- Helpers ----------------
    def ensure_connected(self):
        if not self.serial_mgr.is_connected():
            QMessageBox.warning(self, "Not connected", "Please connect to the serial port first.")
            return False
        return True

    # ---------------- Slots ----------------
    def on_connect(self):
        port = self.port_edit.text().strip()
        try:
            baud = int(self.baud_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid baud rate.")
            return
        try:
            self.serial_mgr.connect(port, baud)
            QMessageBox.information(self, "Connected", f"Connected to {port} at {baud} baud.")
        except Exception as e:
            QMessageBox.critical(self, "Connection error", str(e))

    def on_disconnect(self):
        self.serial_mgr.disconnect()
        QMessageBox.information(self, "Disconnected", "Serial port closed.")

    def on_toggle_dir(self):
        if not self.ensure_connected(): return
        status, _ = send_command(self.serial_mgr.ser, CMD_TOGGLE_DIR)
        QMessageBox.information(self, "Toggle DIR", f"Status: {status}")

    def on_set_dir(self, val):
        if not self.ensure_connected(): return
        status, _ = send_command(self.serial_mgr.ser, CMD_SET_DIR, bytes([val]))
        QMessageBox.information(self, "Set DIR", f"Status: {status}")

    def on_move_mm(self):
        if not self.ensure_connected(): return
        dist = float(self.move_spin.value())
        payload = struct.pack("<f", dist)
        status, _ = send_command(self.serial_mgr.ser, CMD_STEPPER_MOVE, payload)
        if status is None:
            QMessageBox.warning(self, "Move", "No response from MCU.")
            return
        if not wait_until_move_complete(self.serial_mgr.ser):
            QMessageBox.warning(self, "Move", "Move timeout!")
        else:
            QMessageBox.information(self, "Move", "Move completed.")

    def on_read_switch(self):
        if not self.ensure_connected(): return
        status, pl = send_command(self.serial_mgr.ser, CMD_READ_SWITCH)
        if status == STS_ACK and pl:
            self.switch_label.setText("Switch: HIGH" if pl[0] else "Switch: LOW")
        else:
            self.switch_label.setText("Switch: ERROR")

    def on_home(self):
        if not self.ensure_connected(): return
        status, _ = send_command(self.serial_mgr.ser, CMD_HOME)
        if status != STS_ACK:
            QMessageBox.warning(self, "Home", "Home command failed to start.")
            return
        if not wait_until_home_complete(self.serial_mgr.ser):
            QMessageBox.warning(self, "Home", "Homing timeout.")
        else:
            QMessageBox.information(self, "Home", "Homing complete.")

    def on_set_freq(self):
        if not self.ensure_connected(): return
        f_hz = float(self.freq_spin.value())
        f_hz = max(0.0, min(f_hz, 12_000_000.0))
        payload = struct.pack("<f", f_hz)
        status, _ = send_command(self.serial_mgr.ser, CMD_AD9833_SINE_FREQ, payload)
        QMessageBox.information(
            self, "AD9833",
            "Frequency set OK." if status == STS_ACK else f"Set failed (status {status})"
        )
    def on_mute(self):
        if not self.ensure_connected():
            return

        # Send 0 Hz to AD9833
        payload = struct.pack("<f", 0.0)
        status, _ = send_command(self.serial_mgr.ser, CMD_AD9833_SINE_FREQ, payload)

        if status == STS_ACK:
            QMessageBox.information(self, "Speaker Mute", "Speaker muted (frequency = 0 Hz).")
        else:
            QMessageBox.warning(self, "Speaker Mute", f"Failed to mute. Status={status}")

    def on_capture_frame(self):
        if not self.ensure_connected(): return
        send_command(self.serial_mgr.ser, CMD_START_SAMPLING)
        samples = read_adc_frame(self.serial_mgr.ser)
        send_command(self.serial_mgr.ser, CMD_STOP_SAMPLING)

        if samples is None:
            QMessageBox.warning(self, "ADC", "Failed to read ADC frame.")
            return

        self.last_samples = samples
        self._update_time_plot(samples)
        self._update_fft_plot(samples)

    def _update_time_plot(self, samples):
        ax = self.time_canvas.ax
        ax.clear()
        ax.plot(samples)
        ax.set_title("ADC Time Signal")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("ADC value")
        ax.grid(True)
        self.time_canvas.draw()

    def _update_fft_plot(self, samples):
        ax = self.fft_canvas.ax
        ax.clear()
        x = samples.astype(float)
        x -= np.mean(x)
        x *= np.hanning(len(x))
        N = len(x)
        fft_vals = np.fft.rfft(x)
        fft_mag = np.abs(fft_vals) * 2 / N
        freqs = np.fft.rfftfreq(N, 1.0 / FS)
        ax.plot(freqs, fft_mag)
        ax.set_xlim(0, FS/2)
        ax.set_title("FFT magnitude")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True)
        self.fft_canvas.draw()

    def on_set_gain(self):
        if not self.ensure_connected(): return
        v = int(self.gain_spin.value())
        digipot_set(self.serial_mgr.ser, v)

    def on_run_scan(self):
        if not self.ensure_connected(): return

        f_hz = float(self.scan_freq_spin.value())
        start_mm = float(self.scan_start_spin.value())
        end_mm = float(self.scan_end_spin.value())
        step_mm = float(self.scan_step_spin.value())

        try:
            positions, mag_arr, res = scan_kundt(
                self.serial_mgr.ser, f_hz, start_mm, end_mm, step_mm
            )
        except Exception as e:
            QMessageBox.warning(self, "Scan error", str(e))
            return

        p_max = res["p_max"]
        p_min = res["p_min"]
        idx_max = res["idx_max"]
        idx_min = res["idx_min"]
        SWR = res["SWR"]
        R_mag = res["R_mag"]

        self.scan_result_label.setText(
            f"p_max={p_max:.3f} at {positions[idx_max]:.2f} mm, "
            f"p_min={p_min:.3f} at {positions[idx_min]:.2f} mm, "
            f"SWR={SWR:.3f}, |R|={R_mag:.3f}"
        )

        # Plot standing wave
        ax = self.kundt_canvas.ax
        ax.clear()
        ax.plot(positions, mag_arr, label="|P| raw")

        # optional smoothing
        if len(mag_arr) >= 5:
            mag_s = np.convolve(mag_arr, np.ones(5)/5, mode='same')
            ax.plot(positions, mag_s, '--', label="|P| smoothed")

        ax.scatter([positions[idx_max]], [p_max], label="p_max")
        ax.scatter([positions[idx_min]], [p_min], label="p_min")

        ax.set_title("Standing Wave |P| vs Position")
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Amplitude |P|")
        ax.grid(True)
        ax.legend()
        self.kundt_canvas.draw()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1080, 800)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
