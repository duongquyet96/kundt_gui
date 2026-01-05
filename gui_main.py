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
    QGroupBox, QGridLayout, QMessageBox, QTabWidget, QFormLayout, QButtonGroup
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

from commands import *               # FS, command IDs
from serial_comm import send_command
from adc_stream import read_adc_frame
from motor_control import wait_until_move_complete, wait_until_home_complete
from digipot import digipot_set
from scan_kundt import scan_kundt_two_stage, fft_mag_at_freq


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
        self.live_mode = False
        self.live_timer = None

        self.FS = 20000.0      # default sampling rate
        self.FFT_N = 4096       # default FFT length

        # Zoom parameters
        self.adc_zoom_factor = 15.0
        self.adc_zoom_center = 0
        # Y-axis zoom parameters
        self.y_zoom_factor = 5.0
        self.y_center = 1.65   # midpoint of 0–3.3 V

        self.cont_scan_active = False
        self.cont_scan_timer = None
        self.cont_scan_positions = []
        self.cont_scan_mags = []
        self.cont_scan_freq = 1000.0  # will be set from UI
        self.cont_scan_end_mm = 250.0



        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        # Connection
        conn_group = QGroupBox("Connection")
        conn_layout = QHBoxLayout()
        self.port_edit = QLineEdit("COM5")

        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")

        self.connect_btn.clicked.connect(self.on_connect)
        self.disconnect_btn.clicked.connect(self.on_disconnect)

        conn_layout.addWidget(QLabel("Port:"))
        conn_layout.addWidget(self.port_edit)

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

        # --- Home offset controls ---
        self.home_offset_spin = QDoubleSpinBox()
        self.home_offset_spin.setRange(-1000.0, 1000.0)
        self.home_offset_spin.setDecimals(3)
        self.home_offset_spin.setValue(2.9)

        btn_set_home_offset = QPushButton("Set Home Offset (mm)")
        btn_set_home_offset.clicked.connect(self.on_set_home_offset)

        grid.addWidget(QLabel("Home offset (mm):"), 3, 0)
        grid.addWidget(self.home_offset_spin, 3, 1)
        grid.addWidget(btn_set_home_offset, 3, 2)

        btn_home = QPushButton("Home Motor")
        btn_home.clicked.connect(self.on_home)
        grid.addWidget(btn_home, 4, 0)

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

        btn_set_freq = QPushButton("Play Tone")
        btn_set_freq.clicked.connect(self.on_set_freq)

        btn_mute = QPushButton("Mute")
        btn_mute.clicked.connect(self.on_mute)

        fg_layout.addWidget(QLabel("Frequency (Hz):"))
        fg_layout.addWidget(self.freq_spin)
        fg_layout.addWidget(btn_set_freq)
        fg_layout.addWidget(btn_mute)
        
        freq_group.setLayout(fg_layout)
        layout.addWidget(freq_group)

        # ---------------- FFT Settings -----------------
        fft_group = QGroupBox("FFT Settings")
        fft_layout = QHBoxLayout()

        # Sampling frequency input (FS)
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(100.0, 500000.0)
        self.fs_spin.setValue(20000.0)  # default FS = 100 kHz
        self.fs_spin.setDecimals(1)

        # FFT size (N)
        self.fft_n_spin = QSpinBox()
        self.fft_n_spin.setRange(256, 16384)
        self.fft_n_spin.setSingleStep(256)
        self.fft_n_spin.setValue(4096)

        # Apply button
        btn_apply_fft = QPushButton("Apply FFT Settings")
        btn_apply_fft.clicked.connect(self.on_apply_fft_settings)

        fft_layout.addWidget(QLabel("Sampling FS (Hz):"))
        fft_layout.addWidget(self.fs_spin)
        fft_layout.addWidget(QLabel("FFT size N:"))
        fft_layout.addWidget(self.fft_n_spin)
        fft_layout.addWidget(btn_apply_fft)

        fft_group.setLayout(fft_layout)
        layout.addWidget(fft_group)

        adc_group = QGroupBox("ADC / Capture")
        adc_layout = QHBoxLayout()
        #btn_capture = QPushButton("Capture Frame")
        #btn_capture.clicked.connect(self.on_capture_frame)
        #adc_layout.addWidget(btn_capture)
        adc_group.setLayout(adc_layout)
        layout.addWidget(adc_group)

        btn_live_start = QPushButton("Start Live View")
        btn_live_start.clicked.connect(self.on_start_live)

        btn_live_stop = QPushButton("Stop Live View")
        btn_live_stop.clicked.connect(self.on_stop_live)

        adc_layout.addWidget(btn_live_start)
        adc_layout.addWidget(btn_live_stop)


        # ---------- Zoom Controls (Right Side) ----------
        zoom_panel = QWidget()
        zoom_layout = QVBoxLayout(zoom_panel)
        zoom_layout.setContentsMargins(0, 0, 0, 0)

        btn_zoom_in = QPushButton("X+")
        btn_zoom_out = QPushButton("X–")
        btn_zoom_reset = QPushButton("XR")

        btn_y_zoom_in = QPushButton("Y+")
        btn_y_zoom_out = QPushButton("Y–")
        btn_y_reset = QPushButton("YR")

        # Make buttons small
        for b in (btn_zoom_in, btn_zoom_out, btn_zoom_reset):
            b.setFixedSize(40, 30)    # small oscilloscope-style buttons
            b.setStyleSheet("font-size: 16px; padding: 2px;")

        btn_zoom_in.clicked.connect(self.on_zoom_in)
        btn_zoom_out.clicked.connect(self.on_zoom_out)
        btn_zoom_reset.clicked.connect(self.on_zoom_reset)

        zoom_layout.addWidget(btn_zoom_in)
        zoom_layout.addWidget(btn_zoom_out)
        zoom_layout.addWidget(btn_zoom_reset)
        zoom_layout.addStretch()

        
        for b in (btn_y_zoom_in, btn_y_zoom_out, btn_y_reset):
            b.setFixedSize(40, 30)
            b.setStyleSheet("font-size: 16px; padding: 2px;")

        btn_y_zoom_in.clicked.connect(self.on_y_zoom_in)
        btn_y_zoom_out.clicked.connect(self.on_y_zoom_out)
        btn_y_reset.clicked.connect(self.on_y_zoom_reset)

        zoom_layout.addWidget(btn_y_zoom_in)
        zoom_layout.addWidget(btn_y_zoom_out)
        zoom_layout.addWidget(btn_y_reset)

        self.time_canvas = MplCanvas(self, width=5, height=3)
        self.fft_canvas = MplCanvas(self, width=5, height=3)

        # Wrap ADC graph + zoom panel
        adc_container = QWidget()
        adc_hbox = QHBoxLayout(adc_container)
        adc_hbox.setContentsMargins(0, 0, 0, 0)
        adc_hbox.setSpacing(5)

        # The ADC plot should expand as much as possible
        adc_hbox.addWidget(self.time_canvas, stretch=1)

        # Wrap zoom panel inside a right-aligned container
        zoom_panel_container = QWidget()
        zoom_panel_layout = QVBoxLayout(zoom_panel_container)
        zoom_panel_layout.setContentsMargins(0, 0, 0, 0)
        zoom_panel_layout.addWidget(zoom_panel)
        zoom_panel_layout.addStretch()

        adc_hbox.addWidget(zoom_panel_container)
        adc_hbox.setStretchFactor(self.time_canvas, 1)
        adc_hbox.setStretchFactor(zoom_panel_container, 0)

        layout.addWidget(QLabel("Time-domain signal"))
        layout.addWidget(adc_container)


        layout.addWidget(QLabel("FFT magnitude"))
        layout.addWidget(self.fft_canvas)
        layout.addStretch()
        return w

    # ---------------- Kundt scan tab ----------------
    def _build_kundt_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # ---------------- Scan mode buttons (top) ----------------
        mode_row = QWidget()
        mode_row_layout = QHBoxLayout(mode_row)
        mode_row_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_mode_live = QPushButton("Live Scan")
        self.btn_mode_step = QPushButton("Step Scan")
        self.btn_mode_live.setCheckable(True)
        self.btn_mode_step.setCheckable(True)

        self.scan_mode_group = QButtonGroup(self)
        self.scan_mode_group.setExclusive(True)
        self.scan_mode_group.addButton(self.btn_mode_live, 0)  # 0 = Live
        self.scan_mode_group.addButton(self.btn_mode_step, 1)  # 1 = Step
        self.btn_mode_step.setChecked(True)  # default

        mode_row_layout.addWidget(self.btn_mode_live)
        mode_row_layout.addWidget(self.btn_mode_step)
        mode_row_layout.addStretch()
        layout.addWidget(mode_row)

        # ---------------- Compact widget helper ----------------
        def compact_spin(spin):
            spin.setFixedHeight(28)
            spin.setMinimumWidth(150)
            spin.setStyleSheet("padding: 2px;")
            return spin

        # ---------------- Two-column parameter row ----------------
        params_row = QHBoxLayout()
        params_row.setContentsMargins(0, 0, 0, 0)
        params_row.setSpacing(12)

        # ---- Left: Scan Range (always visible) ----
        basic_group = QGroupBox("Scan Range")
        basic_grid = QGridLayout(basic_group)
        basic_grid.setContentsMargins(8, 8, 8, 8)
        basic_grid.setHorizontalSpacing(10)
        basic_grid.setVerticalSpacing(6)

        self.scan_freq_spin = compact_spin(QDoubleSpinBox())
        self.scan_freq_spin.setRange(1.0, 20_000.0)
        self.scan_freq_spin.setValue(1000.0)
        self.scan_freq_spin.setDecimals(2)

        self.scan_start_spin = compact_spin(QDoubleSpinBox())
        self.scan_start_spin.setRange(0.0, 500.0)
        self.scan_start_spin.setValue(5.0)
        self.scan_start_spin.setDecimals(2)

        self.scan_end_spin = compact_spin(QDoubleSpinBox())
        self.scan_end_spin.setRange(0.0, 500.0)
        self.scan_end_spin.setValue(250.0)
        self.scan_end_spin.setDecimals(2)

        basic_grid.addWidget(QLabel("Frequency (Hz):"), 0, 0)
        basic_grid.addWidget(self.scan_freq_spin,       0, 1)
        basic_grid.addWidget(QLabel("Start (mm):"),     1, 0)
        basic_grid.addWidget(self.scan_start_spin,      1, 1)
        basic_grid.addWidget(QLabel("End (mm):"),       2, 0)
        basic_grid.addWidget(self.scan_end_spin,        2, 1)

        params_row.addWidget(basic_group, 1)

        # ---- Right: container that holds Step OR Live group ----
        right_container = QWidget()
        right_v = QVBoxLayout(right_container)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(0)

        # ---- Step Scan parameters (Step mode only) ----
        step_group = QGroupBox("Step Scan Parameters")
        step_grid = QGridLayout(step_group)
        step_grid.setContentsMargins(8, 8, 8, 8)
        step_grid.setHorizontalSpacing(10)
        step_grid.setVerticalSpacing(6)

        self.coarse_step_spin = compact_spin(QDoubleSpinBox())
        self.coarse_step_spin.setRange(0.1, 100.0)
        self.coarse_step_spin.setValue(5.0)
        self.coarse_step_spin.setDecimals(2)

        self.fine_step_spin = compact_spin(QDoubleSpinBox())
        self.fine_step_spin.setRange(0.01, 10.0)
        self.fine_step_spin.setValue(1.0)
        self.fine_step_spin.setDecimals(3)

        self.fine_window_spin = compact_spin(QDoubleSpinBox())
        self.fine_window_spin.setRange(1.0, 100.0)
        self.fine_window_spin.setValue(10.0)
        self.fine_window_spin.setDecimals(2)

        self.lbl_coarse   = QLabel("Coarse step (mm):")
        self.lbl_finestep = QLabel("Fine step (mm):")
        self.lbl_finewin  = QLabel("Fine window (mm):")

        step_grid.addWidget(self.lbl_coarse,        0, 0)
        step_grid.addWidget(self.coarse_step_spin,  0, 1)
        step_grid.addWidget(self.lbl_finestep,      1, 0)
        step_grid.addWidget(self.fine_step_spin,    1, 1)
        step_grid.addWidget(self.lbl_finewin,       2, 0)
        step_grid.addWidget(self.fine_window_spin,  2, 1)

        # ---- Live Scan acquisition (Live mode only) ----
        live_group = QGroupBox("Live Scan Acquisition")
        live_grid = QGridLayout(live_group)
        live_grid.setContentsMargins(8, 8, 8, 8)
        live_grid.setHorizontalSpacing(10)
        live_grid.setVerticalSpacing(6)

        self.live_fs_spin = compact_spin(QDoubleSpinBox())
        self.live_fs_spin.setRange(100.0, 500_000.0)
        self.live_fs_spin.setDecimals(1)
        self.live_fs_spin.setValue(float(getattr(self, "FS", 20000.0)))

        self.live_n_spin = compact_spin(QSpinBox())
        self.live_n_spin.setRange(256, 16384)
        self.live_n_spin.setSingleStep(256)
        self.live_n_spin.setValue(int(getattr(self, "FFT_N", 4096)))

        live_grid.addWidget(QLabel("Sampling rate FS (Hz):"), 0, 0)
        live_grid.addWidget(self.live_fs_spin,               0, 1)
        live_grid.addWidget(QLabel("Sample size N:"),        1, 0)
        live_grid.addWidget(self.live_n_spin,                1, 1)

        right_v.addWidget(step_group)
        right_v.addWidget(live_group)

        params_row.addWidget(right_container, 1)
        layout.addLayout(params_row)

        # ---------------- Action buttons ----------------
        self.btn_run_scan = QPushButton("Run Scan")
        self.btn_run_scan.clicked.connect(self.on_run_scan)
        layout.addWidget(self.btn_run_scan)

        self.btn_cont_start = QPushButton("Start Continuous Scan")
        self.btn_cont_start.clicked.connect(self.on_start_cont_scan)
        layout.addWidget(self.btn_cont_start)

        self.btn_cont_stop = QPushButton("Stop Continuous Scan")
        self.btn_cont_stop.clicked.connect(self.on_stop_cont_scan)
        layout.addWidget(self.btn_cont_stop)

        # ---------------- Plot + results ----------------
        self.kundt_canvas = MplCanvas(self, width=5, height=3)
        layout.addWidget(self.kundt_canvas)

        self.scan_result_label = QLabel("Results: -")
        layout.addWidget(self.scan_result_label)

        layout.addStretch()

        # ---------------- Mode-dependent visibility ----------------
        self.step_scan_only_widgets = [step_group, self.btn_run_scan]
        self.live_scan_only_widgets = [live_group, self.btn_cont_start, self.btn_cont_stop]

        self.scan_mode_group.buttonClicked[int].connect(self.on_scan_mode_changed)
        self.on_scan_mode_changed(1)  # default Step mode

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
            self.serial_mgr.connect(port, 115200)
            QMessageBox.information(self, "Connected", f"Connected to {port}.")
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
        send_command(self.serial_mgr.ser, CMD_SET_DIR, bytes([1]))
        status, _ = send_command(self.serial_mgr.ser, CMD_HOME)
        if status != STS_ACK:
            QMessageBox.warning(self, "Home", "Home command failed to start.")
            return
        if not wait_until_home_complete(self.serial_mgr.ser):
            QMessageBox.warning(self, "Home", "Homing timeout.")
        else:
            QMessageBox.information(self, "Home", "Homing complete.")
    def set_home_offset_mm(self, offset_mm: float):
        payload = struct.pack("<f", float(offset_mm))
        status, _ = send_command(self.serial_mgr.ser, CMD_SET_HOME_OFFSET, payload)
        if status != STS_ACK:
            raise RuntimeError("SET_HOME_OFFSET failed")

    def get_home_offset_mm(self) -> float:
        status, payload = send_command(self.serial_mgr.ser, CMD_GET_HOME_OFFSET)
        if status != STS_ACK or payload is None or len(payload) != 4:
            raise RuntimeError("GET_HOME_OFFSET failed")
        return struct.unpack("<f", payload)[0]
    
    def on_set_home_offset(self):
        if not self.ensure_connected():
            return

        try:
            offset = float(self.home_offset_spin.value())
            self.set_home_offset_mm(offset)
            QMessageBox.information(
                self,
                "Home Offset",
                f"Home offset set to {offset:.3f} mm"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


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
    def on_apply_fft_settings(self):
        self.FS = float(self.fs_spin.value())
        self.FFT_N = int(self.fft_n_spin.value())
        QMessageBox.information(self, "FFT Settings",
                                f"Sampling FS set to {self.FS} Hz\nFFT size N = {self.FFT_N}")

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
    
    
    def on_live_update(self):
        if not self.live_mode:
            return

        try:
            samples = read_adc_frame(self.serial_mgr.ser)
        except Exception:
            return

        if samples is None:
            return

        # Update waveform
        self.last_samples = samples
        self._update_time_plot(samples)

        # Update FFT live
        self._update_fft_plot(samples)

    def on_start_live(self):
        if not self.ensure_connected():
            return

        # Start streaming on STM32
        send_command(self.serial_mgr.ser, CMD_START_SAMPLING)

        # QTimer to update graph every 30 ms
        if self.live_timer is None:
            self.live_timer = QTimer()
            self.live_timer.timeout.connect(self.on_live_update)

        self.live_mode = True
        self.live_timer.start(30)   # ~33 FPS oscilloscope


    def on_stop_live(self):
        if self.live_timer:
            self.live_timer.stop()

        self.live_mode = False

        # Ask STM32 to stop streaming
        send_command(self.serial_mgr.ser, CMD_STOP_SAMPLING)


    def on_zoom_in(self):
        self.adc_zoom_factor *= 1.5
        if self.adc_zoom_factor > 100:
            self.adc_zoom_factor = 100

        if self.last_samples is not None:
            self.adc_zoom_center = len(self.last_samples) // 2
            self._update_time_plot(self.last_samples)


    def on_zoom_out(self):
        self.adc_zoom_factor /= 1.5
        if self.adc_zoom_factor < 1.0:
            self.adc_zoom_factor = 1.0

        if self.last_samples is not None:
            self._update_time_plot(self.last_samples)


    def on_zoom_reset(self):
        self.adc_zoom_factor = 1.0
        self.adc_zoom_center = 0

        if self.last_samples is not None:
            self._update_time_plot(self.last_samples)

    def on_y_zoom_in(self):
        self.y_zoom_factor *= 1.5
        if self.y_zoom_factor > 50:
            self.y_zoom_factor = 50
        if self.last_samples is not None:
            self._update_time_plot(self.last_samples)

    def on_y_zoom_out(self):
        self.y_zoom_factor /= 1.5
        if self.y_zoom_factor < 1.0:
            self.y_zoom_factor = 1.0
        if self.last_samples is not None:
            self._update_time_plot(self.last_samples)

    def on_y_zoom_reset(self):
        self.y_zoom_factor = 1.0
        if self.last_samples is not None:
            self._update_time_plot(self.last_samples)

    def on_scan_mode_changed(self, mode_id: int):
        """
        mode_id: 0 = Live Scan, 1 = Step Scan
        """
        is_live = (mode_id == 0)

        for w in self.step_scan_only_widgets:
            w.setVisible(not is_live)   # show only in Step Scan

        for w in self.live_scan_only_widgets:
            w.setVisible(is_live)       # show only in Live Scan


    def _update_time_plot(self, samples):
        ax = self.time_canvas.ax
        ax.clear()

        # --- Convert ADC → Volts ---
        ADC_VREF = 3.3
        ADC_MAX = 4096.0
        volts = samples * (ADC_VREF / ADC_MAX)

        N = len(volts)

        # ---- Determine visible range based on zoom ----
        if self.adc_zoom_factor <= 1.0:
            idx0, idx1 = 0, N
        else:
            window = int(N / self.adc_zoom_factor)
            half = window // 2
            c = self.adc_zoom_center
            idx0 = max(0, c - half)
            idx1 = min(N, idx0 + window)

        # ---- Plot ----
        ax.plot(range(idx0, idx1), volts[idx0:idx1])

        ax.set_title("ADC Time Signal")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True)

        # ---- FIXED Y-AXIS 0–3.3 V ----
        ax.set_ylim(0.0, 3.3)
        # ---- FIXED OR ZOOMED Y-AXIS ----
        if self.y_zoom_factor <= 1.0:
            # Default 0–3.3 V
            ax.set_ylim(0.0, 3.3)

        else:
            # Full 3.3 V amplitude divided by zoom factor
            half_span = (3.3 / 2) / self.y_zoom_factor

            # Zoom window centered at 1.65 V
            y0 = 1.65 - half_span
            y1 = 1.65 + half_span

            # Clamp to valid voltage range
            y0 = max(0.0, y0)
            y1 = min(3.3, y1)

            ax.set_ylim(y0, y1)

        self.time_canvas.draw()


    def _update_fft_plot(self, samples):
        ax = self.fft_canvas.ax
        ax.clear()

        # Get user-selected FFT size N
        N = self.FFT_N

        # Clip or zero-pad sample array
        if len(samples) > N:
            x = samples[:N].astype(float)
        else:
            x = np.zeros(N, dtype=float)
            x[:len(samples)] = samples.astype(float)

        # Remove DC and apply window
        x -= np.mean(x)
        x *= np.hanning(N)

        # Compute FFT
        fft_vals = np.fft.rfft(x)
        fft_mag = np.abs(fft_vals) * 2.0 / N

        # Frequency axis using user-selected sampling FS
        freqs = np.fft.rfftfreq(N, 1.0 / self.FS)

        ax.plot(freqs, fft_mag)

        # Limit x-axis to 0–Nyquist
        ax.set_xlim(0, freqs[-1])

        # ----- Add ticks every 500 Hz -----
        max_freq = freqs[-1]
        tick_step = 500
        ax.set_xticks(np.arange(0, max_freq + tick_step, tick_step))

        ax.set_title("FFT magnitude")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True)

        self.fft_canvas.draw()


    def on_set_gain(self):
        if not self.ensure_connected(): return
        v = int(self.gain_spin.value())
        digipot_set(self.serial_mgr.ser, v)

    def _mute_speaker(self):
        payload = struct.pack("<f", 0.0)
        send_command(self.serial_mgr.ser, CMD_AD9833_SINE_FREQ, payload)

    def get_position_mm(self):
        status, payload = send_command(self.serial_mgr.ser, CMD_GET_POSITION)
        if status != STS_ACK or not payload or len(payload) < 4:
            return None
        return struct.unpack("<f", payload[:4])[0]

    def find_peaks_and_valleys(self, positions, mags, min_prominence=0.05):
        """
        Very simple local-extrema finder.
        min_prominence is fraction of global max to ignore tiny ripples.
        Returns (peak_indices, valley_indices).
        """
        if len(mags) < 3:
            return [], []

        mags_arr = np.array(mags, dtype=float)
        max_mag = np.max(mags_arr)
        if max_mag <= 0:
            return [], []

        peak_idx = []
        valley_idx = []

        for i in range(1, len(mags_arr) - 1):
            left = mags_arr[i - 1]
            mid = mags_arr[i]
            right = mags_arr[i + 1]

            # relative prominence
            if mid > left and mid > right and mid > min_prominence * max_mag:
                peak_idx.append(i)
            if mid < left and mid < right and mid < (1.0 - min_prominence) * max_mag:
                valley_idx.append(i)

        return peak_idx, valley_idx

    def _update_kundt_live_plot(self):
        if not self.cont_scan_positions:
            return

        ax = self.kundt_canvas.ax
        ax.clear()

        pos = np.array(self.cont_scan_positions, dtype=float)
        mag = np.array(self.cont_scan_mags, dtype=float)

        ax.plot(pos, mag, label="|P|(live)")

        # auto-detect multiple maxima / minima
        peak_idx, valley_idx = self.find_peaks_and_valleys(pos, mag, min_prominence=0.1)

        if peak_idx:
            ax.scatter(pos[peak_idx], mag[peak_idx], color="red", label="Maxima")
        if valley_idx:
            ax.scatter(pos[valley_idx], mag[valley_idx], color="blue", label="Minima")

        ax.set_title("Standing Wave |P| vs Position (Live)")
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("|P| amplitude")
        ax.grid(True)
        ax.legend()

        self.kundt_canvas.draw()

        # Update text label with count of peaks/minima
        self.scan_result_label.setText(
            f"Live peaks: {len(peak_idx)}, minima: {len(valley_idx)}"
        )

    def on_start_cont_scan(self):
        if not self.ensure_connected():
            return

        # read parameters from GUI
        f_hz     = float(self.scan_freq_spin.value())
        start_mm = float(self.scan_start_spin.value())
        end_mm   = float(self.scan_end_spin.value())

        self.cont_scan_freq = f_hz
        self.cont_scan_end_mm = end_mm

        # reset buffers
        self.cont_scan_positions = []
        self.cont_scan_mags = []

        # 1) Home + go to start
        status, _ = send_command(self.serial_mgr.ser, CMD_HOME)
        if status != STS_ACK:
            QMessageBox.warning(self, "Continuous Scan", "Home failed to start.")
            return
        if not wait_until_home_complete(self.serial_mgr.ser):
            QMessageBox.warning(self, "Continuous Scan", "Homing timeout.")
            return

        # Move to start_mm once
        payload = struct.pack("<f", start_mm)
        status, _ = send_command(self.serial_mgr.ser, CMD_STEPPER_MOVE, payload)
        if status != STS_ACK:
            QMessageBox.warning(self, "Continuous Scan", "Move to start failed.")
            return
        if not wait_until_move_complete(self.serial_mgr.ser):
            QMessageBox.warning(self, "Continuous Scan", "Move-to-start timeout.")
            return

        # 2) Set excitation tone
        payload = struct.pack("<f", f_hz)
        status, _ = send_command(self.serial_mgr.ser, CMD_AD9833_SINE_FREQ, payload)
        if status != STS_ACK:
            QMessageBox.warning(self, "Continuous Scan", "Failed to set tone.")
            return

        # 3) Start ADC streaming
        send_command(self.serial_mgr.ser, CMD_START_SAMPLING)

        # 4) Start a long move from start_mm to end_mm (non-blocking on Python side)
        payload = struct.pack("<f", end_mm)
        status, _ = send_command(self.serial_mgr.ser, CMD_STEPPER_MOVE, payload)
        if status != STS_ACK:
            QMessageBox.warning(self, "Continuous Scan", "Scan move failed.")
            # stop ADC, mute
            send_command(self.serial_mgr.ser, CMD_STOP_SAMPLING)
            self._mute_speaker()
            return

        # 5) Setup timer to process frames & plot live
        if self.cont_scan_timer is None:
            self.cont_scan_timer = QTimer()
            self.cont_scan_timer.timeout.connect(self.on_cont_scan_tick)

        self.cont_scan_active = True
        self.cont_scan_timer.start(50)  # every 50 ms

    def on_stop_cont_scan(self):
        if self.cont_scan_timer:
            self.cont_scan_timer.stop()

        self.cont_scan_active = False

        # Stop ADC & mute tone
        send_command(self.serial_mgr.ser, CMD_STOP_SAMPLING)
        self._mute_speaker()

    def on_cont_scan_tick(self):
        if not self.cont_scan_active:
            return

        # 1) Read one ADC frame (blocking until frame available)
        samples = read_adc_frame(self.serial_mgr.ser)
        if samples is None:
            # Could be timeout; you can choose to stop scan or just skip
            return

        # 2) Get current position from MCU
        pos_mm = self.get_position_mm()
        if pos_mm is None:
            return

        # Stop condition: reached end of scan
        if pos_mm >= self.cont_scan_end_mm:
            self.on_stop_cont_scan()
            return

        # 3) Compute magnitude at excitation frequency
        mag, f_bin = fft_mag_at_freq(samples, self.FS, self.cont_scan_freq)

        # 4) Append to trace
        self.cont_scan_positions.append(pos_mm)
        self.cont_scan_mags.append(mag)

        # 5) Update live standing-wave plot with auto peak detection
        self._update_kundt_live_plot()

    def on_run_scan(self):
        if not self.ensure_connected():
            return
        if self.live_mode:
            self.on_stop_live()
        if self.cont_scan_active:
            self.on_stop_cont_scan()

        self.serial_mgr.ser.reset_input_buffer()
        
        f_hz     = float(self.scan_freq_spin.value())
        start_mm = float(self.scan_start_spin.value())
        end_mm   = float(self.scan_end_spin.value())

        coarse_step_mm = float(self.coarse_step_spin.value())
        fine_step_mm   = float(self.fine_step_spin.value())
        fine_window_mm = float(self.fine_window_spin.value())

        # Turn on tone
        payload = struct.pack("<f", f_hz)
        send_command(self.serial_mgr.ser, CMD_AD9833_SINE_FREQ, payload)

        # Run the two-stage scan
        try:
            result = scan_kundt_two_stage(
                self.serial_mgr.ser,
                f_hz,
                start_mm,
                end_mm,
                coarse_step_mm,
                fine_step_mm,
                fine_window_mm,
                self.FS    # sampling frequency from GUI
            )
        except Exception as e:
            QMessageBox.warning(self, "Scan Error", str(e))
            self._mute_speaker()
            return

        # Turn tone off
        self._mute_speaker()

        # ----------------------------
        # Extract results
        # ----------------------------

        coarse_pos = result["coarse_positions"]
        coarse_mag = result["coarse_mag"]

        fine_max_pos = result["fine_max_positions"]
        fine_max_mag = result["fine_max_mag"]

        fine_min_pos = result["fine_min_positions"]
        fine_min_mag = result["fine_min_mag"]

        # Find fine max/min
        idx_fmax = int(np.argmax(fine_max_mag))
        idx_fmin = int(np.argmin(fine_min_mag))

        real_pmax = fine_max_mag[idx_fmax]
        real_pmax_x = fine_max_pos[idx_fmax]

        real_pmin = fine_min_mag[idx_fmin]
        real_pmin_x = fine_min_pos[idx_fmin]

        # Compute SWR
        p_min_safe = max(real_pmin, 1e-9)
        SWR = real_pmax / p_min_safe
        R_mag = (SWR - 1.0) / (SWR + 1.0)

        # Output summary
        self.scan_result_label.setText(
            f"Coarse peak≈ {coarse_pos[np.argmax(coarse_mag)]:.1f} mm | "
            f"Refined Pmax={real_pmax:.3f} at {real_pmax_x:.2f} mm, "
            f"Pmin={real_pmin:.3f} at {real_pmin_x:.2f} mm, "
            f"SWR={SWR:.3f}, |R|={R_mag:.3f}"
        )

        # ----------------------------
        # Plot coarse + fine scans
        # ----------------------------
        ax = self.kundt_canvas.ax
        ax.clear()

        # coarse curve
        ax.plot(coarse_pos, coarse_mag, "k--", label="Coarse scan")

        # fine scans
        ax.plot(fine_max_pos, fine_max_mag, "r-", label="Fine region (Pmax)")
        ax.plot(fine_min_pos, fine_min_mag, "b-", label="Fine region (Pmin)")

        # markers
        ax.scatter([real_pmax_x], [real_pmax], c="red", s=80)
        ax.scatter([real_pmin_x], [real_pmin], c="blue", s=80)

        ax.set_title("Two-stage Kundt Scan")
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("|P| amplitude")
        ax.grid(True)
        ax.legend()

        self.kundt_canvas.draw()

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
    QWidget {
        font-size: 18px;
    }
    QPushButton {
        font-size: 20px;
        padding: 12px;
        min-height: 40px;
    }
    QLineEdit, QDoubleSpinBox, QSpinBox {
        font-size: 20px;
        min-height: 40px;     /* match button height */
        padding: 6px;         /* improves internal spacing */
    }
    QTabBar::tab {
        font-size: 18px;
        padding: 10px 20px;
    }
""")

    win = MainWindow()
    win.resize(1980, 1380)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
