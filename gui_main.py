import sys
import struct
import numpy as np
import serial
import time

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QGridLayout, QComboBox,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QAction, QActionGroup, QMenuBar,
    QGroupBox, QGridLayout, QMessageBox, QTabWidget, QButtonGroup, QAction, QFileDialog, QMessageBox
)
from PyQt5.QtSerialPort import QSerialPortInfo
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence

from commands import *               # FS, command IDs
from serial_comm import send_command
from adc_stream import read_adc_frame, read_packet
from motor_control import wait_until_move_complete, wait_until_home_complete
from scan_kundt import fft_mag_at_freq


class SerialManager:
    def __init__(self):
        self.ser = None

    def connect(self, port, baud):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = serial.Serial(port, baud, timeout=0.3)

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

class ScanResultDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan Results")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        grid = QGridLayout()

        self.labels = {}

        fields = [
            ("Mode", "mode"),
            ("Pmax", "Pmax"),
            ("Pmax position (mm)", "x_max"),
            ("Pmin", "Pmin"),
            ("Pmin position (mm)", "x_min"),
            ("SWR", "SWR"),
            ("|R|", "R"),
            ("Points", "points"),
        ]

        for row, (title, key) in enumerate(fields):
            lbl_title = QLabel(f"{title}:")
            lbl_value = QLabel("-")
            lbl_title.setStyleSheet("font-weight: bold;")
            grid.addWidget(lbl_title, row, 0)
            grid.addWidget(lbl_value, row, 1)
            self.labels[key] = lbl_value

        layout.addLayout(grid)

    def update_results(self, results: dict):
        if not results:
            return

        for key, lbl in self.labels.items():
            if key in results:
                val = results[key]
                if isinstance(val, float):
                    lbl.setText(f"{val:.6g}")
                else:
                    lbl.setText(str(val))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_port = None
        self.setWindowTitle("Kundt Tube Controller GUI")
        self.serial_mgr = SerialManager()
        self.last_samples = None
        self.live_mode = False
        self.live_timer = None

        self.FS = 20000.0      # default sampling rate
        self.FFT_N = 4096       # default FFT length

        # --- Step-scan synchronization ---
        self.settle_ms = 50            # wait after each move before sampling (typ. 50–200 ms)

        # Zoom parameters
        self.adc_zoom_factor = 15.0
        self.adc_zoom_center = 0
        # Y-axis zoom parameters
        self.y_zoom_factor = 5.0
        self.y_center = 1.65   # midpoint of 0–3.3 V

        self.pulses_per_mm = 3200.0 / (np.pi * 14.0)  # temporary: match MCU
        self.tim3_psc = 840 - 1
        self.tim3_arr = 24

        self.last_scan_results = None

        self._build_ui()
        self._build_menubar()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.showNormal()
    def closeEvent(self, event):
        try:
            self.serial_mgr.disconnect()
        except Exception:
            pass
        event.accept()

    def _refresh_com_ports(self):
            self.port_combo.clear()

            ports = QSerialPortInfo.availablePorts()
            for port in ports:
                # Show user-friendly name, keep system port internally
                self.port_combo.addItem(
                    f"{port.portName()} — {port.description()}",
                    port.portName()
                )

            if self.port_combo.count() == 0:
                self.port_combo.addItem("No ports found", None)
                self.port_combo.setEnabled(False)
            else:
                self.port_combo.setEnabled(True)

    def _build_menubar(self):
        menubar = self.menuBar()  # QMainWindow built-in

        # Example standard menus
        file_menu = menubar.addMenu("File")
        save_kundt_action = QAction("Save Kundt Scan…", self)
        save_kundt_action.triggered.connect(self.on_save_kundt_scan)

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)  # Ctrl+Q (Cmd+Q on mac)
        exit_action.triggered.connect(self.close)   # closes main window -> exits app
        file_menu.addSeparator()

        file_menu.addAction(save_kundt_action)
        file_menu.addAction(exit_action)

        
        edit_menu = menubar.addMenu("Edit")

        # Connect menu (what you asked for)
        connect_menu = menubar.addMenu("Connect")

        # Submenu: Ports
        self.ports_menu = connect_menu.addMenu("Port")

        # Make ports mutually exclusive (radio behavior)
        self.port_action_group = QActionGroup(self)
        self.port_action_group.setExclusive(True)

        # Refresh ports
        refresh_ports_action = QAction("Refresh Ports", self)
        refresh_ports_action.triggered.connect(self._refresh_ports_menu)
        connect_menu.addAction(refresh_ports_action)

        connect_menu.addSeparator()

        # Connect / Disconnect actions
        self.connect_action = QAction("Connect", self)
        self.connect_action.triggered.connect(self.on_connect)

        self.disconnect_action = QAction("Disconnect", self)
        self.disconnect_action.triggered.connect(self.on_disconnect)

        connect_menu.addAction(self.connect_action)
        connect_menu.addAction(self.disconnect_action)

        # Initial population
        self._refresh_ports_menu()


    def _refresh_ports_menu(self):
        self.ports_menu.clear()
        for a in list(self.port_action_group.actions()):
            self.port_action_group.removeAction(a)

        ports = QSerialPortInfo.availablePorts()

        if not ports:
            no_ports = QAction("No ports found", self)
            no_ports.setEnabled(False)
            self.ports_menu.addAction(no_ports)
            self.selected_port = None
            return

        # Populate ports as checkable actions
        for p in ports:
            label = p.portName()
            desc = p.description().strip()
            if desc:
                label += f" — {desc}"

            act = QAction(label, self)
            act.setCheckable(True)
            act.setData(p.portName())  # store "COMx"
            act.triggered.connect(self._on_port_selected)

            self.port_action_group.addAction(act)
            self.ports_menu.addAction(act)

        # Auto-select:
        # 1) keep current selection if still present
        # 2) else select first port
        selected = None
        if self.selected_port:
            for act in self.port_action_group.actions():
                if act.data() == self.selected_port:
                    selected = act
                    break

        if selected is None:
            selected = self.port_action_group.actions()[0]
            self.selected_port = selected.data()

        selected.setChecked(True)

    def _on_port_selected(self):
        act = self.sender()
        if act is None:
            return
        self.selected_port = act.data()

    def on_save_kundt_scan(self):
        if not hasattr(self, "kundt_canvas"):
            QMessageBox.warning(self, "Save", "No Kundt plot available.")
            return

        # Ask user for filename
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Kundt Scan",
            "kundt_scan.png",
            "PNG Image (*.png)"
        )

        if not path:
            return  # user cancelled

        try:
            # Save plot
            self.kundt_canvas.figure.savefig(
                path,
                dpi=300,
                bbox_inches="tight"
            )

            # Optional: save numeric results next to the image
            if hasattr(self, "last_scan_results"):
                import json, os
                base, _ = os.path.splitext(path)
                data_path = base + ".json"

                with open(data_path, "w") as f:
                    json.dump(self.last_scan_results, f, indent=2)

            QMessageBox.information(
                self,
                "Save",
                f"Kundt scan saved successfully:\n{path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save failed",
                f"Could not save Kundt scan:\n{e}"
            )

    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_motor_tab(), "Position")
        tabs.addTab(self._build_signal_tab(), "Signal / ADC / FFT")
        tabs.addTab(self._build_kundt_tab(), "Kundt Scan")
        tabs.addTab(self._build_gain_tab(), "Gain (PGA113)")
        tabs.addTab(self._build_temp_tab(), "Temperature")
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
        self.home_offset_spin.setValue(-2.5)

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

        # ---------------- Signal Generator ----------------
        freq_group = QGroupBox("Signal Generator (AD9833)")
        fg_layout = QHBoxLayout(freq_group)

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

        # ---------------- FFT Settings ----------------
        fft_group = QGroupBox("FFT Settings")
        fft_layout = QHBoxLayout(fft_group)

        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(100.0, 500000.0)
        self.fs_spin.setValue(20000.0)
        self.fs_spin.setDecimals(1)

        self.fft_n_spin = QSpinBox()
        self.fft_n_spin.setRange(256, 16384)
        self.fft_n_spin.setSingleStep(256)
        self.fft_n_spin.setValue(4096)

        btn_apply_fft = QPushButton("Apply FFT Settings")
        btn_apply_fft.clicked.connect(self.on_apply_fft_settings)

        fft_layout.addWidget(QLabel("Sampling FS (Hz):"))
        fft_layout.addWidget(self.fs_spin)
        fft_layout.addWidget(QLabel("FFT size N:"))
        fft_layout.addWidget(self.fft_n_spin)
        fft_layout.addWidget(btn_apply_fft)

        # ---------------- ADC / Capture (RIGHT SIDE) ----------------
        adc_group = QGroupBox("ADC / Capture")
        adc_layout = QHBoxLayout(adc_group)

        btn_live_start = QPushButton("Start Live View")
        btn_live_start.clicked.connect(self.on_start_live)
        btn_live_start.setStyleSheet(
            "QPushButton {"
            "  background-color: #2e7d32;"
            "  color: white;"
            "  font-weight: bold;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #a5d6a7;"
            "  color: #eeeeee;"
            "}"
        )


        btn_live_stop = QPushButton("Stop Live View")
        btn_live_stop.clicked.connect(self.on_stop_live)
        btn_live_stop.setStyleSheet(
            "QPushButton {"
            "  background-color: #c62828;"
            "  color: white;"
            "  font-weight: bold;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #ef9a9a;"
            "  color: #eeeeee;"
            "}"
        )
        adc_layout.addWidget(btn_live_start)
        adc_layout.addWidget(btn_live_stop)

        # ================= TOP ROW: left stack + right group =================
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        left_stack = QVBoxLayout()
        left_stack.setSpacing(10)
        left_stack.addWidget(freq_group)
        left_stack.addWidget(fft_group)

        top_row.addLayout(left_stack, stretch=1)
        top_row.addWidget(adc_group, stretch=0)

        layout.addLayout(top_row)
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

        self.btn_mode_rms = QPushButton("RMS Scan1")
        self.btn_mode_rms_cont = QPushButton("RMS Scan")
        self.btn_mode_fft      = QPushButton("FFT Scan")
        
        for b in (self.btn_mode_rms, self.btn_mode_rms_cont, self.btn_mode_fft):
            b.setCheckable(True)

        self.scan_mode_group = QButtonGroup(self)
        self.scan_mode_group.setExclusive(True)
        self.scan_mode_group.addButton(self.btn_mode_rms, 0)  # 0 = RMS
        self.scan_mode_group.addButton(self.btn_mode_rms_cont, 1)  # RMS continuous
        self.scan_mode_group.addButton(self.btn_mode_fft, 2)  # 1 = FFT
        self.btn_mode_fft.setChecked(True)  # default

        #mode_row_layout.addWidget(self.btn_mode_rms)
        mode_row_layout.addWidget(self.btn_mode_rms_cont)
        mode_row_layout.addWidget(self.btn_mode_fft)

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
        self.scan_freq_spin.setRange(90.0, 2000.0)
        self.scan_freq_spin.setValue(1000.0)
        self.scan_freq_spin.setDecimals(2)

        self.extrema_count_spin = QSpinBox()
        self.extrema_count_spin.setRange(1, 20)
        self.extrema_count_spin.setValue(3)  # default: 3 maxima + 3 minima

        basic_grid.addWidget(QLabel("Extrema per type (max & min):"), 3, 0)
        basic_grid.addWidget(self.extrema_count_spin,              3, 1)


        self.scan_start_spin = compact_spin(QDoubleSpinBox())
        self.scan_start_spin.setRange(-2.5, 500.0)
        self.scan_start_spin.setValue(-2.0)
        self.scan_start_spin.setDecimals(2)

        self.scan_end_spin = compact_spin(QDoubleSpinBox())
        self.scan_end_spin.setRange(0.0, 900.0)
        self.scan_end_spin.setValue(0.0)
        self.scan_end_spin.setDecimals(2)

        basic_grid.addWidget(QLabel("Frequency (Hz):"), 0, 0)
        basic_grid.addWidget(self.scan_freq_spin,       0, 1)
        basic_grid.addWidget(QLabel("Start (mm):"),     1, 0)
        basic_grid.addWidget(self.scan_start_spin,      1, 1)
        basic_grid.addWidget(QLabel("End (mm):"),       2, 0)
        basic_grid.addWidget(self.scan_end_spin,        2, 1)

        params_row.addWidget(basic_group, 1)

        # ---- Middle: Acquisition (FS, N, FFT resolution) ----
        acq_group = QGroupBox("Acquisition")
        acq_grid = QGridLayout(acq_group)
        acq_grid.setContentsMargins(8, 8, 8, 8)
        acq_grid.setHorizontalSpacing(10)
        acq_grid.setVerticalSpacing(6)

        self.rms_win_spin = QSpinBox()
        self.rms_win_spin.setRange(128, 16384)
        self.rms_win_spin.setSingleStep(128)
        self.rms_win_spin.setValue(1024)

        self.rms_hop_spin = QSpinBox()
        self.rms_hop_spin.setRange(64, 16384)
        self.rms_hop_spin.setSingleStep(64)
        self.rms_hop_spin.setValue(218)

        acq_grid.addWidget(QLabel("RMS window N:"), 0, 2)
        acq_grid.addWidget(self.rms_win_spin,       0, 3)

        # Spatial phase resolution (deg) for continuous RMS while moving
        self.rms_phase_deg_spin = QDoubleSpinBox()
        self.rms_phase_deg_spin.setRange(0.5, 10.0)
        self.rms_phase_deg_spin.setSingleStep(0.5)
        self.rms_phase_deg_spin.setDecimals(1)
        self.rms_phase_deg_spin.setValue(1.0)  # professor default = 1°

        # Add to the same layout as RMS window/hop (adjust row/col as needed)
        acq_grid.addWidget(QLabel("Spatial phase (deg):"), 1, 3)
        acq_grid.addWidget(self.rms_phase_deg_spin,       1, 4)


        btn_apply_acq = QPushButton("Apply")
        btn_apply_acq.clicked.connect(self.on_apply_kundt_acq)
        acq_grid.addWidget(btn_apply_acq, 5, 1)


        self.kundt_fs_spin = compact_spin(QDoubleSpinBox())
        self.kundt_fs_spin.setRange(100.0, 500_000.0)
        self.kundt_fs_spin.setDecimals(1)
        self.kundt_fs_spin.setValue(float(getattr(self, "FS", 20000.0)))

        self.kundt_n_spin = compact_spin(QSpinBox())
        self.kundt_n_spin.setRange(256, 16384)
        self.kundt_n_spin.setSingleStep(256)
        self.kundt_n_spin.setValue(int(getattr(self, "FFT_N", 4096)))

        self.lbl_fft_res = QLabel("")  # will be filled by updater

        acq_grid.addWidget(QLabel("Sampling FS (Hz):"), 0, 0)
        acq_grid.addWidget(self.kundt_fs_spin,         0, 1)
        acq_grid.addWidget(QLabel("Sample size N:"),   1, 0)
        acq_grid.addWidget(self.kundt_n_spin,          1, 1)
        acq_grid.addWidget(QLabel("FFT resolution:"),  2, 0)
        acq_grid.addWidget(self.lbl_fft_res,           2, 1)

        params_row.addWidget(acq_group, 1)

        def _update_fft_resolution_label():
            fs = float(self.kundt_fs_spin.value())
            n = int(self.kundt_n_spin.value())
            self.FS = fs
            self.FFT_N = n
            df = fs / max(n, 1)
            self.lbl_fft_res.setText(f"Δf = {df:.3f} Hz/bin")

        self.kundt_fs_spin.valueChanged.connect(lambda _=None: _update_fft_resolution_label())
        self.kundt_n_spin.valueChanged.connect(lambda _=None: _update_fft_resolution_label())
        _update_fft_resolution_label()

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

        # Put step_group into the right column container
        right_v.addWidget(step_group)

        # Assemble the two columns into the row, then add to main layout
        params_row.addWidget(basic_group, 1)
        params_row.addWidget(right_container, 1)
        layout.addLayout(params_row)


# ---------------- Action buttons (right aligned) ----------------
        run_row = QHBoxLayout()
        run_row.addStretch()  # pushes button to the right

        self.btn_run_scan = QPushButton("Run Scan")
        self.btn_run_scan.clicked.connect(self.on_run_scan)

        run_row.addWidget(self.btn_run_scan)
        layout.addLayout(run_row)
        right_v.addWidget(self.btn_run_scan)


        # ---------------- Plot + results ----------------
        self.kundt_canvas = MplCanvas(self, width=5, height=3)
        self.kundt_canvas.setMinimumHeight(600)
        layout.addWidget(self.kundt_canvas)

        #self.scan_result_label = QLabel("Results: -")
        #layout.addWidget(self.scan_result_label)
        btn_show_results = QPushButton("Show Results")
        btn_show_results.clicked.connect(self.on_show_results)
        layout.addWidget(btn_show_results)

        layout.addStretch()

        return w
    # ---------------- Gain tab ----------------
    def _build_gain_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        layout.addWidget(QLabel("PGA113 gain:"))

        self.pga_gain_combo = QComboBox()
        self.pga_gain_combo.addItems(["1x", "2x", "5x", "10x", "20x", "50x", "100x", "200x"])
        layout.addWidget(self.pga_gain_combo)

        btn_set_pga = QPushButton("Set Gain (PGA113)")
        btn_set_pga.clicked.connect(self.on_set_pga_gain)
        layout.addWidget(btn_set_pga)

        layout.addStretch()
        return w

    # ---------------- Temperature tab ----------------
    def _build_temp_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        self.temp_label = QLabel("Temperature: --.- °C")
        self.temp_label.setAlignment(Qt.AlignCenter)
        self.temp_label.setStyleSheet("font-size: 28px; font-weight: 600;")

        btn_temp = QPushButton("Temperature")   # button name exactly “Temperature”
        btn_temp.setMinimumHeight(80)
        btn_temp.clicked.connect(self.on_read_temperature)

        layout.addStretch()
        layout.addWidget(self.temp_label)
        layout.addSpacing(20)
        layout.addWidget(btn_temp)
        layout.addStretch()

        return w

    # ---------------- Helpers ----------------
    def ensure_connected(self):
        if not self.serial_mgr.is_connected():
            QMessageBox.warning(self, "Not connected", "Please connect to the serial port first.")
            return False
        return True
    
    def mcu_get_temperature_c(self) -> float:
        """
        Query MCU for BME280 temperature.
        Returns temperature in °C (float).
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected")

        status, pl = send_command(self.serial_mgr.ser, CMD_READ_TEMP)

        if status != STS_ACK or pl is None or len(pl) != 4:
            raise RuntimeError(f"GET_TEMP failed (status={status}, payload_len={0 if pl is None else len(pl)})")

        return struct.unpack("<f", pl)[0]

    # ---------------- Slots ----------------
    def on_connect(self):
        port_name = getattr(self, "selected_port", None)
        if not port_name:
            QMessageBox.warning(
                self,
                "Connection",
                "No COM port selected.\n\nUse: Connect → Port"
            )
            return

        baud = 115200  # <-- set to whatever your MCU uses

        try:
            print(f"Connecting to {port_name} @ {baud}...")
            self.serial_mgr.connect(port_name, baud)

            # Optional: keep for display/logging
            self.serial_port = port_name

            QMessageBox.information(self, "Connected", f"Connected to {port_name} @ {baud}.")
        except serial.SerialException as e:
            self.serial_mgr.disconnect()
            QMessageBox.critical(self, "Connection failed", f"Could not open {port_name}:\n{e}")

    def on_disconnect(self):
        self.serial_mgr.disconnect()
        QMessageBox.information(self, "Disconnected", "Serial port closed.")

    def on_get_temperature(self):
        if not self.ensure_connected():
            return

        # Safety: don't send framed commands while ADC streaming
        if getattr(self, "live_mode", False):
            QMessageBox.warning(self, "Temperature", "Stop Live View before reading temperature.")
            return

        try:
            t = self.mcu_get_temperature_c()  # you add/keep this helper
            if hasattr(self, "temp_value_label"):
                self.temp_value_label.setText(f"{t:.2f} °C")
        except Exception as e:
            QMessageBox.warning(self, "Temperature", str(e))

    def on_toggle_dir(self):
        if not self.ensure_connected(): return
        status, _ = send_command(self.serial_mgr.ser, CMD_TOGGLE_DIR)
        QMessageBox.information(self, "Toggle DIR", f"Status: {status}")

    def on_set_dir(self, val):
        if not self.ensure_connected(): return
        status, _ = send_command(self.serial_mgr.ser, CMD_SET_DIR, bytes([val]))
        QMessageBox.information(self, "Set DIR", f"Status: {status}")
    
    def _motor_velocity_mm_s(self) -> float:
        # TIM3 clock is 84 MHz with your clock config
        tim3_clk = 84_000_000.0
        f_step = tim3_clk / ((self.tim3_psc + 1.0) * (self.tim3_arr + 1.0))  # pulses/s
        return (f_step / float(self.pulses_per_mm))/2

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
    def on_apply_kundt_acq(self):
        fs_ui = float(self.kundt_fs_spin.value())
        n_ui  = int(self.kundt_n_spin.value())

        try:
            fs_eff = self.mcu_set_sampling_freq(fs_ui)
        except Exception as e:
            QMessageBox.warning(self, "Acquisition", f"Failed to set MCU sampling rate:\n{e}")
            return

        self.FS = fs_eff
        self.FFT_N = n_ui

        # reflect quantized value
        self.kundt_fs_spin.blockSignals(True)
        self.kundt_fs_spin.setValue(fs_eff)
        self.kundt_fs_spin.blockSignals(False)

        # update label
        df = fs_eff / max(n_ui, 1)
        self.lbl_fft_res.setText(f"Δf = {df:.3f} Hz/bin")

    def mcu_set_sampling_freq(self, fs_hz: float) -> float:
        """
        Ask MCU to set ADC sampling frequency.
        Returns the effective sampling frequency reported by MCU (float Hz).
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected")

        fs_req = int(round(float(fs_hz)))
        payload = struct.pack("<I", fs_req)  # uint32 little-endian

        status, pl = send_command(self.serial_mgr.ser, CMD_SET_SAMPLING_FREQ, payload)

        if status != STS_ACK or pl is None or len(pl) < 4:
            raise RuntimeError(f"SET_SAMPLING_FREQ failed (status={status}, payload={pl})")

        fs_eff = struct.unpack("<I", pl[:4])[0]
        return float(fs_eff)
    def mcu_get_sampling_freq(self) -> float:
        """
        Query MCU for current effective ADC sampling frequency (Hz).
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected")

        status, pl = send_command(self.serial_mgr.ser, CMD_GET_SAMPLING_FREQ)

        if status != STS_ACK or pl is None or len(pl) < 4:
            raise RuntimeError(f"GET_SAMPLING_FREQ failed (status={status}, payload={pl})")

        fs_eff = struct.unpack("<I", pl[:4])[0]
        return float(fs_eff)
    def on_read_temperature(self):
        if not self.ensure_connected():
            return
        try:
            t = self.mcu_get_temperature_c()
            self.temp_label.setText(f"Temperature: {t:.2f} °C")
        except Exception as e:
            QMessageBox.warning(self, "Temperature", str(e))

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
        fs_ui = float(self.fs_spin.value())
        n_ui  = int(self.fft_n_spin.value())

        # 1) Push FS to MCU and read back effective FS
        try:
            fs_eff = self.mcu_set_sampling_freq(fs_ui)
        except Exception as e:
            QMessageBox.warning(self, "FFT Settings", f"Failed to set MCU sampling rate:\n{e}")
            return

        # 2) Update GUI state with *effective* FS (important: FFT axis must match real sampling)
        self.FS = fs_eff
        self.FFT_N = n_ui

        # Update the spinbox to reflect quantization, so user sees the truth
        self.fs_spin.blockSignals(True)
        self.fs_spin.setValue(fs_eff)
        self.fs_spin.blockSignals(False)

        QMessageBox.information(
            self, "FFT Settings",
            f"MCU Sampling FS set to {fs_eff:.1f} Hz (requested {fs_ui:.1f} Hz)\n"
            f"FFT size N = {self.FFT_N}"
        )

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

    def on_set_pga_gain(self):
        if not self.ensure_connected():
            return

        # Safety: don't send framed commands while ADC streaming
        if getattr(self, "live_mode", False):
            QMessageBox.warning(self, "PGA113 Gain", "Stop Live View before changing PGA gain.")
            return

        gain_map = {
            "1x": 0, "2x": 1, "5x": 2, "10x": 3,
            "20x": 4, "50x": 5, "100x": 6, "200x": 7,
        }

        label = self.pga_gain_combo.currentText()
        gain_code = gain_map[label]

        # CMD_PGA_SET_GAIN must exist in commands.py
        status, pl = send_command(self.serial_mgr.ser, CMD_PGA_SET, bytes([gain_code]))

        if status == STS_ACK:
            QMessageBox.information(self, "PGA113 Gain", f"PGA113 gain set to {label}.")
        else:
            extra = ""
            # If MCU returns [gain_code, hal_status] on error, show it
            if pl and len(pl) >= 2:
                extra = f"\nReturned: gain_code={pl[0]}, hal_status={pl[1]}"
            QMessageBox.warning(self, "PGA113 Gain", f"Set failed (status={status}).{extra}")


    def _mute_speaker(self):
        payload = struct.pack("<f", 0.0)
        send_command(self.serial_mgr.ser, CMD_AD9833_SINE_FREQ, payload)

    def _move_abs_mm(self, pos_mm: float):
        status, _ = send_command(self.serial_mgr.ser, CMD_STEPPER_MOVE, struct.pack("<f", float(pos_mm)))
        if status != STS_ACK:
            raise RuntimeError(f"Move failed (to {pos_mm:.2f} mm).")
        if not wait_until_move_complete(self.serial_mgr.ser):
            raise RuntimeError(f"Move timeout (to {pos_mm:.2f} mm).")
    
    def _settle_after_move(self):
        """Simple fixed delay after motor move (Option A)."""
        if self.settle_ms > 0:
            time.sleep(self.settle_ms / 1000.0)

    def get_position_mm(self):
        status, payload = send_command(self.serial_mgr.ser, CMD_GET_POSITION)
        if status != STS_ACK or not payload or len(payload) < 4:
            return None
        return struct.unpack("<f", payload[:4])[0]

    def find_peaks_and_valleys(self, positions, mags):
        """
        Mathematical extrema detection:
        - Uses derivative sign changes
        - Enforces strict max/min alternation
        - No arbitrary mm thresholds
        """

        x = np.asarray(positions, dtype=float)
        y = np.asarray(mags, dtype=float)

        if len(y) < 3:
            return [], []

        # Ensure sorted by x
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        orig_idx = order

        # First derivative (central difference)
        dy = np.gradient(y, x)

        peak_idx = []
        valley_idx = []

        # Detect zero-crossings in derivative
        for i in range(1, len(dy)):
            if dy[i-1] > 0 and dy[i] <= 0:
                peak_idx.append(i)
            elif dy[i-1] < 0 and dy[i] >= 0:
                valley_idx.append(i)

        # ---- Enforce alternation ----
        extrema = []

        for i in peak_idx:
            extrema.append(("max", i))
        for i in valley_idx:
            extrema.append(("min", i))

        # sort by x-position
        extrema.sort(key=lambda t: x[t[1]])

        filtered = []
        last_type = None

        for t, i in extrema:
            if t == last_type:
                # keep the stronger one
                if t == "max":
                    if y[i] > y[filtered[-1][1]]:
                        filtered[-1] = (t, i)
                else:
                    if y[i] < y[filtered[-1][1]]:
                        filtered[-1] = (t, i)
            else:
                filtered.append((t, i))
                last_type = t

        # Split back into peaks and valleys
        peaks = [orig_idx[i] for t, i in filtered if t == "max"]
        valleys = [orig_idx[i] for t, i in filtered if t == "min"]

        return peaks, valleys

    def find_extrema_standing_wave(
        self,
        x,
        y,
        f_hz: float,
        T_c: float,
        min_sep_frac_lambda: float = 1/8,
        include_endpoints: bool = False,
        smooth_frac_lambda: float = 1/20
    ):
        """
        Standing-wave extrema finder (improved):

        - Sorts by x (robust).
        - Lightly smooths y to suppress ripple noise (window based on wavelength).
        - Finds extrema via derivative sign changes on the smoothed y.
        - Refines each extrema by local search on the ORIGINAL y (snaps markers).
        - Enforces alternation and minimum spacing (~fraction of wavelength).
        - Returns indices into ORIGINAL (unsorted) arrays.
        """
        import numpy as np

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 7 or y.size != x.size:
            return np.array([], dtype=int), np.array([], dtype=int)

        # ---- sort by x ----
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]

        # speed of sound and wavelength
        c = 331.3 + 0.606 * float(T_c)                 # m/s
        lam_mm = (c / max(float(f_hz), 1e-9)) * 1000.0 # mm

        min_sep_mm = max(1e-6, float(min_sep_frac_lambda) * lam_mm)

        # ---- smoothing window based on wavelength ----
        # choose a smoothing span ~ lam * smooth_frac_lambda (in mm), convert to samples
        dx_med = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
        smooth_mm = max(dx_med, float(smooth_frac_lambda) * lam_mm)
        smooth_N = int(round(smooth_mm / max(dx_med, 1e-9)))

        # keep it odd and within sane bounds
        smooth_N = max(5, min(smooth_N, 101))
        if smooth_N % 2 == 0:
            smooth_N += 1

        # simple moving-average smoothing (no scipy needed)
        if smooth_N >= 5:
            kernel = np.ones(smooth_N, dtype=float) / float(smooth_N)
            ys_s = np.convolve(ys, kernel, mode="same")
        else:
            ys_s = ys.copy()

        # ---- derivative sign changes on smoothed curve ----
        dy = np.diff(ys_s)
        s = np.sign(dy)

        # fill zeros to avoid missing sign changes
        for i in range(1, s.size):
            if s[i] == 0:
                s[i] = s[i - 1]
        if s.size and s[0] == 0:
            nz = np.nonzero(s)[0]
            if nz.size:
                s[: nz[0] + 1] = s[nz[0]]

        cand_peaks = []
        cand_valleys = []
        for i in range(1, s.size):
            if s[i - 1] > 0 and s[i] < 0:
                cand_peaks.append(i)
            elif s[i - 1] < 0 and s[i] > 0:
                cand_valleys.append(i)

        cand_peaks = np.array(cand_peaks, dtype=int)
        cand_valleys = np.array(cand_valleys, dtype=int)

        # optional endpoints
        if include_endpoints and dy.size >= 2:
            if dy[0] < 0:
                cand_peaks = np.concatenate(([0], cand_peaks))
            elif dy[0] > 0:
                cand_valleys = np.concatenate(([0], cand_valleys))

            if dy[-1] > 0:
                cand_peaks = np.concatenate((cand_peaks, [xs.size - 1]))
            elif dy[-1] < 0:
                cand_valleys = np.concatenate((cand_valleys, [xs.size - 1]))

        # ---- refine candidates: snap to true extrema in ORIGINAL ys within ±refine_N ----
        # refine span: about 1/16 λ, but at least a few samples
        refine_mm = max(2.0 * dx_med, (lam_mm / 16.0))
        refine_N = int(round(refine_mm / max(dx_med, 1e-9)))
        refine_N = max(3, min(refine_N, 200))

        def refine_idx(i0: int, kind: str) -> int:
            lo = max(0, i0 - refine_N)
            hi = min(xs.size, i0 + refine_N + 1)
            seg = ys[lo:hi]
            if seg.size == 0:
                return i0
            if kind == "peak":
                j = int(np.argmax(seg))
            else:
                j = int(np.argmin(seg))
            return lo + j

        peaks = [refine_idx(int(i), "peak") for i in cand_peaks]
        valleys = [refine_idx(int(i), "valley") for i in cand_valleys]

        # ---- merge events, sort, enforce alternation and spacing ----
        events = []
        for i in peaks:
            events.append((xs[i], i, "peak", ys[i]))
        for i in valleys:
            events.append((xs[i], i, "valley", ys[i]))
        events.sort(key=lambda t: t[0])

        if not events:
            return np.array([], dtype=int), np.array([], dtype=int)

        # alternation: if same type, keep stronger
        alt = []
        for ev in events:
            if not alt:
                alt.append(ev)
                continue
            if ev[2] != alt[-1][2]:
                alt.append(ev)
            else:
                if ev[2] == "peak":
                    if ev[3] > alt[-1][3]:
                        alt[-1] = ev
                else:
                    if ev[3] < alt[-1][3]:
                        alt[-1] = ev

        # spacing: if too close, keep stronger
        filtered = []
        for ev in alt:
            if not filtered:
                filtered.append(ev)
                continue
            if (ev[0] - filtered[-1][0]) >= min_sep_mm:
                filtered.append(ev)
            else:
                prev = filtered[-1]
                if ev[2] == "peak":
                    if ev[3] > prev[3]:
                        filtered[-1] = ev
                else:
                    if ev[3] < prev[3]:
                        filtered[-1] = ev

        peak_idx_sorted = np.array([i for _, i, t, _ in filtered if t == "peak"], dtype=int)
        valley_idx_sorted = np.array([i for _, i, t, _ in filtered if t == "valley"], dtype=int)

        # map back to original indexing
        peak_idx = order[peak_idx_sorted] if peak_idx_sorted.size else np.array([], dtype=int)
        valley_idx = order[valley_idx_sorted] if valley_idx_sorted.size else np.array([], dtype=int)

        return peak_idx, valley_idx

    def _tone_rms_iq(self, x: np.ndarray, fs: float, f0: float) -> float:
        """
        Narrowband RMS at f0 using quadrature (no FFT).
        Returns RMS of the sinusoidal component at f0.
        """
        N = len(x)
        n = np.arange(N, dtype=np.float32)
        w = 2.0 * np.pi * float(f0) / float(fs)

        c = np.cos(w * n)
        s = np.sin(w * n)

        # I/Q correlation (acts like narrowband extraction)
        I = float(np.mean(x * c))
        Q = float(np.mean(x * s))

        A = 2.0 * np.sqrt(I*I + Q*Q)   # estimate sine amplitude
        return A / np.sqrt(2.0)        # convert amplitude -> RMS

    def _rms_continuous_scan_fast(
        self,
        start_mm: float,
        end_mm: float,
        window_N: int,
        hop_N: int,
        skip_ms: float = 150.0
    ):
        """
        Continuous RMS scan while motor moves from start_mm to end_mm.

        Fix vs. old version:
        - Spatial mapping uses *measured* start/end positions (CMD_GET_POSITION)
        instead of assuming the move duration from v_mm_s.
        - Still avoids framed commands during streaming (USB ADC packets only).
        """
        fs = float(self.FS)
        if fs <= 0:
            raise RuntimeError("Invalid FS. Ensure MCU sampling frequency is set/read correctly.")

        user_window_N = int(window_N)
        if user_window_N <= 0:
            raise ValueError("Require window_N > 0")

        user_hop_N = int(hop_N)
        if user_hop_N <= 0:
            user_hop_N = user_window_N

        # --- Frequency and motion parameters (needed for window sizing) ---
        f_hz = float(self.scan_freq_spin.value())  # your current design
        if f_hz <= 0:
            raise RuntimeError("Invalid excitation frequency.")

        v_mm_s = abs(float(self._motor_velocity_mm_s()))
        if v_mm_s < 1e-6:
            v_mm_s = 20.6  # fallback

        # Temperature-based speed of sound (must be BEFORE streaming)
        T_c = float(self.mcu_get_temperature_c())
        self._last_scan_temp_c = float(T_c)  # optional: let caller reuse
        c_m_s = 331.3 + 0.606 * T_c

        # 1° phase criterion
        # dx = λ/360, dt_max = dx/v, M_max = fs*dt_max
        v_m_s = v_mm_s / 1000.0
        lam_m = c_m_s / float(f_hz)
        dt_max = lam_m / (360.0 * max(v_m_s, 1e-9))
        M_max = int(np.floor(fs * dt_max))

        # Use window limited by M_max (but keep a practical minimum)
        window_N = max(128, min(user_window_N, max(128, M_max)))

        # Hop sanity
        hop_eff = int(user_hop_N)
        if hop_eff <= 0 or hop_eff > window_N:
            hop_eff = window_N

        # Discard startup transient (in samples, based on window midpoint index)
        skip_samples = int(round((float(skip_ms) / 1000.0) * fs))
        if skip_samples < 0:
            skip_samples = 0

        # -------------------------
        # Measure *actual* start pos
        # -------------------------
        # (Framed command is OK here because we're NOT streaming yet)
        #p0 = self.get_position_mm()
        #if p0 is None:
        #    p0 = float(start_mm)
        #measured_start_mm = float(p0)

        buf = np.empty(0, dtype=np.uint16)
        total_received = 0  # total ADC samples received since start of streaming

        mid_samples = []   # window midpoint sample indices (global)
        rms_vals = []

        # Flush stale packets, start streaming
        self.serial_mgr.ser.reset_input_buffer()
        status, _ = send_command(self.serial_mgr.ser, CMD_START_SAMPLING)
        if status != STS_ACK:
            raise RuntimeError("Failed to start sampling.")

        # Start move to end (absolute)
        status, _ = send_command(self.serial_mgr.ser, CMD_STEPPER_MOVE, struct.pack("<f", float(end_mm)))
        if status != STS_ACK:
            send_command(self.serial_mgr.ser, CMD_STOP_SAMPLING)
            raise RuntimeError("Failed to start move to end.")

        # Stream length control: keep your model-based cap, but it is now ONLY a safety net.
        L_mm = abs(float(end_mm - start_mm))
        T_move_model = L_mm / max(v_mm_s, 1e-9)
        target_samples = int(T_move_model * fs)  # bigger margin than before

        try:
            while total_received < target_samples:
                if hasattr(self, "scan_running") and not self.scan_running:
                    break

                # Read one ADC packet
                _, pkt = read_packet(self.serial_mgr.ser, USB_SAMPLES_PER_PACKET)
                if pkt is None or pkt.size == 0:
                    continue

                buf = np.concatenate((buf, pkt))
                total_received += int(pkt.size)

                # Process as many full windows as available
                while buf.size >= window_N:
                    if hasattr(self, "scan_running") and not self.scan_running:
                        break

                    win = buf[:window_N].astype(np.float32)

                    # ADC -> volts, remove DC bias (Vref)
                    volts = win * (3.3 / 4096.0)
                    volts -= float(np.mean(volts))

                    # ---- Narrowband tone RMS at f_hz (quadrature, no FFT) ----
                    N = volts.size
                    n = np.arange(N, dtype=np.float32)
                    w = 2.0 * np.pi * float(f_hz) / float(fs)

                    c = np.cos(w * n)
                    s = np.sin(w * n)

                    I = float(np.mean(volts * c))
                    Q = float(np.mean(volts * s))

                    A = 2.0 * np.sqrt(I * I + Q * Q)   # sine amplitude estimate
                    rms = float(A / np.sqrt(2.0))      # amplitude -> RMS

                    # Global sample index of window midpoint
                    win_start_global = total_received - buf.size
                    win_mid_global = win_start_global + (window_N // 2)

                    # Discard startup transient windows
                    if win_mid_global >= skip_samples:
                        mid_samples.append(win_mid_global)
                        rms_vals.append(rms)

                    # Advance by hop (allows overlap)
                    buf = buf[hop_eff:]

        finally:
            # Stop streaming FIRST
            send_command(self.serial_mgr.ser, CMD_STOP_SAMPLING)
            

        # -----------------------
        # Measure *actual* end pos
        # -----------------------
        # (Framed command is OK again because streaming is stopped)
        #try:
            #wait_until_move_complete(self.serial_mgr.ser)
        #except Exception:
            #pass

        #p1 = self.get_position_mm()
        #if p1 is None:
        #    p1 = float(end_mm)
        #measured_end_mm = float(p1)

        if len(mid_samples) < 2:
            return np.array([], dtype=float), np.array([], dtype=float)

        # Map sample indices -> positions spanning [measured_start_mm, measured_end_mm]
        mids = np.asarray(mid_samples, dtype=float)
        y = np.asarray(rms_vals, dtype=float)

        n0 = float(mids[0])
        n1 = float(mids[-1])
        den = max(n1 - n0, 1.0)

        alpha = (mids - n0) / den
        #x = measured_start_mm + alpha * (measured_end_mm - measured_start_mm)
        x = start_mm + alpha * (end_mm - start_mm)

        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def _measure_metric(self, samples: np.ndarray, mode_id: int, f_hz: float) -> float:
        """
        mode_id: 0 = RMS, 1 = FFT
        Returns scalar metric for this position.
        """
        samples = np.asarray(samples, dtype=float)

        # Convert ADC codes to volts
        ADC_VREF = 3.3
        ADC_MAX = 4096.0
        volts = samples * (ADC_VREF / ADC_MAX)

        # Remove DC
        volts = volts - np.mean(volts)

        if mode_id == 0:
            # RMS of AC component
            return float(np.sqrt(np.mean(volts * volts)))

        # FFT magnitude at selected frequency
        mag, _ = fft_mag_at_freq(samples, self.FS, f_hz)
        return float(mag)
    
    def _extrema_and_reflection(self, positions: np.ndarray, mags: np.ndarray):
        positions = np.asarray(positions, dtype=float)
        mags = np.asarray(mags, dtype=float)

        if len(mags) < 3:
            raise RuntimeError("Too few points to find extrema.")

        mags_s = mags

        i_max = int(np.argmax(mags_s))
        i_min = int(np.argmin(mags_s))

        Pmax = float(mags_s[i_max])
        Pmin = float(mags_s[i_min])
        x_max = float(positions[i_max])
        x_min = float(positions[i_min])

        Pmin_safe = max(Pmin, 1e-12)
        SWR = Pmax / Pmin_safe
        R_mag = (SWR - 1.0) / (SWR + 1.0)

        return (Pmax, x_max, Pmin, x_min, SWR, R_mag, mags_s)

    def find_all_extrema(
    self,
    positions: np.ndarray,
    values: np.ndarray,
    min_prom_frac: float = 0.05,
    min_dx_mm: float = 5.0
):
        """
        Find ALL local maxima/minima with:
        - min_prom_frac: ignore tiny ripples (fraction of full range)
        - min_dx_mm: enforce minimum spacing between accepted extrema

        Returns:
        maxima: list[(x, y)]
        minima: list[(x, y)]
        """
        x = np.asarray(positions, dtype=float)
        y = np.asarray(values, dtype=float)

        if x.size < 3:
            return [], []

        # Ensure sorted by position
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        y_rng = max(y_max - y_min, 1e-12)

        prom_abs = float(min_prom_frac) * y_rng
        min_dx = float(min_dx_mm)

        # Candidate extrema by neighbor comparison
        cand_max = []
        cand_min = []
        for i in range(1, len(y) - 1):
            yl, ym, yr = y[i - 1], y[i], y[i + 1]

            # local maximum
            if ym > yl and ym > yr:
                # simple prominence proxy vs immediate neighbors
                prom = ym - max(yl, yr)
                if prom >= prom_abs:
                    cand_max.append(i)

            # local minimum
            if ym < yl and ym < yr:
                prom = min(yl, yr) - ym
                if prom >= prom_abs:
                    cand_min.append(i)

        def _enforce_spacing(idxs, prefer="high"):
            """Keep extrema separated by min_dx, preferring higher (max) or lower (min)."""
            if not idxs:
                return []

            # sort candidates by x
            idxs = sorted(idxs, key=lambda i: x[i])

            kept = []
            for i in idxs:
                if not kept:
                    kept.append(i)
                    continue

                if abs(x[i] - x[kept[-1]]) >= min_dx:
                    kept.append(i)
                else:
                    # too close: keep the "better" one
                    if prefer == "high":
                        if y[i] > y[kept[-1]]:
                            kept[-1] = i
                    else:
                        if y[i] < y[kept[-1]]:
                            kept[-1] = i

            return kept

        keep_max = _enforce_spacing(cand_max, prefer="high")
        keep_min = _enforce_spacing(cand_min, prefer="low")

        maxima = [(float(x[i]), float(y[i])) for i in keep_max]
        minima = [(float(x[i]), float(y[i])) for i in keep_min]
        return maxima, minima

    def _scan_range_metric(self, mode_id: int, f_hz: float, x0: float, x1: float, step_mm: float):
        if step_mm <= 0:
            raise ValueError("Step must be > 0.")
        if x1 < x0:
            x0, x1 = x1, x0

        # build positions including end
        n_steps = int(np.floor((x1 - x0) / step_mm)) + 1
        pos_list = [x0 + i * step_mm for i in range(max(1, n_steps))]
        if pos_list[-1] < x1 - 1e-9:
            pos_list.append(x1)

        positions = []
        metrics = []

        for pos in pos_list:
            self._move_abs_mm(pos)
            QTimer.singleShot(0, lambda: None)  # minimal yield (optional)
            send_command(self.serial_mgr.ser, CMD_START_SAMPLING)
            samples = read_adc_frame(self.serial_mgr.ser)
            send_command(self.serial_mgr.ser, CMD_STOP_SAMPLING)

            if samples is None:
                raise RuntimeError(f"ADC frame read failed at position {pos:.2f} mm.")

            metric = self._measure_metric(samples, mode_id, f_hz)
            positions.append(pos)
            metrics.append(metric)

        return np.array(positions, dtype=float), np.array(metrics, dtype=float)

    def _run_step_scan_metric(
        self,
        mode_id: int,
        f_hz: float,
        start_mm: float,
        end_mm: float,
        step_mm: float
    ):
        """
        Homes, moves to start, steps to end.
        Assumes CMD_STEPPER_MOVE is RELATIVE (as your UI 'Distance (mm)' implies).
        Returns (positions, metrics).
        """
        # Home
        status, _ = send_command(self.serial_mgr.ser, CMD_HOME)
        if status != STS_ACK:
            raise RuntimeError("Home failed to start.")
        if not wait_until_home_complete(self.serial_mgr.ser):
            raise RuntimeError("Homing timeout.")

        # Move to start (relative from home)
        status, _ = send_command(self.serial_mgr.ser, CMD_STEPPER_MOVE, struct.pack("<f", float(start_mm)))
        if status != STS_ACK:
            raise RuntimeError("Move to start failed.")
        if not wait_until_move_complete(self.serial_mgr.ser):
            raise RuntimeError("Move-to-start timeout.")

        # Set tone (optional for RMS, but harmless; required for FFT use-case)
        status, _ = send_command(self.serial_mgr.ser, CMD_AD9833_SINE_FREQ, struct.pack("<f", float(f_hz)))
        if status != STS_ACK:
            raise RuntimeError("Failed to set tone.")

        positions = []
        metrics = []

        # Build positions list to avoid floating drift
        if step_mm <= 0:
            raise ValueError("Step must be > 0.")
        n_steps = int(np.floor((end_mm - start_mm) / step_mm)) + 1
        pos_list = [start_mm + i * step_mm for i in range(max(1, n_steps))]
        if pos_list[-1] < end_mm - 1e-9:
            pos_list.append(end_mm)  # ensure end included

        # At start position already; measure then step relative
        for i, pos in enumerate(pos_list):
            # Allow mechanics to settle before sampling
            self._settle_after_move()
            # Capture one frame
            send_command(self.serial_mgr.ser, CMD_START_SAMPLING)
            samples = read_adc_frame(self.serial_mgr.ser)
            send_command(self.serial_mgr.ser, CMD_STOP_SAMPLING)

            if samples is None:
                raise RuntimeError(f"ADC frame read failed at position {pos:.2f} mm.")

            metric = self._measure_metric(samples, mode_id, f_hz)

            positions.append(pos)
            metrics.append(metric)

            # Move to next position (ABSOLUTE)
            if i < len(pos_list) - 1:
                next_pos = float(pos_list[i + 1])
                status, _ = send_command(self.serial_mgr.ser, CMD_STEPPER_MOVE, struct.pack("<f", next_pos))
                if status != STS_ACK:
                    raise RuntimeError("Step move failed.")
                if not wait_until_move_complete(self.serial_mgr.ser):
                    raise RuntimeError("Step move timeout.")


        # Mute after scan
        self._mute_speaker()

        return np.array(positions, dtype=float), np.array(metrics, dtype=float)

    def on_show_results(self):
        if not self.last_scan_results:
            QMessageBox.information(self, "No Results", "No scan results available yet.")
            return

        if not hasattr(self, "_result_dialog"):
            self._result_dialog = ScanResultDialog(self)

        self._result_dialog.update_results(self.last_scan_results)
        self._result_dialog.show()
        self._result_dialog.raise_()
        self._result_dialog.activateWindow()

    def _auto_pga_gain_from_freq(self, f_hz: float) -> str:
        """
        Map frequency to PGA113 gain (discrete set).
        Requirement: 100 Hz -> 1x, 2000 Hz -> 200x.
        Uses log-log interpolation and snaps to supported gains.
        """
        f_min = 100.0
        f_max = 2000.0
        g_min = 1.0
        g_max = 200.0

        f = max(f_min, min(float(f_hz), f_max))

        # log-log interpolation
        alpha = (np.log10(f) - np.log10(f_min)) / (np.log10(f_max) - np.log10(f_min))
        g_cont = g_min * (g_max / g_min) ** alpha

        gains = np.array([1, 2, 5, 10, 20, 50, 100, 200], dtype=float)
        g_sel = gains[np.argmin(np.abs(gains - g_cont))]

        return f"{int(g_sel)}x"
        
    def _mcu_set_pga_gain_code(self, gain_code: int, retries: int = 3) -> bool:
        """
        Robust PGA set:
        - ensure streaming is stopped
        - flush RX
        - retry a few times to survive framing desync
        """
        if not self.ensure_connected():
            return False

        for _ in range(max(1, int(retries))):
            try:
                # Ensure MCU isn't streaming ADC packets
                send_command(self.serial_mgr.ser, CMD_STOP_SAMPLING)

                # Flush stale bytes that can corrupt the next framed response
                self.serial_mgr.ser.reset_input_buffer()
                time.sleep(0.02)

                status, _ = send_command(self.serial_mgr.ser, CMD_PGA_SET, bytes([int(gain_code) & 0xFF]))
                if status == STS_ACK:
                    return True

            except Exception:
                # If parser throws because of garbage bytes, try again
                pass

            # small backoff
            time.sleep(0.05)

        return False

    def _auto_scan_end_from_freq(
        self,
        f_hz: float,
        start_mm: float,
        extrema_each: int,            # m = number of maxima AND number of minima
        margin_mm: float = 10.0,
        min_len_mm: float = 80.0
    ):
        """
        Choose end_mm so we scan enough length to observe approx:
        extrema_each maxima AND extrema_each minima.

        Adjacent extrema spacing ≈ λ/4, and a sequence of 2m extrema has (2m-1) gaps:
        L ≈ (2m-1) * (λ/4) + margin
        If limited by travel (scan_end_spin.maximum()), reduce m accordingly.

        Returns: (end_mm, used_extrema_each, lambda_mm)
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected")
        if getattr(self, "live_mode", False):
            raise RuntimeError("Stop Live View before auto end-mm (needs temperature read).")

        f = max(1.0, float(f_hz))
        m_req = max(1, int(extrema_each))

        # mandatory temperature from sensor (robust)
        T_c = float(self.mcu_get_temperature_c())

        # speed of sound and wavelength
        c = 331.3 + 0.606 * T_c          # m/s
        lam_mm = (c / f) * 1000.0        # mm

        max_end = float(self.scan_end_spin.maximum())

        # required length for requested m maxima AND m minima
        desired_len = (2.0 * m_req - 1.0) * (lam_mm / 4.0) + float(margin_mm)
        desired_len = max(desired_len, float(min_len_mm))
        end_mm = float(start_mm) + desired_len

        if end_mm > max_end:
            # available length from start to max end, minus margin
            avail_len = max(0.0, max_end - float(start_mm) - float(margin_mm))

            # Find max m that fits: (2m-1)*(λ/4) <= avail_len
            if lam_mm <= 1e-9:
                m_fit = 1
            else:
                m_fit = int(np.floor((2.0 * (avail_len / (lam_mm / 4.0)) + 1.0) / 2.0))
                m_fit = max(1, m_fit)

            m_use = min(m_req, m_fit)

            desired_len = (2.0 * m_use - 1.0) * (lam_mm / 4.0) + float(margin_mm)
            desired_len = max(desired_len, float(min_len_mm))
            end_mm = float(start_mm) + desired_len
            end_mm = min(end_mm, max_end)
            return end_mm, int(m_use), float(lam_mm)

        return min(end_mm, max_end), int(m_req), float(lam_mm)

    def on_run_scan(self):
        if not self.ensure_connected():
            return

        # Stop live view if running
        if self.live_mode:
            self.on_stop_live()

        # Flush any stale stream bytes
        self.serial_mgr.ser.reset_input_buffer()

        mode_id = self.scan_mode_group.checkedId()  # 0=RMS step, 2=FFT step, 1=RMS continuous fast
        if mode_id == 0:
            mode_name = "RMS"
        elif mode_id == 1:
            mode_name = "Fast"
        elif mode_id == 2:
            mode_name = "FFT"
        else:
            mode_name = "?"

        f_hz     = float(self.scan_freq_spin.value())
        start_mm = float(self.scan_start_spin.value())

        # -------------------------------------------------------
        # AUTO scan end based on extrema count (and travel limit)
        # -------------------------------------------------------
        requested_extrema = int(self.extrema_count_spin.value())
        try:
            end_mm, used_extrema, lam_mm = self._auto_scan_end_from_freq(
                f_hz=f_hz,
                start_mm=start_mm,
                extrema_each=requested_extrema,
                margin_mm=10.0,
                min_len_mm=80.0
            )
        except Exception as e:
            QMessageBox.warning(self, "Scan Error", str(e))
            return

        if used_extrema < requested_extrema:
            QMessageBox.warning(
                self,
                "Scan Range Limited",
                "Requested scan length exceeds motor travel limit.\n\n"
                f"Motor limit: {self.scan_end_spin.maximum():.0f} mm\n"
                f"Frequency: {f_hz:.1f} Hz   λ ≈ {lam_mm:.1f} mm\n\n"
                f"Reducing extrema target from {requested_extrema} to {used_extrema} "
                f"(≈ {used_extrema} maxima + {used_extrema} minima)."
            )
            self.extrema_count_spin.blockSignals(True)
            self.extrema_count_spin.setValue(used_extrema)
            self.extrema_count_spin.blockSignals(False)

        # Show chosen end-mm immediately
        self.scan_end_spin.blockSignals(True)
        self.scan_end_spin.setValue(end_mm)
        self.scan_end_spin.blockSignals(False)

        if mode_id == 1:
            try:
                fs = float(self.FS)
                if fs <= 0:
                    raise RuntimeError("Invalid FS (sampling rate).")

                if f_hz <= 0:
                    raise RuntimeError("Invalid excitation frequency.")

                phase_deg = float(self.rms_phase_deg_spin.value())  # NEW user control
                phase_deg = max(0.1, phase_deg)

                T_c = float(self.mcu_get_temperature_c())
                c_m_s = 331.3 + 0.606 * T_c

                # Motor speed(mm/s -> m/s)
                v_mm_s = abs(float(self._motor_velocity_mm_s()))
                v_m_s = v_mm_s / 1000.0
                if v_m_s <= 1e-9:
                    raise RuntimeError("Motor velocity is zero/invalid.")

                M = int(np.floor(fs * c_m_s *phase_deg / (360.0 * v_m_s * float(f_hz))))
                M = max(128, M)
                M = min(M, int(self.rms_win_spin.maximum()))

                # Update ONLY the existing RMS Window N spinbox
                self.rms_win_spin.blockSignals(True)
                self.rms_win_spin.setValue(M)
                self.rms_win_spin.blockSignals(False)

                #time window
                dt_ms = 1000.0 * (M / fs)
                if hasattr(self, "statusBar"):
                    try:
                        self.statusBar().showMessage(
                            f"Fast RMS window set: N={M} (dt≈{dt_ms:.1f} ms) @ T={T_c:.1f}°C, v={v_mm_s:.1f} mm/s",
                            5000
                        )
                    except Exception:
                        pass

            except Exception as e:
                QMessageBox.warning(self, "Scan Error", f"Failed to set RMS Window N: {e}")
                return

            QApplication.processEvents()


        coarse_mm = float(self.coarse_step_spin.value())
        fine_mm   = float(self.fine_step_spin.value())
        fine_win  = float(self.fine_window_spin.value())

        # --- AUTO PGA GAIN BASED ON FREQUENCY ---
        auto_gain = self._auto_pga_gain_from_freq(f_hz)

        gain_map = {
            "1x": 0, "2x": 1, "5x": 2, "10x": 3,
            "20x": 4, "50x": 5, "100x": 6, "200x": 7,
        }
        gain_code = gain_map[auto_gain]

        # Update GUI selector (visual feedback)
        self.pga_gain_combo.blockSignals(True)
        self.pga_gain_combo.setCurrentText(auto_gain)
        self.pga_gain_combo.blockSignals(False)

        ok = self._mcu_set_pga_gain_code(gain_code, retries=3)
        if not ok:
            QMessageBox.warning(
                self,
                "Auto Gain",
                f"Auto PGA gain set failed (wanted {auto_gain}).\nContinuing with current PGA gain."
            )

        # Allow analog chain to settle
        time.sleep(0.05)

        # --- Always home first (consistent reference) ---
        try:
            status, _ = send_command(self.serial_mgr.ser, CMD_HOME)
            if status != STS_ACK:
                raise RuntimeError("Home failed to start.")
            if not wait_until_home_complete(self.serial_mgr.ser):
                raise RuntimeError("Homing timeout.")
            time.sleep(0.5)

            # Set excitation tone
            status, _ = send_command(self.serial_mgr.ser, CMD_AD9833_SINE_FREQ, struct.pack("<f", float(f_hz)))
            if status != STS_ACK:
                raise RuntimeError("Failed to set tone.")
            time.sleep(0.5)

            # ============================================================
            # MODE 1: RMS Continuous (Fast)
            # ============================================================
            if mode_id == 1:
                self._move_abs_mm(start_mm)
                time.sleep(0.5)

                winN = int(self.rms_win_spin.value())
                hopN = int(self.rms_hop_spin.value())

                positions, metrics = self._rms_continuous_scan_fast(
                    start_mm=start_mm,
                    end_mm=end_mm,
                    window_N=winN,
                    hop_N=hopN
                )
                if not wait_until_move_complete(self.serial_mgr.ser):
                    raise RuntimeError("Move timeout in RMS continuous scan.")
                if positions.size < 2:
                    raise RuntimeError("Continuous RMS scan produced too few points.")

                # --- Basic global extrema (kept for now; we'll improve averaging later) ---
                Pmax, x_max, Pmin, x_min, SWR, R_mag, metrics_s = self._extrema_and_reflection(positions, metrics)

                # --- PHYSICALLY CORRECT extrema detection: spacing relative to wavelength ---
                # Use the temperature already measured earlier for the 1° rule if available,
                # otherwise fall back to a fresh read (only if NOT streaming now).
                try:
                    T_use = float(T_c)  # T_c exists in this mode_id==1 block (used for RMS window rule)
                except Exception:
                    T_use = 20.0

                # min_sep_frac_lambda controls how aggressively we reject non-standing-wave ripples.
                # Typical: 1/8 of lambda is a good compromise; 1/6 is stricter; 1/10 is looser.
                peak_idx, valley_idx = self.find_extrema_standing_wave(
                    positions,
                    metrics_s,
                    f_hz=f_hz,
                    T_c=T_use,
                    min_sep_frac_lambda=1/8,
                    include_endpoints=False
                )

                # Store extrema as lists of (x_mm, y) like your previous output format
                maxima = [(float(positions[i]), float(metrics_s[i])) for i in peak_idx]
                minima = [(float(positions[i]), float(metrics_s[i])) for i in valley_idx]

                self.last_scan_results = {
                    "mode": mode_name,
                    "Pmax": Pmax, "x_max": x_max,
                    "Pmin": Pmin, "x_min": x_min,
                    "SWR": SWR, "R": R_mag,
                    "maxima": maxima,
                    "minima": minima,
                }


                ax = self.kundt_canvas.ax
                ax.clear()
                ax.plot(positions, metrics_s, label="RMS Continuous")

                # Use the temperature we already read 
                T_use = float(T_c)
                peak_idx, valley_idx = self.find_extrema_standing_wave(
                    positions,
                    metrics_s,
                    f_hz=f_hz,
                    T_c=T_use,
                    min_sep_frac_lambda=1/8
                )

                if len(peak_idx) > 0:
                    ax.scatter(positions[peak_idx], metrics_s[peak_idx],
                            c="red", s=50, label=f"Maxima ({len(peak_idx)})")
                if len(valley_idx) > 0:
                    ax.scatter(positions[valley_idx], metrics_s[valley_idx],
                            c="blue", s=50, label=f"Minima ({len(valley_idx)})")

                ax.set_title("Continuous RMS Scan (Fast)")
                ax.set_xlabel("Position (mm)")
                ax.set_ylabel("RMS (V)")
                ax.grid(True)
                ax.legend()
                self.kundt_canvas.draw()
                return

            # ============================================================
            # MODE 0: RMS Step  |  MODE 2: FFT Step
            # ============================================================
            coarse_pos, coarse_y = self._scan_range_metric(mode_id, f_hz, start_mm, end_mm, coarse_mm)

            i_cmax = int(np.argmax(coarse_y))
            i_cmin = int(np.argmin(coarse_y))
            x_cmax = float(coarse_pos[i_cmax])
            x_cmin = float(coarse_pos[i_cmin])

            half = fine_win / 2.0
            max_x0 = max(start_mm, x_cmax - half)
            max_x1 = min(end_mm,   x_cmax + half)
            min_x0 = max(start_mm, x_cmin - half)
            min_x1 = min(end_mm,   x_cmin + half)

            fine_max_pos, fine_max_y = self._scan_range_metric(mode_id, f_hz, max_x0, max_x1, fine_mm)
            fine_min_pos, fine_min_y = self._scan_range_metric(mode_id, f_hz, min_x0, min_x1, fine_mm)

            i_fmax = int(np.argmax(fine_max_y))
            i_fmin = int(np.argmin(fine_min_y))

            Pmax = float(fine_max_y[i_fmax])
            x_max = float(fine_max_pos[i_fmax])
            Pmin = float(fine_min_y[i_fmin])
            x_min = float(fine_min_pos[i_fmin])

            Pmin_safe = max(Pmin, 1e-12)
            SWR = Pmax / Pmin_safe
            R_mag = (SWR - 1.0) / (SWR + 1.0)

            self.last_scan_results = {
                "mode": mode_name,
                "Pmax": Pmax,
                "x_max": x_max,
                "Pmin": Pmin,
                "x_min": x_min,
                "SWR": SWR,
                "R": R_mag,
            }

            ax = self.kundt_canvas.ax
            ax.clear()
            ax.plot(coarse_pos, coarse_y, "k--", label="Coarse")
            ax.plot(fine_max_pos, fine_max_y, "r-", label="Fine (around max)")
            ax.plot(fine_min_pos, fine_min_y, "b-", label="Fine (around min)")
            ax.scatter([x_max], [Pmax], c="red", s=80, label="Pmax")
            ax.scatter([x_min], [Pmin], c="blue", s=80, label="Pmin")
            ax.set_title(f"Two-Stage Step Scan ({mode_name})")
            ax.set_xlabel("Position (mm)")
            ax.set_ylabel("Metric")
            ax.grid(True)
            ax.legend()
            self.kundt_canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "Scan Error", str(e))
        finally:
            time.sleep(1)
            self._mute_speaker()

    def set_pga_gain(ser, gain_label: str):
        gain_code = GAIN_LABEL_TO_CODE[gain_label]
        send_command(ser, CMD_PGA_SET, bytes([gain_code]))


def _clamp(x, lo, hi):
    return max(lo, min(x, hi))

def build_scaled_qss(scale: float) -> str:
    # Scale helper with a sensible lower bound so text never becomes unreadable
    def px(v: int) -> int:
        return max(10, int(round(v * scale)))

    # If you want different baselines for 1080p vs 2K, change the base numbers below.
    return f"""
    QWidget {{
        font-size: {px(18)}px;
    }}

    QGroupBox {{
        font-size: {px(18)}px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: {px(10)}px;
        padding: 0 {px(6)}px;
    }}

    QPushButton {{
        font-size: {px(20)}px;
        padding: {px(10)}px {px(14)}px;
        min-height: {px(38)}px;
    }}

    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
        font-size: {px(20)}px;
        min-height: {px(38)}px;
        padding: {px(6)}px {px(10)}px;
    }}

    QTabWidget::pane {{
        border-top: 1px solid palette(mid);
    }}

    QTabBar::tab {{
        font-size: {px(18)}px;
        padding: {px(8)}px {px(16)}px;
        min-height: {px(34)}px;
    }}

    QLabel {{
        font-size: {px(18)}px;
    }}
    """

def apply_ui_scaling(app: QApplication, baseline_height: int = 1440) -> None:
    # Use availableGeometry to respect taskbar / dock
    screen = app.primaryScreen().availableGeometry()

    # Scaling by height is usually more stable across monitors than width
    scale = screen.height() / float(baseline_height)

    # Clamp: prevent extremes on very small or huge screens
    # 1080p -> 0.75 when baseline is 1440
    scale = _clamp(scale, 0.70, 1.25)

    app.setStyleSheet(build_scaled_qss(scale))

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)

    apply_ui_scaling(app, baseline_height=1440)  # baseline = your 2K/1440p look

    win = MainWindow()
    win.setMinimumSize(800, 600)
    # Size main window relative to screen
    screen = app.primaryScreen().availableGeometry()

    w = int(screen.width() * 0.90)
    h = int(screen.height() * 0.90)  # use 90% height instead of 50%

    # Clamp to available screen size (prevents Qt setGeometry warnings)
    w = min(w, screen.width())
    h = min(h, screen.height())

    win.resize(w, h)


    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
