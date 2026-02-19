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
        The data processing happens during the function, not after scanning
        Return an array of x and its y values.
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

        # --- Frequency and motion parameters  ---
        f_hz = float(self.scan_freq_spin.value())
        if f_hz <= 0:
            raise RuntimeError("Invalid excitation frequency.")

        v_mm_s = abs(float(self._motor_velocity_mm_s())) # Berechnung der Geschwindigkeit
        if v_mm_s < 1e-6:
            v_mm_s = 20.6  # fallback

        # Temperaturmessung
        T_c = float(self.mcu_get_temperature_c())
        self._last_scan_temp_c = float(T_c)
        c_m_s = 331.3 + 0.606 * T_c #Schallgeschwindikeit

        # 1° Ortauflösung
        # dx = λ/360, dt_max = dx/v, M_max = fs*dt_max
        v_m_s = v_mm_s / 1000.0
        lam_m = c_m_s / float(f_hz)
        dt_max = lam_m / (360.0 * max(v_m_s, 1e-9))
        M_max = int(np.floor(fs * dt_max))

        # Maximale Fensterbreit (minimum 128)
        window_N = max(128, min(user_window_N, max(128, M_max)))

        # dieser Wert wird nicht benutzt, Artefakt aus alter Software
        hop_eff = int(user_hop_N)
        if hop_eff <= 0 or hop_eff > window_N:
            hop_eff = window_N

        # Sample überspringen wegen Einschwingsverhalten
        skip_samples = int(round((float(skip_ms) / 1000.0) * fs))
        if skip_samples < 0:
            skip_samples = 0

        # -------------------------nicht mehr verwendet
        # Measure *actual* start pos, 
        # -------------------------
        # (Framed command is OK here because we're NOT streaming yet)
        #p0 = self.get_position_mm()
        #if p0 is None:
        #    p0 = float(start_mm)
        #measured_start_mm = float(p0)

        buf = np.empty(0, dtype=np.uint16)
        total_received = 0  # total ADC samples received since start of streaming

        mid_samples = []   # window midpoint sample indices
        rms_vals = []

        # Flush packets, start streaming
        self.serial_mgr.ser.reset_input_buffer()
        status, _ = send_command(self.serial_mgr.ser, CMD_START_SAMPLING) #start-Kommando an Mikrocontroller
        if status != STS_ACK:
            raise RuntimeError("Failed to start sampling.")

        # Vom Anfang bis Ende bewegen
        status, _ = send_command(self.serial_mgr.ser, CMD_STEPPER_MOVE, struct.pack("<f", float(end_mm)))
        if status != STS_ACK:
            send_command(self.serial_mgr.ser, CMD_STOP_SAMPLING)
            raise RuntimeError("Failed to start move to end.")

        # Datenpaket-Breite Kontrolle
        L_mm = abs(float(end_mm - start_mm))
        T_move_model = L_mm / max(v_mm_s, 1e-9)
        target_samples = int(T_move_model * fs)

        try:
            while total_received < target_samples: # Schleife, bis die berechnete Datenmenge erreicht wird
                if hasattr(self, "scan_running") and not self.scan_running:
                    break

                # Read one ADC packet
                _, pkt = read_packet(self.serial_mgr.ser, USB_SAMPLES_PER_PACKET)
                if pkt is None or pkt.size == 0:
                    continue

                buf = np.concatenate((buf, pkt)) #Datenpaket abspeichern
                total_received += int(pkt.size)

                # Process as many full windows as available
                while buf.size >= window_N:
                    if hasattr(self, "scan_running") and not self.scan_running:
                        break

                    win = buf[:window_N].astype(np.float32)

                    # ADC -> Spannung
                    volts = win * (3.3 / 4096.0)
                    volts -= float(np.mean(volts))
                    """ IQ Methode für eine bessere Amplitude-Detektion, 
                        aber ich benutze lieber Pure-RMS damit der Vergleich zwischen RMS und FFT deutlicher wird
                    
                    N = volts.size
                    n = np.arange(N, dtype=np.float32)
                    w = 2.0 * np.pi * float(f_hz) / float(fs)

                    c = np.cos(w * n)
                    s = np.sin(w * n)

                    I = float(np.mean(volts * c))
                    Q = float(np.mean(volts * s))

                    A = 2.0 * np.sqrt(I * I + Q * Q)   # sine amplitude estimate
                    rms = float(A / np.sqrt(2.0))      # amplitude -> RMS
                    """
                    rms = float(np.sqrt(np.mean(volts * volts))) # RMS berechnung

                    # 
                    win_start_global = total_received - buf.size
                    win_mid_global = win_start_global + (window_N // 2)

                    # einige Samples überspringen
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

        # Index von Sample und seinen Wert zuweisen
        mids = np.asarray(mid_samples, dtype=float)
        y = np.asarray(rms_vals, dtype=float)

        n0 = float(mids[0])
        n1 = float(mids[-1])
        den = max(n1 - n0, 1.0)

        alpha = (mids - n0) / den
        #x = measured_start_mm + alpha * (measured_end_mm - measured_start_mm)
        x = start_mm + alpha * (end_mm - start_mm)

        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)