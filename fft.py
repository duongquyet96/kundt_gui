def _measure_metric(self, samples: np.ndarray, mode_id: int, f_hz: float) -> float:
        """
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

def _scan_range_metric(self, mode_id: int, f_hz: float, x0: float, x1: float, step_mm: float):
        """
        Beweg den Wagen vom Start zum Endpunkt
        Nach der Bewegung wird eine Sample aufgenommen
        Von dieser Sample wird eine FFT durchgeführt
        die Rückgabe ist ein Array von Indizien und deren Werte 
        """
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
        Standing-wave extrema finder:

        - Sorts by x.
        - Lightly smooths y .
        - Finds extrema via sign changes.
        - Refines each extrema by local search on the ORIGINAL y.
        - Enforces alternation and minimum spacing.
        - Returns indices into original arrays.
        """
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



def _run_scan_fft_step(self, mode_name: str, f_hz: float, start_mm: float, end_mm: float):
        """
        Mode 1: Große FFT step scan dann Fine-scans und extrapolieren.
        """
        try:
            """
            vorbereitung aller Variablen für die Funktion
            """
            self._prepare_scan_common(f_hz)

            coarse_mm = float(self.coarse_step_spin.value())
            fine_mm   = float(self.fine_step_spin.value())
            fine_win  = float(self.fine_window_spin.value())
            half = fine_win / 2.0

            FFT_MODE_ID = 2

            try:
                T_c = float(self.mcu_get_temperature_c())
            except Exception:
                T_c = 20.0

            # 1) Grober Scan mit _scan_range_metric
            coarse_pos, coarse_y = self._scan_range_metric(FFT_MODE_ID, f_hz, start_mm, end_mm, coarse_mm)
            if len(coarse_pos) < 3:
                raise RuntimeError("FFT coarse scan returned too few points.")

            # 2) Alle Extrema herausfinden
            peak_idx, valley_idx = self.find_extrema_standing_wave(
                coarse_pos,
                coarse_y,
                f_hz=f_hz,
                T_c=T_c,
                min_sep_frac_lambda=1/8,
                include_endpoints=False
            )

            if (len(peak_idx) + len(valley_idx)) == 0:
                i_cmax = int(np.argmax(coarse_y))
                i_cmin = int(np.argmin(coarse_y))
                peak_idx = np.array([i_cmax], dtype=int)
                valley_idx = np.array([i_cmin], dtype=int)

            refined_maxima = []
            refined_minima = []
            fine_traces = []

            # Helper: refine index j within (x_f, y_f)
            def refine_at_index(x_f, y_f, j):
                pt = self._refine_extrema_quadratic(x_f, y_f, [int(j)])[0]
                return (float(pt[0]), float(pt[1]))

            # 3) Feiner Scan aller Maximum
            for i in peak_idx:
                x0 = float(coarse_pos[i])
                a = max(start_mm, x0 - half)
                b = min(end_mm,   x0 + half)
                if b - a < max(fine_mm * 2.0, 1e-6):
                    continue

                x_f, y_f = self._scan_range_metric(FFT_MODE_ID, f_hz, a, b, fine_mm)
                if len(x_f) < 3:
                    continue

                j = int(np.argmax(y_f))
                refined_maxima.append(refine_at_index(x_f, y_f, j))
                fine_traces.append((x_f, y_f, f"Fine max @~{x0:.1f}mm"))

            # 4) Feiner Scan aller Minimum
            for i in valley_idx:
                x0 = float(coarse_pos[i])
                a = max(start_mm, x0 - half)
                b = min(end_mm,   x0 + half)
                if b - a < max(fine_mm * 2.0, 1e-6):
                    continue

                x_f, y_f = self._scan_range_metric(FFT_MODE_ID, f_hz, a, b, fine_mm)
                if len(x_f) < 3:
                    continue

                j = int(np.argmin(y_f))
                refined_minima.append(refine_at_index(x_f, y_f, j))
                fine_traces.append((x_f, y_f, f"Fine min @~{x0:.1f}mm"))

            if (len(refined_maxima) + len(refined_minima)) == 0:
                raise RuntimeError("FFT fine scan produced no refined extrema (check fine_win / steps).")

            # 5) Global extrema from refined sets
            if refined_maxima:
                x_max, Pmax = max(refined_maxima, key=lambda t: t[1])
            else:
                i = int(np.argmax(coarse_y))
                x_max, Pmax = float(coarse_pos[i]), float(coarse_y[i])

            if refined_minima:
                x_min, Pmin = min(refined_minima, key=lambda t: t[1])
            else:
                i = int(np.argmin(coarse_y))
                x_min, Pmin = float(coarse_pos[i]), float(coarse_y[i])

            Pmin_safe = max(float(Pmin), 1e-12)
            SWR = float(Pmax) / Pmin_safe
            R_mag = (SWR - 1.0) / (SWR + 1.0)

            self.last_scan_results = {
                "mode": mode_name,
                "Pmax": float(Pmax), "x_max": float(x_max),
                "Pmin": float(Pmin), "x_min": float(x_min),
                "SWR": float(SWR), "R": float(R_mag),
                "maxima": [(float(x), float(y)) for (x, y) in refined_maxima],
                "minima": [(float(x), float(y)) for (x, y) in refined_minima],
            }

            # 6) Plot
            ax = self.kundt_canvas.ax
            ax.clear()
            ax.plot(coarse_pos, coarse_y, "k--", label="Coarse (FFT metric)")

            for (x_f, y_f, lbl) in fine_traces:
                ax.plot(x_f, y_f, "-", linewidth=1.0, alpha=0.7, label=lbl)

            if refined_maxima:
                xm = np.array([p[0] for p in refined_maxima], dtype=float)
                ym = np.array([p[1] for p in refined_maxima], dtype=float)
                ax.scatter(xm, ym, c="red", s=50, label=f"Refined maxima ({len(refined_maxima)})")

            if refined_minima:
                xn = np.array([p[0] for p in refined_minima], dtype=float)
                yn = np.array([p[1] for p in refined_minima], dtype=float)
                ax.scatter(xn, yn, c="blue", s=50, label=f"Refined minima ({len(refined_minima)})")

            ax.scatter([x_max], [Pmax], c="red", s=120, marker="*", label="Global Pmax")
            ax.scatter([x_min], [Pmin], c="blue", s=120, marker="*", label="Global Pmin")

            ax.set_title("FFT Scan: Coarse + Fine (all extrema) + refined peaks")
            ax.set_xlabel("Position (mm)")
            ax.set_ylabel("Metric (FFT)")
            ax.grid(True)

            # Legend dedup
            handles, labels = ax.get_legend_handles_labels()
            seen = set()
            h2, l2 = [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    h2.append(h)
                    l2.append(l)
            ax.legend(h2, l2)

            self.kundt_canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "Scan Error", str(e))
        finally:
            time.sleep(1)
            self._mute_speaker()