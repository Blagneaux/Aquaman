#!/usr/bin/env python3
"""
UDP Receiver — Interface Pro (complet)
- Map de pression realtime (Canvas)
- Paramètres utilisateur : FPS, vmin, vmax (Apply / Reset)
- Filtre passe-haut par capteur (O(1) ring buffer) + option EMA (IIR)
- Écriture CSV, calcul pertes (timestamps invalides + gaps)
- + TriggerBox: ajout telemetrie CNC (X/Y/Z/A/M) depuis l'ESP, voyant UI vert/rouge
"""

import socket
import struct
import csv
import time
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import ttkbootstrap as ttkb
from ttkbootstrap import Style
import pandas as pd
import os
import bisect
from queue import Queue, Empty
from collections import deque
import numpy as np
import math

# =====================
# CONFIG TRIGGERBOX
# =====================
TRIG_PORT = 12345
CNC_ESP_IP = "192.168.0.112"
CSV_DELIMITER = ";"
CSV_SWAP_CNC_YZ = False
APP_VERSION = "Sensors Network v2.1-hold-missing-esp"
CNC_CSV_MAX_ALIGN_DELTA_US = 100_000
CNC_CSV_FALLBACK_TO_LATEST = True
CNC_HISTORY_MAX_SAMPLES = 20_000

# struct CncSample (28 octets, little-endian)
# uint64_t timestamp_sync_us;  # Q
# float    x;                  # f
# float    y;                  # f
# float    z;                  # f
# float    a;                  # f
# uint8_t  motion_state;       # B
# uint8_t  reserved[3];        # 3x
CNC_SAMPLE_FMT = "<Q f f f f B 3x"
CNC_SAMPLE_SIZE = struct.calcsize(CNC_SAMPLE_FMT)  # 28


def format_csv_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


# =====================
# TELEMETRIE CNC
# =====================
class CncTelemetryState:
    def __init__(self):
        self._lock = threading.Lock()
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.a = 0.0
        self.motion_state = 0
        self.valid = False
        self._last_rx_time = 0.0
        self._history_ts = []
        self._history_samples = []

    def update(self, timestamp_sync_us: int, x: float, y: float, z: float, a: float, motion_state: int):
        with self._lock:
            ts = int(timestamp_sync_us)
            sample = (float(x), float(y), float(z), float(a), 1 if motion_state else 0)
            self.x, self.y, self.z, self.a, self.motion_state = sample
            self.motion_state = 1 if motion_state else 0
            self.valid = True
            self._last_rx_time = time.time()

            if ts > 0:
                if not self._history_ts or ts >= self._history_ts[-1]:
                    self._history_ts.append(ts)
                    self._history_samples.append(sample)
                else:
                    idx = bisect.bisect_left(self._history_ts, ts)
                    self._history_ts.insert(idx, ts)
                    self._history_samples.insert(idx, sample)

                overflow = len(self._history_ts) - CNC_HISTORY_MAX_SAMPLES
                if overflow > 0:
                    del self._history_ts[:overflow]
                    del self._history_samples[:overflow]

    def get(self):
        with self._lock:
            return self.x, self.y, self.z, self.a, self.motion_state, self.valid

    def get_at(self, timestamp_sync_us: int, max_delta_us: int = CNC_CSV_MAX_ALIGN_DELTA_US):
        with self._lock:
            if not self._history_ts:
                return 0.0, 0.0, 0.0, 0.0, 0, False, None

            ts = int(timestamp_sync_us)
            idx = bisect.bisect_left(self._history_ts, ts)
            candidates = []
            if idx < len(self._history_ts):
                candidates.append(idx)
            if idx > 0:
                candidates.append(idx - 1)
            if not candidates:
                return 0.0, 0.0, 0.0, 0.0, 0, False, None

            best_idx = min(candidates, key=lambda i: abs(self._history_ts[i] - ts))
            delta_us = int(self._history_ts[best_idx] - ts)
            if abs(delta_us) > int(max_delta_us):
                return 0.0, 0.0, 0.0, 0.0, 0, False, delta_us

            x, y, z, a, motion_state = self._history_samples[best_idx]
            return x, y, z, a, motion_state, True, delta_us

    def maybe_timeout_to_idle(self, timeout_s=3.0):
        with self._lock:
            now = time.time()
            if self.valid and self.motion_state == 1 and self._last_rx_time and (now - self._last_rx_time > timeout_s):
                self.motion_state = 0
                return True
        return False


CNC = CncTelemetryState()


class CncReceiverThread(threading.Thread):
    def __init__(self, log_callback, ui_queue=None, stop_event=None):
        super().__init__(daemon=True)
        self.ui_queue = ui_queue
        self.log_callback = log_callback
        self.sock = None
        self.stop_event = stop_event or threading.Event()

    def _post_cnc_state(self, timestamp_sync_us, x, y, z, a, motion_state, valid):
        if not self.ui_queue:
            return
        try:
            self.ui_queue.put_nowait({
                "type": "cnc_state",
                "timestamp_sync_us": int(timestamp_sync_us),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "a": float(a),
                "motion_state": int(motion_state),
                "valid": bool(valid),
            })
        except Exception:
            pass

    def run(self):
        try:
            self.log_callback(f"[CNC] Ecoute sur UDP {TRIG_PORT}, source attendue={CNC_ESP_IP}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("0.0.0.0", TRIG_PORT))
            self.sock.settimeout(1.0)

            while not self.stop_event.is_set():
                if CNC.maybe_timeout_to_idle(timeout_s=3.0):
                    x, y, z, a, motion_state, valid = CNC.get()
                    self._post_cnc_state(0, x, y, z, a, motion_state, valid)

                try:
                    data, addr = self.sock.recvfrom(8192)
                except socket.timeout:
                    continue
                except Exception as e:
                    self.log_callback(f"[CNC] Erreur recv: {e}")
                    continue

                src_ip, _ = addr
                if src_ip != CNC_ESP_IP:
                    continue

                packet_size = len(data)
                if packet_size < CNC_SAMPLE_SIZE or (packet_size % CNC_SAMPLE_SIZE) != 0:
                    self.log_callback(f"[CNC] Paquet invalide {packet_size} octets depuis {src_ip}")
                    continue

                last_sample = None
                for offset in range(0, packet_size, CNC_SAMPLE_SIZE):
                    try:
                        ts_sync_us, x, y, z, a, motion_state = struct.unpack_from(CNC_SAMPLE_FMT, data, offset)
                    except Exception as e:
                        self.log_callback(f"[CNC] Unpack error: {e}")
                        last_sample = None
                        break

                    CNC.update(ts_sync_us, x, y, z, a, int(motion_state))
                    last_sample = (ts_sync_us, x, y, z, a, motion_state)

                if last_sample:
                    ts_sync_us, x, y, z, a, motion_state = last_sample
                    self._post_cnc_state(ts_sync_us, x, y, z, a, motion_state, True)

        except Exception as e:
            self.log_callback(f"[CNC] Thread error: {e}")
        finally:
            try:
                if self.sock:
                    self.sock.close()
            except Exception:
                pass
            self.log_callback("[CNC] Thread arrete.")


# =====================
# UDP Receiver Thread (capteurs)
# =====================
class UDPReceiverThread(threading.Thread):
    """
    Thread qui écoute les paquets UDP capteurs, décode et écrit dans un CSV,
    et poste des mises à jour dans ui_queue pour affichage.
    """
    BYTES_PER_SAMPLE = 24  # Q (8) + 8*h (2 bytes each) => 8 + 16 = 24 bytes
    SAMPLE_STRUCT = struct.Struct("=Q8h")
    SOCKET_TIMEOUT_S = 0.20
    SOCKET_RCVBUF_BYTES = 8 * 1024 * 1024
    STATS_UPDATE_INTERVAL_S = 0.25
    GAP_STATS_INTERVAL_S = 0.50
    CSV_FLUSH_INTERVAL_S = 1.00
    UI_EPOCH_UPDATE_INTERVAL_S = 0.10
    PARTIAL_EPOCH_TIMEOUT_S = 0.75

    def __init__(self, ip, port, sample_period, expected_devices, file_path, log_callback, ui_queue=None):
        super().__init__(daemon=True)
        self.ip = ip
        self.port = port
        self.sample_period = int(sample_period)
        self.expected_devices = max(1, int(expected_devices))
        self.file_path = file_path
        self.log_callback = log_callback
        self.ui_queue = ui_queue

        self.stop_event = threading.Event()
        self.epochs = {}  # epoch -> {"slots": {slot: sensor_values}, "created_at": time.time()}
        self.sock = None
        self.csv_file = None
        self.csv_writer = None

        self.invalid_timestamps = 0
        self.valid_samples_received = 0
        self.rows_written = 0
        self.complete_rows_written = 0
        self.partial_rows_written = 0
        self.held_rows_written = 0
        self.held_slots_reused = 0
        self.blank_slots_written = 0
        self.packets_malformed = 0
        self.extra_devices_dropped = 0
        self.gap_missing_lines = 0
        self.packets_received = 0

        self.start_time = None
        self.last_stats_post = 0.0
        self.last_gap_post = 0.0
        self.last_flush = 0.0
        self.last_ui_epoch_post = 0.0
        self.last_written_timestamp = None
        self.last_finalized_epoch = -1
        self.recent_packets = deque(maxlen=300)
        self.cnc_csv_debug_lines = 0
        self.device_slots = {}
        self.device_order = []
        self.extra_device_ips = set()
        self.last_sensor_values = [None] * self.expected_devices
        self.header_written = False

    def run(self):
        try:
            self.log_callback(f"Ouverture du socket UDP sur {self.ip}:{self.port}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.SOCKET_RCVBUF_BYTES)
            self.sock.bind((self.ip, self.port))
            self.sock.settimeout(self.SOCKET_TIMEOUT_S)

            folder = os.path.dirname(self.file_path)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

            self.csv_file = open(self.file_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file, delimiter=CSV_DELIMITER)
            self._write_header()

            self.log_callback(f"Enregistrement UDP activé : {self.file_path}")
            actual_rcvbuf = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            self.log_callback(f"[UDP] Buffer socket RX: {actual_rcvbuf} octets")
            if "OneDrive" in os.path.abspath(self.file_path):
                self.log_callback("[UDP] Attention: ecriture dans OneDrive, cela peut augmenter les pertes.")
            self.log_callback(
                "[CNC] Mapping CSV: CNC_X=X, "
                f"CNC_Y={'Z' if CSV_SWAP_CNC_YZ else 'Y'}, "
                f"CNC_Z={'Y' if CSV_SWAP_CNC_YZ else 'Z'}, CNC_A=A, "
                f"align<={CNC_CSV_MAX_ALIGN_DELTA_US}us, "
                f"fallback={'latest' if CNC_CSV_FALLBACK_TO_LATEST else 'none'}"
            )
            now = time.time()
            self.start_time = now
            self.last_stats_post = now
            self.last_gap_post = now
            self.last_flush = now

            while not self.stop_event.is_set():
                try:
                    data, addr = self.sock.recvfrom(65535)
                except socket.timeout:
                    now = time.time()
                    self._expire_old_epochs(now, force=False)
                    self._flush_csv_if_needed(now, force=False)
                    self._post_runtime_stats(now, force=False)
                    self._post_gap_stats(now, force=False)
                    continue
                except Exception as e:
                    self.log_callback(f"Erreur socket.recvfrom : {e}")
                    continue

                self.packets_received += 1
                src_ip = addr[0]
                data_len = len(data)

                if data_len == 0:
                    continue

                if data_len % self.BYTES_PER_SAMPLE != 0:
                    self.packets_malformed += 1
                    self.log_callback(f"⚠ Paquet mal formé ({data_len} octets) de {src_ip}")
                    self._post_packet_info(src_ip, data_len, 0, None, malformed=True)
                    continue

                samples = data_len // self.BYTES_PER_SAMPLE
                valid_samples = 0
                first_ts = None
                slot = self._ensure_device_slot(src_ip)
                if slot is None:
                    self._post_packet_info(src_ip, data_len, samples, None, malformed=False)
                    continue

                try:
                    for sample in struct.iter_unpack(self.SAMPLE_STRUCT.format, data):
                        timestamp = sample[0]
                        if first_ts is None:
                            first_ts = timestamp

                        if timestamp < 1_000_000:
                            self.invalid_timestamps += 1
                            continue

                        valid_samples += 1
                        self.valid_samples_received += 1
                        sensor_values = sample[1:9]
                        epoch = (timestamp // self.sample_period) * self.sample_period

                        if epoch <= self.last_finalized_epoch:
                            continue

                        epoch_state = self.epochs.get(epoch)
                        if epoch_state is None:
                            epoch_state = {"slots": {}, "created_at": time.time()}
                            self.epochs[epoch] = epoch_state

                        epoch_state["slots"][slot] = sensor_values
                except Exception as e:
                    self.packets_malformed += 1
                    self.log_callback(f"Erreur unpack ({src_ip}): {e}")
                    self._post_packet_info(src_ip, data_len, 0, first_ts, malformed=True)
                    continue

                self.recent_packets.appendleft({
                    "time": time.time(),
                    "src": src_ip,
                    "len": data_len,
                    "samples": samples,
                    "valid": valid_samples,
                    "first_ts": first_ts
                })
                self._post_packet_info(src_ip, data_len, samples, first_ts, malformed=False)

                now = time.time()
                self._expire_old_epochs(now, force=False)
                self._flush_csv_if_needed(now, force=False)
                self._post_runtime_stats(now, force=False)
                self._post_gap_stats(now, force=False)

        except Exception as e:
            self.log_callback(f"Erreur dans le thread UDP : {e}")

        finally:
            now = time.time()
            self._expire_old_epochs(now, force=True)
            self._flush_csv_if_needed(now, force=True)
            self._post_runtime_stats(now, force=True)
            self._post_gap_stats(now, force=True, final=True)

            if self.csv_file:
                self.csv_file.close()
            if self.sock:
                self.sock.close()

            duration = (time.time() - self.start_time) if self.start_time else 0
            self.log_callback("Thread UDP arrêté.")
            self.log_callback(f"Durée totale d'échantillonnage : {duration:.2f} secondes")

            total_expected_lines = self.rows_written + self.gap_missing_lines
            denom = total_expected_lines + self.invalid_timestamps
            loss_percent = (
                100.0 * (self.invalid_timestamps + self.gap_missing_lines) / denom
            ) if denom else 0.0
            try:
                if self.ui_queue:
                    self.ui_queue.put_nowait({
                        "type": "final_stats",
                        "total_lines": self.rows_written,
                        "lost_lines": self.invalid_timestamps,
                        "loss_percent": loss_percent,
                        "packets_received": self.packets_received,
                        "partial_rows": self.partial_rows_written,
                        "held_rows": self.held_rows_written,
                        "held_slots": self.held_slots_reused,
                        "blank_slots": self.blank_slots_written,
                        "complete_rows": self.complete_rows_written,
                        "gap_lost_lines": self.gap_missing_lines,
                    })
            except Exception:
                pass

    def _write_header(self):
        if self.header_written:
            return
        header = ["timestamp"]
        for idx in range(self.expected_devices):
            header.extend([f"ESP{idx+1}_Sensor{i+1}" for i in range(8)])
        header += ["CNC_X", "CNC_Y", "CNC_Z", "CNC_A", "CNC_Moving"]
        self.csv_writer.writerow(header)
        self.header_written = True

    def _ensure_device_slot(self, src_ip):
        if src_ip in self.device_slots:
            return self.device_slots[src_ip]

        if len(self.device_order) >= self.expected_devices:
            if src_ip not in self.extra_device_ips:
                self.extra_device_ips.add(src_ip)
                self.extra_devices_dropped += 1
                self.log_callback(f"[UDP] Source supplÃ©mentaire ignorÃ©e: {src_ip}")
            return None

        slot = len(self.device_order)
        self.device_slots[src_ip] = slot
        self.device_order.append(src_ip)
        self.log_callback(f"[UDP] Module dÃ©tectÃ©: slot ESP{slot+1} <- {src_ip}")
        return slot

    def _append_cnc_columns(self, row, common_timestamp):
        cnc_x, cnc_y, cnc_z, cnc_a, cnc_moving, cnc_valid, cnc_delta_us = CNC.get_at(common_timestamp)
        cnc_source = "aligned"
        if not cnc_valid and CNC_CSV_FALLBACK_TO_LATEST:
            cnc_x, cnc_y, cnc_z, cnc_a, cnc_moving, cnc_valid = CNC.get()
            cnc_source = "latest"
            cnc_delta_us = None

        if cnc_valid:
            csv_cnc_y, csv_cnc_z = (cnc_z, cnc_y) if CSV_SWAP_CNC_YZ else (cnc_y, cnc_z)
            if self.cnc_csv_debug_lines < 5:
                self.log_callback(
                    "[CNC CSV] raw "
                    f"X={cnc_x:.3f}, Y={cnc_y:.3f}, Z={cnc_z:.3f}, A={cnc_a:.3f} -> "
                    f"CSV_X={cnc_x:.3f}, CSV_Y={csv_cnc_y:.3f}, "
                    f"CSV_Z={csv_cnc_z:.3f}, CSV_A={cnc_a:.3f}, "
                    f"source={cnc_source}, align_delta_us={cnc_delta_us}"
                )
                self.cnc_csv_debug_lines += 1
            row += [
                format_csv_float(cnc_x),
                format_csv_float(csv_cnc_y),
                format_csv_float(csv_cnc_z),
                format_csv_float(cnc_a),
                int(cnc_moving),
            ]
        else:
            if self.cnc_csv_debug_lines < 5:
                self.log_callback(
                    f"[CNC CSV] aucun echantillon CNC aligne pour timestamp={common_timestamp} "
                    f"(delta_us={cnc_delta_us})"
                )
                self.cnc_csv_debug_lines += 1
            row += ["", "", "", "", 0]

    def _write_epoch(self, epoch, epoch_state, complete):
        common_timestamp = int(epoch + self.sample_period // 2)
        row = [common_timestamp]
        pressures = np.full(self.expected_devices * 8, np.nan, dtype=float)
        slots = epoch_state["slots"]
        held_in_row = 0
        blank_in_row = 0

        for slot in range(self.expected_devices):
            sensor_values = slots.get(slot)
            base = slot * 8
            if sensor_values is None:
                sensor_values = self.last_sensor_values[slot]
                if sensor_values is None:
                    row.extend([""] * 8)
                    blank_in_row += 1
                    continue
                held_in_row += 1
                row.extend(sensor_values)
                pressures[base:base + 8] = np.asarray(sensor_values, dtype=float)
                continue
            row.extend(sensor_values)
            pressures[base:base + 8] = np.asarray(sensor_values, dtype=float)
            self.last_sensor_values[slot] = tuple(sensor_values)

        self._append_cnc_columns(row, common_timestamp)
        self.csv_writer.writerow(row)

        self.rows_written += 1
        if complete:
            self.complete_rows_written += 1
        else:
            self.partial_rows_written += 1
        if held_in_row > 0:
            self.held_rows_written += 1
            self.held_slots_reused += held_in_row
        if blank_in_row > 0:
            self.blank_slots_written += blank_in_row

        if self.last_written_timestamp is not None:
            gap = common_timestamp - self.last_written_timestamp
            if gap > self.sample_period:
                if gap % self.sample_period == 0:
                    self.gap_missing_lines += max(0, (gap // self.sample_period) - 1)
                else:
                    self.gap_missing_lines += max(0, int(round(gap / self.sample_period)) - 1)
        self.last_written_timestamp = common_timestamp
        self.last_finalized_epoch = max(self.last_finalized_epoch, epoch)

        now = time.time()
        if self.ui_queue and (now - self.last_ui_epoch_post >= self.UI_EPOCH_UPDATE_INTERVAL_S):
            try:
                self.ui_queue.put_nowait({
                    "type": "epoch_written",
                    "timestamp": common_timestamp,
                    "devices": self.device_order.copy(),
                    "num_devices": self.expected_devices,
                    "pressures": pressures,
                    "complete": bool(complete),
                })
                self.last_ui_epoch_post = now
            except Exception:
                pass

    def _expire_old_epochs(self, now, force=False):
        if not self.epochs:
            return
        while self.epochs:
            oldest_epoch = min(self.epochs.keys())
            epoch_state = self.epochs[oldest_epoch]
            is_complete = len(epoch_state["slots"]) >= self.expected_devices
            is_stale = force or ((now - epoch_state["created_at"]) >= self.PARTIAL_EPOCH_TIMEOUT_S)
            if not is_complete and not is_stale:
                break
            self.epochs.pop(oldest_epoch, None)
            self._write_epoch(oldest_epoch, epoch_state, complete=is_complete)

    def _flush_csv_if_needed(self, now, force=False):
        if not self.csv_file:
            return
        if force or (now - self.last_flush >= self.CSV_FLUSH_INTERVAL_S):
            self.csv_file.flush()
            self.last_flush = now

    def _post_runtime_stats(self, now, force=False):
        if force or (now - self.last_stats_post >= self.STATS_UPDATE_INTERVAL_S):
            self._post_packet_info("", 0, 0, None, malformed=False)
            self.last_stats_post = now

    def _post_gap_stats(self, now, force=False, final=False):
        if not self.ui_queue:
            return
        if not force and (now - self.last_gap_post < self.GAP_STATS_INTERVAL_S):
            return

        total_expected_lines = self.rows_written + self.gap_missing_lines
        loss_gap_percent = (
            100.0 * self.gap_missing_lines / total_expected_lines
        ) if total_expected_lines else 0.0
        try:
            self.ui_queue.put_nowait({
                "type": "gap_stats",
                "lost_count": int(self.gap_missing_lines),
                "total_intervals": int(total_expected_lines),
                "loss_gap_percent": float(loss_gap_percent),
                "final": final
            })
            self.last_gap_post = now
        except Exception:
            pass

    def _post_packet_info(self, src_ip, data_len, samples, first_ts, malformed=False):
        """Envoie un message à l'UI pour la mise à jour des stats"""
        try:
            if self.ui_queue:
                total_expected = self.rows_written + self.gap_missing_lines + self.invalid_timestamps
                loss_percent = (
                    100.0 * (self.invalid_timestamps + self.gap_missing_lines) / total_expected
                ) if total_expected else 0.0
                self.ui_queue.put_nowait({
                    "type": "packet",
                    "src": src_ip,
                    "len": data_len,
                    "samples": samples,
                    "first_ts": first_ts,
                    "valid": samples if not malformed else 0,
                    "malformed": malformed,
                    "packets_received": self.packets_received,
                    "total_lines": self.rows_written,
                    "lost_lines": self.invalid_timestamps,
                    "gap_lost_lines": self.gap_missing_lines,
                    "partial_rows": self.partial_rows_written,
                    "held_rows": self.held_rows_written,
                    "held_slots": self.held_slots_reused,
                    "blank_slots": self.blank_slots_written,
                    "complete_rows": self.complete_rows_written,
                    "loss_percent": loss_percent,
                })
        except Exception:
            pass

    def _compute_and_post_gap_stats(self, final=False):
        self._post_gap_stats(time.time(), force=True, final=final)

    def stop(self):
        self.stop_event.set()


# =====================
# GUI
# =====================
class UDPReceiverGUI:
    def __init__(self, master):
        self.master = master
        self.master.title(APP_VERSION)
        self.style = Style(theme="superhero")

        # default params (conformes à ton code base)
        self.udp_ip = "0.0.0.0"
        self.udp_port = 12346
        self.sample_period = 5000  # µs  (200 Hz)
        self.expected_devices = tk.IntVar(value=1)
        self.file_path = tk.StringVar(value="data.csv")

        # map params (user)
        self.fps = tk.IntVar(value=10)
        self.vmin_user = tk.StringVar(value="")  # empty => auto
        self.vmax_user = tk.StringVar(value="")  # empty => auto
        self.hp_enabled = tk.BooleanVar(value=True)
        self.hp_window = tk.IntVar(value=30)  # nombre d'échantillons pour la moyenne (filtre)
        self.hp_mode_ema = tk.BooleanVar(value=False)  # False = ring buffer (O(1)), True = EMA

        # thread + queue
        self.ui_queue = Queue()
        self.receiver_thread = None
        self.cnc_thread = None
        self._stop_all = threading.Event()

        # EMA scaling for automatic vmin/vmax
        self._ema_vmin = None
        self._ema_vmax = None
        self._ema_alpha = 0.02
        self._last_draw = 0.0

        # positions (p0..p11) - layout existant // pX -> X = Id of the ESP, location (row,line)
        p0 = [(1,0),(2,0),(2,1),(3,1),(3,2),(4,2),(4,3),(5,3)]
        p1 = [(7,0),(8,0),(7,1),(8,1),(6,2),(7,2),(6,3),(5,4)]
        p2 = [(3,0),(4,0),(5,0),(6,0),(4,1),(5,1),(6,1),(5,2)]
        p3 = [(9,1),(9,2),(8,2),(8,3),(7,3),(7,4),(6,4),(6,5)]
        p4 = [(9,7),(9,8),(8,7),(8,8),(7,6),(7,7),(6,6),(5,5)]
        p5 = [(9,3),(9,4),(9,5),(9,6),(8,4),(8,5),(8,6),(7,5)]
        p6 = [(8,9),(7,9),(7,8),(6,8),(6,7),(5,7),(5,6),(4,6)]
        p7 = [(2,9),(1,9),(2,8),(1,8),(2,7),(3,7),(3,6),(4,5)]
        p8 = [(6,9),(5,9),(4,9),(3,9),(5,8),(4,8),(3,8),(4,7)]
        p9 = [(0,8),(0,7),(1,7),(1,6),(2,6),(2,5),(3,5),(3,4)]
        p10= [(0,2),(0,1),(1,2),(1,1),(2,3),(2,2),(3,3),(4,4)]
        p11= [(0,6),(0,5),(0,4),(0,3),(1,5),(1,4),(1,3),(2,4)]
        pos = np.vstack([np.array(p, dtype=int) for p in [p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11]])
        self.UIx, self.UIy = pos[:,0], pos[:,1]
        self.max_positions = len(self.UIx)  # 96

        # ========= HP FILTER STATE =========
        self._hp_win = max(1, int(self.hp_window.get()))
        self.hp_buf = [ [0.0]*self._hp_win for _ in range(self.max_positions) ]
        self.hp_idx = [0]*self.max_positions
        self.hp_cnt = [0]*self.max_positions
        self.hp_sum = [0.0]*self.max_positions
        self.ema = [0.0]*self.max_positions
        self._alpha = self._compute_alpha_from_window()

        # UI state
        self.packet_rows = deque(maxlen=300)
        self._poll_after_id = None
        self.cnc_x_var = tk.StringVar(value="--")
        self.cnc_y_var = tk.StringVar(value="--")
        self.cnc_z_var = tk.StringVar(value="--")
        self.cnc_a_var = tk.StringVar(value="--")
        self.cnc_motion_var = tk.StringVar(value="Hors ligne")

        # build UI
        self._build_ui()

        # start ui poll
        self._poll_ui()

    # ----- helpers HP -----
    def _compute_alpha_from_window(self):
        Ts = self.sample_period / 1_000_000.0  # µs -> s
        N = max(1, int(self.hp_window.get()))
        tau = max((N * Ts) / 2.0, 1e-6)
        alpha = 1.0 - math.exp(-Ts / tau)
        return float(alpha)

    def _reinit_hp_buffers(self):
        self._hp_win = max(1, int(self.hp_window.get()))
        self.hp_buf = [ [0.0]*self._hp_win for _ in range(self.max_positions) ]
        self.hp_idx = [0]*self.max_positions
        self.hp_cnt = [0]*self.max_positions
        self.hp_sum = [0.0]*self.max_positions
        self.ema = [0.0]*self.max_positions
        self._alpha = self._compute_alpha_from_window()

    # ---------------- UI build ----------------
    def _build_ui(self):
        root = self.master
        root.geometry("1100x860")
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=1)
        root.columnconfigure(0, weight=1)

        # Top controls frame
        top = ttkb.Frame(root, padding=(8,8))
        top.grid(row=0, column=0, sticky="ew")
        for i in range(16):
            top.columnconfigure(i, weight=0)
        top.columnconfigure(3, weight=1)

        ttkb.Label(top, text="Nombre d'ESP:").grid(row=0, column=0, sticky="w")
        ttkb.Entry(top, textvariable=self.expected_devices, width=6).grid(row=0, column=1, sticky="w", padx=(4,12))

        ttkb.Label(top, text="Fichier CSV:").grid(row=0, column=2, sticky="w")
        ttkb.Entry(top, textvariable=self.file_path, width=45).grid(row=0, column=3, columnspan=4, sticky="ew", padx=(4,8))
        ttkb.Button(top, text="Parcourir", command=self.browse_file).grid(row=0, column=7, sticky="w")

        self.btn_run = ttkb.Button(top, text="Run", bootstyle="success", command=self.start_receiving)
        self.btn_stop = ttkb.Button(top, text="Stop", bootstyle="danger", command=self.stop_receiving, state=tk.DISABLED)
        self.btn_graph = ttkb.Button(top, text="Graph", bootstyle="info", command=self.show_graph)
        self.btn_scan = ttkb.Button(top, text="Scan", bootstyle="secondary", command=self.scan_devices)

        self.btn_run.grid(row=0, column=8, padx=6, sticky="e")
        self.btn_stop.grid(row=0, column=9, padx=6, sticky="e")
        self.btn_graph.grid(row=0, column=10, padx=6, sticky="e")
        self.btn_scan.grid(row=0, column=11, padx=6, sticky="e")

        # Voyant CNC
        ttkb.Label(top, text="CNC").grid(row=0, column=12, padx=(16,4), sticky="e")
        self.cam_led = tk.Canvas(top, width=20, height=20,highlightthickness=0, bd=0,bg="#2b2b2b")   # couleur foncée par défaut
        self.cam_led.grid(row=0, column=13, padx=(0,8), sticky="w")
        self._cam_led_oval = self.cam_led.create_oval(3, 3, 17, 17, fill="red", outline="black")
        ttkb.Label(top, textvariable=self.cnc_motion_var).grid(row=0, column=14, padx=(0,12), sticky="w")
        ttkb.Label(top, text="X").grid(row=1, column=8, sticky="e", pady=(6,0))
        ttkb.Label(top, textvariable=self.cnc_x_var, width=10).grid(row=1, column=9, sticky="w", padx=(4,8), pady=(6,0))
        ttkb.Label(top, text="Y").grid(row=1, column=10, sticky="e", pady=(6,0))
        ttkb.Label(top, textvariable=self.cnc_y_var, width=10).grid(row=1, column=11, sticky="w", padx=(4,8), pady=(6,0))
        ttkb.Label(top, text="Z").grid(row=1, column=12, sticky="e", pady=(6,0))
        ttkb.Label(top, textvariable=self.cnc_z_var, width=10).grid(row=1, column=13, sticky="w", padx=(4,8), pady=(6,0))
        ttkb.Label(top, text="A").grid(row=1, column=14, sticky="e", pady=(6,0))
        ttkb.Label(top, textvariable=self.cnc_a_var, width=10).grid(row=1, column=15, sticky="w", padx=(4,0), pady=(6,0))

        # Middle: map (left) + stats+log (right)
        middle = ttkb.Frame(root, padding=(8,4))
        middle.grid(row=1, column=0, sticky="nsew")
        middle.rowconfigure(0, weight=1)
        middle.columnconfigure(0, weight=2)
        middle.columnconfigure(1, weight=1)

        # Left: Map
        left = ttkb.Labelframe(middle, text="Map de pression (live)", padding=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,6), pady=4)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        # Canvas sizing and cell layout
        self.cell_size = 38
        self.canvas_pad = 12
        self.canvas_w = 10 * self.cell_size + 2 * self.canvas_pad
        self.canvas_h = 10 * self.cell_size + 2 * self.canvas_pad

        self.canvas = tk.Canvas(left, width=self.canvas_w, height=self.canvas_h, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # create rects for each sensor position
        self._rects = []
        for (gx, gy) in zip(self.UIx.astype(int), self.UIy.astype(int)):
            x0 = self.canvas_pad + gx * self.cell_size + 4
            y0 = 9*self.cell_size + 4 + self.canvas_pad - ( gy * self.cell_size + 4)
            x1 = x0 + self.cell_size - 8
            y1 = y0 + self.cell_size - 8
            r = self.canvas.create_rectangle(x0, y0, x1, y1, fill="#777777", outline="")
            self._rects.append(r)

        # legend texts on canvas
        self.legend_min = self.canvas.create_text(10, self.canvas_h - 10, anchor="w", fill="#ffffff", font=("TkDefaultFont", 8), text="min")
        self.legend_mid = self.canvas.create_text(self.canvas_w//2, self.canvas_h - 10, anchor="center", fill="#ffffff", font=("TkDefaultFont", 8), text="mid")
        self.legend_max = self.canvas.create_text(self.canvas_w - 10, self.canvas_h - 10, anchor="e", fill="#ffffff", font=("TkDefaultFont", 8), text="max")

        # Right: stats + logs
        right = ttkb.Frame(middle)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        stats_frame = ttkb.Labelframe(right, text="Stats", padding=6)
        stats_frame.grid(row=0, column=0, sticky="ew", pady=(0,6))
        stats_frame.columnconfigure(1, weight=1)

        self.lbl_packets = ttkb.Label(stats_frame, text="Paquets: 0")
        self.lbl_lines = ttkb.Label(stats_frame, text="Lignes CSV: 0")
        self.lbl_lost = ttkb.Label(stats_frame, text="TS invalides: 0")
        self.lbl_losspct = ttkb.Label(stats_frame, text="Perte estimée: 0.00 % | Hold: 0")
        self.lbl_gap = ttkb.Label(stats_frame, text="Perte (gap): N/A")

        self.lbl_packets.grid(row=0, column=0, sticky="w")
        self.lbl_lines.grid(row=1, column=0, sticky="w")
        self.lbl_lost.grid(row=2, column=0, sticky="w")
        self.lbl_losspct.grid(row=3, column=0, sticky="w")
        self.lbl_gap.grid(row=4, column=0, sticky="w")

        # Log
        log_frame = ttkb.Labelframe(right, text="Log", padding=6)
        log_frame.grid(row=2, column=0, sticky="nsew")
        self.log_text = scrolledtext.ScrolledText(log_frame, height=16)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Controls under map
        ctrl = ttkb.Frame(left, padding=4)
        ctrl.grid(row=1, column=0, sticky="ew", pady=(6,0))
        ctrl.columnconfigure(9, weight=1)

        ttkb.Label(ctrl, text="FPS:").grid(row=0, column=0, sticky="w")
        ttkb.Entry(ctrl, textvariable=self.fps, width=6).grid(row=0, column=1, sticky="w", padx=(4,8))

        ttkb.Label(ctrl, text="Val min:").grid(row=0, column=2, sticky="w")
        ttkb.Entry(ctrl, textvariable=self.vmin_user, width=8).grid(row=0, column=3, sticky="w", padx=(4,8))

        ttkb.Label(ctrl, text="Val max:").grid(row=0, column=4, sticky="w")
        ttkb.Entry(ctrl, textvariable=self.vmax_user, width=8).grid(row=0, column=5, sticky="w", padx=(4,8))

        ttkb.Button(ctrl, text="Appliquer", bootstyle="info", command=self.apply_user_params).grid(row=0, column=6, sticky="w", padx=(4,8))
        ttkb.Button(ctrl, text="Reset auto", bootstyle="secondary", command=self.reset_user_params).grid(row=0, column=7, sticky="e")

        ttkb.Checkbutton(ctrl, text="Filtre passe-haut", variable=self.hp_enabled).grid(row=1, column=0, sticky="w", pady=(6,0))
        ttkb.Label(ctrl, text="HP window:").grid(row=1, column=2, sticky="e", padx=(8,0))
        ttkb.Entry(ctrl, textvariable=self.hp_window, width=6).grid(row=1, column=3, sticky="w", padx=(4,8))
        ttkb.Checkbutton(ctrl, text="Mode EMA (IIR)", variable=self.hp_mode_ema).grid(row=1, column=4, sticky="w", padx=(12,0))

        # Bottom status bar
        status = ttkb.Frame(root, padding=(6,4))
        status.grid(row=3, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Idle")
        self.status_lbl = ttkb.Label(status, textvariable=self.status_var, anchor="w")
        self.status_lbl.grid(row=0, column=0, sticky="w")

        # init CNC display
        x, y, z, a, motion_state, valid = CNC.get()
        self._set_cnc_display(x, y, z, a, motion_state, valid)

    # ----- UI helpers -----
    def _set_cam_led(self, state: int):
        color = "green" if state else "red"
        try:
            self.cam_led.itemconfig(self._cam_led_oval, fill=color)
        except Exception:
            pass

    def _set_cnc_display(self, x, y, z, a, motion_state: int, valid: bool):
        self._set_cam_led(1 if valid and motion_state else 0)

        if not valid:
            self.cnc_motion_var.set("Hors ligne")
            self.cnc_x_var.set("--")
            self.cnc_y_var.set("--")
            self.cnc_z_var.set("--")
            self.cnc_a_var.set("--")
            return

        self.cnc_motion_var.set("En mouvement" if motion_state else "A l'arret")
        self.cnc_x_var.set(f"{float(x):.3f}")
        self.cnc_y_var.set(f"{float(y):.3f}")
        self.cnc_z_var.set(f"{float(z):.3f}")
        self.cnc_a_var.set(f"{float(a):.3f}")

    def browse_file(self):
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if filename:
            self.file_path.set(filename)

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        try:
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        except Exception:
            print(f"[{timestamp}] {message}")

    # ----- Thread control -----
    def start_receiving(self):
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.log("Le thread capteurs est déjà en cours.")
            return

        # (Re)init HP buffers/alpha according to current settings
        self._reinit_hp_buffers()
        self.log(
            "[CNC] Mapping CSV actif: CNC_X=X, "
            f"CNC_Y={'Z' if CSV_SWAP_CNC_YZ else 'Y'}, "
            f"CNC_Z={'Y' if CSV_SWAP_CNC_YZ else 'Z'}, CNC_A=A, "
            f"align<={CNC_CSV_MAX_ALIGN_DELTA_US}us, "
            f"fallback={'latest' if CNC_CSV_FALLBACK_TO_LATEST else 'none'}"
        )

        fn = os.path.dirname(self.file_path.get())
        if fn and not os.path.exists(fn):
            os.makedirs(fn, exist_ok=True)

        # Thread capteurs
        self.receiver_thread = UDPReceiverThread(
            self.udp_ip, self.udp_port, self.sample_period,
            self.expected_devices.get(), self.file_path.get(),
            log_callback=self.log, ui_queue=self.ui_queue
        )
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

        # Thread telemetrie CNC
        if not self.cnc_thread or not self.cnc_thread.is_alive():
            self.cnc_thread = CncReceiverThread(
                log_callback=self.log, ui_queue=self.ui_queue
            )
            self.cnc_thread.daemon = True
            self.cnc_thread.start()

        self.status_var.set("En écoute...")
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_graph.config(state=tk.DISABLED)
        self.log("Réception démarrée (capteurs + télémétrie CNC).")

    def stop_receiving(self):
        # stop capteurs
        if self.receiver_thread:
            self.log("Arrêt du thread capteurs...")
            self.receiver_thread.stop()
            self.receiver_thread.join(timeout=2.0)
            if self.receiver_thread.is_alive():
                self.log("Le thread UDP capteurs ne s'est pas arrêté à temps.")
            else:
                self.log("Thread capteurs arrêté proprement.")
            self.receiver_thread = None

        # stop telemetrie CNC
        if self.cnc_thread:
            self.log("Arrêt du thread CNC...")
            self.cnc_thread.stop_event.set()
            self.cnc_thread.join(timeout=2.0)
            if self.cnc_thread.is_alive():
                self.log("Le thread CNC ne s'est pas arrêté à temps.")
            else:
                self.log("Thread CNC arrêté proprement.")
            self.cnc_thread = None

        x, y, z, a, _motion_state, valid = CNC.get()
        self._set_cnc_display(x, y, z, a, 0, valid)

        self.status_var.set("Idle")
        self.btn_run.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_graph.config(state=tk.NORMAL)

    def scan_devices(self):
        def ping(ip):
            try:
                result = subprocess.run(["ping", "-c", "1", "-W", "1", ip],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                return result.returncode == 0
            except Exception:
                return False

        def worker():
            self.log("Scan du réseau...")
            base = "192.168.0."
            for i in range(100, 130):
                ip = base + str(i)
                if ping(ip):
                    self.log(f"[OK] {ip} actif")
                else:
                    self.log(f"[KO] {ip} inactif")
            self.log("Scan terminé.")
        threading.Thread(target=worker, daemon=True).start()

    def show_graph(self):
        filepath = self.file_path.get()
        if not os.path.exists(filepath):
            self.log("Fichier CSV introuvable.")
            return
        try:
            import matplotlib.pyplot as plt
            df = pd.read_csv(filepath, sep=CSV_DELIMITER, decimal=",")
            if "timestamp" not in df.columns:
                self.log("Pas de colonne 'timestamp' dans le fichier.")
                return

            df["timestamp_s"] = df["timestamp"].astype(np.int64) / 1_000_000.0
            # ne pas tracer les colonnes CNC ici
            sensor_columns = [c for c in df.columns if c not in ("timestamp", "timestamp_s", "CNC_X", "CNC_Y", "CNC_Z", "CNC_A", "CNC_Moving")]
            if not sensor_columns:
                self.log("Aucune colonne capteur détectée.")
                return

            n_sensors = len(sensor_columns)
            est_esp_by_file = n_sensors // 8
            num_esp = min(self.expected_devices.get(), est_esp_by_file) if self.expected_devices.get() > 0 else est_esp_by_file
            if num_esp == 0:
                plt.figure(figsize=(12,6))
                for col in sensor_columns:
                    plt.plot(df["timestamp_s"], df[col], label=col)
                plt.legend()
                plt.xlabel("Temps (s)")
                plt.title("Données capteurs (toutes colonnes)")
                plt.grid(True)
                plt.show()
                return

            groups = [sensor_columns[i*8:(i+1)*8] for i in range(num_esp)]
            plt.figure(figsize=(12,6))
            for i, g in enumerate(groups):
                if len(g) >= 1:
                    col = g[0]
                    plt.plot(df["timestamp_s"], df[col], label=f"ESP{i+1} - {col}")
            plt.legend(fontsize=8)
            plt.xlabel("Temps (s)")
            plt.title("Évolution du capteur 1 pour chaque ESP")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.log(f"Erreur lors du graphe : {e}")

    # ----- Poll UI queue -----
    def _poll_ui(self):
        """
        Drain ui_queue, process messages.
        Drawing is throttled by FPS inside _update_pressure_map.
        """
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                typ = msg.get("type")
                if typ == "packet":
                    # update stats labels
                    self.lbl_packets.config(text=f"Paquets: {msg.get('packets_received', 0)}")
                    self.lbl_lines.config(text=f"Lignes CSV: {msg.get('total_lines', 0)}")
                    self.lbl_lost.config(text=f"TS invalides: {msg.get('lost_lines', 0)}")
                    pct = float(msg.get('loss_percent', 0.0))
                    held_rows = int(msg.get('held_rows', 0))
                    held_slots = int(msg.get('held_slots', 0))
                    self.lbl_losspct.config(text=f"Perte estimée: {pct:.2f} % | Hold: {held_rows} lignes / {held_slots} ESP")

                elif typ == "epoch_written":
                    ts = msg.get("timestamp")
                    devices = msg.get("devices", [])
                    num_devices = int(msg.get("num_devices", 0))
                    pressures = msg.get("pressures", None)
                    self.btn_graph.config(state=tk.NORMAL)

                    if pressures is not None:
                        try:
                            num_devices = num_devices if num_devices > 0 else (len(devices) if len(devices) > 0 else self.expected_devices.get())
                            self._update_pressure_map(pressures, int(num_devices), ts)
                        except Exception as e:
                            self.log(f"Erreur mise à jour map: {e}")

                elif typ == "gap_stats":
                    lost_count = msg.get("lost_count", 0)
                    total_intervals = msg.get("total_intervals", 0)
                    pct = msg.get("loss_gap_percent", 0.0)
                    self.lbl_gap.config(text=f"Perte (gap): {lost_count}/{total_intervals} ({pct:.2f}%)")

                elif typ == "final_stats":
                    self.log(
                        f"Stats finales: lignes={msg.get('total_lines')} "
                        f"ts_invalides={msg.get('lost_lines')} "
                        f"gaps={msg.get('gap_lost_lines', 0)} "
                        f"hold_lignes={msg.get('held_rows', 0)} "
                        f"hold_esp={msg.get('held_slots', 0)} "
                        f"blancs={msg.get('blank_slots', 0)} "
                        f"({msg.get('loss_percent'):.2f}%)"
                    )
                    self.lbl_packets.config(text=f"Paquets: {msg.get('packets_received', 0)}")

                elif typ == "cnc_state":
                    self._set_cnc_display(
                        msg.get("x", 0.0),
                        msg.get("y", 0.0),
                        msg.get("z", 0.0),
                        msg.get("a", 0.0),
                        int(msg.get("motion_state", 0)),
                        bool(msg.get("valid", False))
                    )

        except Empty:
            pass

        # schedule next poll according to FPS
        try:
            fps = max(1, int(self.fps.get()))
        except Exception:
            fps = 10
            self.fps.set(fps)
        poll_ms = int(1000 / max(20, fps))  # poll queue à max 20Hz (dessin est déjà throttlé)
        if self._poll_after_id:
            try:
                self.master.after_cancel(self._poll_after_id)
            except Exception:
                pass
        self._poll_after_id = self.master.after(poll_ms, self._poll_ui)

    # -----------------
    # Color mapping and helpers
    # -----------------
    def _val_to_rgb(self, v_norm):
        """ Blue (0) -> White (0.5) -> Red (1.0). v_norm in [0,1] or NaN. """
        if v_norm is None or math.isnan(v_norm):
            return "#d9d9d9"  # grey
        v = max(0.0, min(1.0, float(v_norm)))
        if v <= 0.5:
            t = v / 0.5
            r = int(255 * t)
            g = int(255 * t)
            b = 255
        else:
            t = (v - 0.5) / 0.5
            r = 255
            g = int(255 * (1.0 - t))
            b = int(255 * (1.0 - t))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _parse_user_float(self, s):
        try:
            if s is None: return None
            ss = str(s).strip()
            if ss == "": return None
            return float(ss)
        except Exception:
            return None

    def _update_scaling_ema(self, display_values):
        valid = display_values[~np.isnan(display_values)]
        if valid.size == 0:
            return
        cur_min = float(np.nanmin(valid))
        cur_max = float(np.nanmax(valid))
        if self._ema_vmin is None:
            self._ema_vmin = cur_min
            self._ema_vmax = cur_max
        else:
            a = self._ema_alpha
            self._ema_vmin = (1-a)*self._ema_vmin + a*cur_min
            self._ema_vmax = (1-a)*self._ema_vmax + a*cur_max

    # -----------------
    # Update map (main)
    # -----------------
    def _update_pressure_map(self, pressures, num_devices, timestamp=None):
        try:
            arr = np.asarray(pressures, dtype=float).flatten()
        except Exception:
            return

        vals = np.full(self.max_positions, np.nan)
        count = min(len(arr), int(num_devices) * 8, self.max_positions)
        if count > 0:
            vals[:count] = arr[:count]

        desired_hp = max(1, int(self.hp_window.get()))
        if desired_hp != self._hp_win:
            self._reinit_hp_buffers()

        filtered = np.full_like(vals, np.nan)

        if self.hp_enabled.get():
            if self.hp_mode_ema.get():
                alpha = self._alpha
                ema = self.ema
                for i in range(self.max_positions):
                    v = vals[i]
                    if np.isnan(v):
                        continue
                    vi = float(v)
                    ema[i] = (1.0 - alpha) * ema[i] + alpha * vi
                    filtered[i] = vi - ema[i]
            else:
                win = self._hp_win
                hp_buf = self.hp_buf
                hp_idx = self.hp_idx
                hp_cnt = self.hp_cnt
                hp_sum = self.hp_sum

                for i in range(self.max_positions):
                    v = vals[i]
                    if np.isnan(v):
                        continue
                    vi = float(v)
                    idx = hp_idx[i]
                    if hp_cnt[i] < win:
                        hp_buf[i][idx] = vi
                        hp_sum[i] += vi
                        hp_cnt[i] += 1
                        hp_idx[i] = (idx + 1) % win
                    else:
                        old = hp_buf[i][idx]
                        hp_buf[i][idx] = vi
                        hp_sum[i] += vi - old
                        hp_idx[i] = (idx + 1) % win
                    baseline = hp_sum[i] / hp_cnt[i]
                    filtered[i] = vi - baseline
        else:
            filtered = vals.copy()

        # throttling by FPS
        try:
            fps = max(1, int(self.fps.get()))
        except Exception:
            fps = 10
            self.fps.set(fps)
        now = time.time()
        if now - self._last_draw < 1.0 / fps:
            if self._parse_user_float(self.vmin_user.get()) is None and self._parse_user_float(self.vmax_user.get()) is None:
                self._update_scaling_ema(filtered)
            return
        self._last_draw = now

        user_vmin = self._parse_user_float(self.vmin_user.get())
        user_vmax = self._parse_user_float(self.vmax_user.get())

        if user_vmin is not None and user_vmax is not None:
            vmin, vmax = float(user_vmin), float(user_vmax)
        else:
            self._update_scaling_ema(filtered)
            if self._ema_vmin is None or self._ema_vmax is None:
                valid = filtered[~np.isnan(filtered)]
                if valid.size:
                    vmin = float(np.nanmin(valid))
                    vmax = float(np.nanmax(valid))
                else:
                    vmin, vmax = 0.0, 1.0
            else:
                vmin = float(self._ema_vmin)
                vmax = float(self._ema_vmax)

        if vmin == vmax:
            vmax = vmin + 1.0
        rng = vmax - vmin if (vmax - vmin) != 0 else 1.0

        rects = self._rects
        for idx, rect in enumerate(rects):
            val = filtered[idx]
            if np.isnan(val):
                color = "#d9d9d9"
            else:
                vnorm = (val - vmin) / rng
                color = self._val_to_rgb(vnorm)
            try:
                self.canvas.itemconfig(rect, fill=color)
            except Exception:
                pass

        try:
            self.canvas.itemconfig(self.legend_min, text=f"{vmin:.0f}")
            mid = (vmin + vmax) / 2.0
            self.canvas.itemconfig(self.legend_mid, text=f"{mid:.0f}")
            self.canvas.itemconfig(self.legend_max, text=f"{vmax:.0f}")
        except Exception:
            pass

        try:
            self.master.title(APP_VERSION)
        except Exception:
            pass

    # apply & reset user params
    def apply_user_params(self):
        try:
            fps = max(1, int(self.fps.get()))
        except Exception:
            fps = 10
            self.fps.set(fps)

        if max(1, int(self.hp_window.get())) != self._hp_win:
            self._reinit_hp_buffers()
        else:
            self._alpha = self._compute_alpha_from_window()

        self.log(f"Paramètres appliqués : FPS={fps}, vmin={self.vmin_user.get() or 'auto'}, "
                 f"vmax={self.vmax_user.get() or 'auto'}, HP={'on' if self.hp_enabled.get() else 'off'}, "
                 f"Mode={'EMA' if self.hp_mode_ema.get() else f'MA(N={self._hp_win})'}, alpha={self._alpha:.4g}")

    def reset_user_params(self):
        self.fps.set(10)
        self.vmin_user.set("")
        self.vmax_user.set("")
        self.hp_enabled.set(True)
        self.hp_window.set(30)
        self.hp_mode_ema.set(False)
        self._ema_vmin = None
        self._ema_vmax = None
        self._reinit_hp_buffers()
        self.log("Paramètres réinitialisés (auto + HP par défaut).")

    # teardown / close
    def close(self):
        if self._poll_after_id:
            try:
                self.master.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None
        self.stop_receiving()

# =====================
# MAIN
# =====================
def main():
    root = ttkb.Window(title="Sensors Network", size=(1200, 800))
    gui = UDPReceiverGUI(root)

    def _on_close():
        try:
            gui.stop_receiving()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()

if __name__ == "__main__":
    main()