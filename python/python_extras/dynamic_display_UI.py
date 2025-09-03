import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

PATH = r"D:\data_Corentin\TEST_024.csv"
DURATION = 20          # rows required between plot updates
WINDOW_MULT = 2         # keep two durations worth of rows
WINDOW_ROWS = WINDOW_MULT * DURATION


class ReplayTailer:
    """Mimics CsvTailer.read_new_rows() by replaying an existing CSV in chunks."""
    def __init__(self, path, chunk=20):
        self.f = open(path, "r", newline="", encoding="utf-8")
        # read header
        header_line = self.f.readline()
        self.header = next(csv.reader([header_line]))
        self.chunk = chunk

    def read_new_rows(self):
        rows, n = [], 0
        while n < self.chunk:
            line = self.f.readline()
            if not line or not line.endswith("\n"):
                # no more complete lines
                break
            parts = next(csv.reader([line]))
            try:
                ts = float(parts[0])
                pressures = np.asarray(parts[1:97], dtype=float)
                if pressures.size != 96:
                    continue
                rows.append((ts, pressures))
                n += 1
            except:
                continue
        return rows

    def close(self):
        self.f.close()

# ---- Tail the CSV incrementally (no full reloads) ----
class CsvTailer:
    def __init__(self, path):
        self.f = open(path, "r", newline="", encoding="utf-8")
        # read header once
        header_line = self._read_complete_line()
        if header_line is None:
            raise RuntimeError("CSV has no header yet.")
        self.header = next(csv.reader([header_line]))
        # basic sanity: expect 1 timestamp + 96 pressures
        if len(self.header) < 97:
            raise RuntimeError(f"Expected at least 97 columns, got {len(self.header)}")
        self.partial_pos = self.f.tell()

    def _read_complete_line(self):
        """Return the next line only if it ends with '\n'. Otherwise rewind and return None."""
        pos = self.f.tell()
        line = self.f.readline()
        if not line or not line.endswith("\n"):
            self.f.seek(pos)
            return None
        return line

    def read_new_rows(self):
        """
        Read any fully written new lines since the last call.
        Returns a list of tuples: (timestamp: float, pressures: np.ndarray shape (96,))
        """
        rows = []
        while True:
            line = self._read_complete_line()
            if line is None:
                break
            parts = next(csv.reader([line]))
            # Parse timestamp and pressures to floats; ignore any extra cols beyond 97
            try:
                ts = float(parts[0])
                pressures = np.asarray(parts[1:97], dtype=float)
            except Exception as e:
                # If a row is malformed, skip it rather than crashing the animation
                # (You can log/print e if helpful for debugging)
                continue
            if pressures.size != 96:
                continue
            rows.append((ts, pressures))
        return rows

    def close(self):
        self.f.close()


# ---- layout ----
p0 = [(1,0), (2,0), (2,1), (3,1), (3,2), (4,2), (4,3), (5,3)]
p1 = [(7,0), (8,0), (7,1), (8,1), (6,2), (7,2), (6,3), (5,4)]
p2 = [(3,0), (4,0), (5,0), (6,0), (4,1), (5,1), (6,1), (5,2)]
p3 = [(9,1), (9,2), (8,2), (8,3), (7,3), (7,4), (6,4), (6,5)]
p4 = [(9,7), (9,8), (8,7), (8,8), (7,6), (7,7), (6,6), (5,5)]
p5 = [(9,3), (9,4), (9,5), (9,6), (8,4), (8,5), (8,6), (7,5)]
p6 = [(8,9), (7,9), (7,8), (6,8), (6,7), (5,7), (5,6), (4,6)]
p7 = [(2,9), (1,9), (2,8), (1,8), (2,7), (3,7), (3,6), (4,5)]
p8 = [(6,9), (5,9), (4,9), (3,9), (5,8), (4,8), (3,8), (4,7)]
p9 = [(0,8), (0,7), (1,7), (1,6), (2,6), (2,5), (3,5), (3,4)]
p10 = [(0,2), (0,1), (1,2), (1,1), (2,3), (2,2), (3,3), (4,4)]
p11 = [(0,6), (0,5), (0,4), (0,3), (1,5), (1,4), (1,3), (2,4)]
sensor_positions = [p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11]
pos = np.vstack([np.array(p, dtype=int) for p in sensor_positions])
x, y = pos[:,0], pos[:,1]

def compute_recent_mean(state, window_rows=20):
    """Return mean pressure vector over the last `window_rows` rows in the buffer."""
    if len(state["buffer"]) == 0:
        return np.zeros(96, dtype=float)
    rows_to_use = min(window_rows, len(state["buffer"]))
    # Take last `rows_to_use` rows
    pressures = np.array([state["buffer"][-i][1] for i in range(1, rows_to_use+1)])
    return pressures.mean(axis=0)

def append_new_rows_to_state(tailer, state):
    new_rows = tailer.read_new_rows()
    if not new_rows:
        return 0
    # Initialize t0 on the first successfully parsed row
    if state["t0"] is None:
        state["t0"] = new_rows[0][0]  # first timestamp in this batch
    # Append to ring buffer and bump the global counter
    for row in new_rows:
        state["buffer"].append(row)
        state["rows_seen"] += 1
    return len(new_rows)

def get_row_by_global_index(global_idx, state):
    """
    Map a global row index to a position inside the ring buffer.
    Returns (timestamp, pressures) or None if that row has been trimmed.
    """
    start_global = state["rows_seen"] - len(state["buffer"])
    pos_in_buffer = global_idx - start_global
    if pos_in_buffer < 0 or pos_in_buffer >= len(state["buffer"]):
        return None
    # deque supports indexing, it's O(n) but with small window it's fine
    return state["buffer"][pos_in_buffer]

def generate_data(duration, last_index, state):
    """
    Decide whether enough new rows have arrived.
    Returns (time, pressure_vector, new_index, ok)
    """
    if len(state["buffer"]) == 0:
        return None, None, last_index, False

    if last_index is None:
        # On first draw, wait until we have duration rows total
        if state["rows_seen"] >= duration:
            new_index = duration - 1  # zero-based
        else:
            return None, None, last_index, False
    else:
        rows_since = state["rows_seen"] - (last_index + 1)
        if rows_since >= duration:
            new_index = last_index + duration
        else:
            return None, None, last_index, False

    row = get_row_by_global_index(new_index, state)
    if row is None:
        # We trimmed past what we need; as a safeguard, jump to the most recent available row.
        # (This should not occur if WINDOW_ROWS >= DURATION.)
        newest_idx = state["rows_seen"] - 1
        row = get_row_by_global_index(newest_idx, state)
        new_index = newest_idx

    ts, pressure = row
    time = float(ts - state["t0"]) / 5000 // 200  # keep your original time math
    return time, pressure, new_index, True

def update(_frame, tailer, state, sc, title):
    # 1) Pull in only the newly appended lines (cheap)
    append_new_rows_to_state(tailer, state)

    # 2) If enough new rows accumulated, advance the plot
    time, vals, new_index, ok = generate_data(DURATION, state["last_index"], state)
    if ok:
        state["last_index"] = new_index
        # Compute baseline from the last 20 rows
        baseline = compute_recent_mean(state, window_rows=20)
        vals_corrected = (vals.astype(float) - baseline) / 4 # divide by 4 to match real pressure
        print(vals_corrected)
        sc.set_array(vals_corrected)
        title.set_text(f"Pressure map — t = {time:.2f}")
    return (sc,)

def main():
    # tailer = CsvTailer(PATH)
    tailer = ReplayTailer(PATH, chunk=DURATION)  # for offline “new rows” simulation
    state = {
        "buffer": deque(maxlen=WINDOW_ROWS),   # holds tuples (timestamp: float, pressures: np.ndarray)
        "rows_seen": 0,                        # total rows ever observed (monotonic)
        "last_index": None,                    # last global index used to update plot
        "t0": None,                            # first timestamp seen
    }

    # ---- Plot scaffold ----
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(x, y, c=np.zeros(len(x)), cmap="seismic", vmin=-4, vmax=4,
                    s=300, marker="s", edgecolor="none")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(-0.5, 9.5)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.grid(True, linewidth=0.5, alpha=0.4)
    cbar = plt.colorbar(sc, ax=ax, label="Pressure")
    title = ax.set_title("Pressure map")
    ani = FuncAnimation(fig, update, fargs=(tailer, state, sc, title), interval=DURATION, blit=False, cache_frame_data=False)
    plt.show()

    # Optional cleanup
    tailer.close()

if __name__ == "__main__":
    main()