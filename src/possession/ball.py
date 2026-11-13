"""Ball reconstruction: fill short gaps, flag long gaps unknown, derive speed and acceleration."""
import numpy as np

FPS = 25
MAX_GAP = 12  # frames (0.5 s) that are interpolated; longer gaps stay unknown


def reconstruct(bx, by, max_gap=MAX_GAP):
    """Return dict: x, y (NaN where unknown), observed (bool), known (bool), speed, acc (m/s, m/s^2)."""
    n = len(bx)
    obs = np.isfinite(bx) & np.isfinite(by)
    idx = np.flatnonzero(obs)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    x[idx], y[idx] = bx[idx], by[idx]
    known = obs.copy()
    if len(idx) > 1:
        gaps = np.diff(idx)
        for a, g in zip(idx[:-1][gaps > 1], gaps[gaps > 1]):
            if g - 1 <= max_gap:
                f = np.arange(a + 1, a + g)
                t = (f - a) / g
                b = a + g
                x[f] = x[a] + t * (x[b] - x[a])
                y[f] = y[a] + t * (y[b] - y[a])
                known[f] = True
    # speed / acceleration from central differences; only where neighbours are known
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    vx[1:-1] = (x[2:] - x[:-2]) * FPS / 2
    vy[1:-1] = (y[2:] - y[:-2]) * FPS / 2
    speed = np.hypot(vx, vy)
    acc = np.full(n, np.nan)
    acc[1:-1] = (speed[2:] - speed[:-2]) * FPS / 2
    return {"x": x, "y": y, "observed": obs, "known": known, "speed": speed, "acc": acc}
