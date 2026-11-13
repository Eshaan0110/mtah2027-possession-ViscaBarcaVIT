"""External validation data: Metrica open sample games (CSV format, games 1 and 2).

Other games, human-made event data: allowed by RULES.md. There are no per-frame possession labels in
that repository, so labels are DERIVED from events (an approximation, see NOTES.md):
  alive, team = team of the latest PASS / SHOT / RECOVERY / SET PIECE event (BALL LOST and CHALLENGE
  do not change it);  dead = from a BALL OUT end / FAULT RECEIVED / goal SHOT end until the next SET PIECE.
Source: https://github.com/metrica-sports/sample-data (acknowledged as requested by its README).
"""
import os

import numpy as np
import pandas as pd

from . import loader

DIR = os.path.join(loader.ROOT, "cache", "external", "metrica", "data")
PITCH = (105.0, 68.0)
FPS = 25


def _read_tracking(path):
    df = pd.read_csv(path, skiprows=2, header=0)
    cols = list(df.columns)
    players = []
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            players.append((df[c].to_numpy(float), df[cols[i + 1]].to_numpy(float)))
    ball = (df["Ball"].to_numpy(float), df[cols[cols.index("Ball") + 1]].to_numpy(float))
    return df["Period"].to_numpy(), players, ball


def load_game(g):
    base = os.path.join(DIR, "Sample_Game_%d" % g)
    per, hp, ball = _read_tracking(os.path.join(base, "Sample_Game_%d_RawTrackingData_Home_Team.csv" % g))
    _, ap, _ = _read_tracking(os.path.join(base, "Sample_Game_%d_RawTrackingData_Away_Team.csv" % g))
    slots = hp + ap
    px = np.stack([s[0] for s in slots], 1) * PITCH[0]
    py = np.stack([s[1] for s in slots], 1) * PITCH[1]
    # speed from positions (m/s), centred difference
    ps = np.full_like(px, np.nan)
    ps[1:-1] = np.hypot(px[2:] - px[:-2], py[2:] - py[:-2]) * FPS / 2
    arrays = {"px": px, "py": py, "ps": ps, "bx": ball[0] * PITCH[0], "by": ball[1] * PITCH[1]}
    team = np.array([0] * len(hp) + [1] * len(ap), dtype=np.int8)
    events = pd.read_csv(os.path.join(base, "Sample_Game_%d_RawEventsData.csv" % g))
    return arrays, team, per, events


def derive_labels(events, period, n):
    """Per-row labels H/A/D (X outside play) derived from events. Row i = tracking frame i+1."""
    lab = np.full(n, "X", dtype="<U1")
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        a, b = rows[0], rows[-1] + 1
        ev = events[events["Period"] == p]
        timeline = []  # (frame_row, state)
        for _, e in ev.iterrows():
            t, sub = e["Type"], str(e["Subtype"])
            sf, ef = int(e["Start Frame"]) - 1, int(e["End Frame"]) - 1
            if t in ("PASS", "RECOVERY", "SHOT", "SET PIECE"):
                state = "H" if e["Team"] == "Home" else "A"
                if t == "SET PIECE":
                    timeline.append((sf, state))
                elif t == "SHOT" and "GOAL" in sub:
                    timeline.append((sf, state))
                    timeline.append((max(ef, sf), "D"))
                else:
                    timeline.append((sf, state))
            elif t == "BALL OUT":
                timeline.append((max(ef, sf), "D"))
            elif t == "FAULT RECEIVED":
                timeline.append((sf, "D"))
        timeline.sort(key=lambda x: x[0])
        cur, pos = "X", a
        for f, s in timeline:
            f = min(max(f, a), b)
            lab[pos:f] = cur
            cur, pos = s, f
        lab[pos:b] = cur
        # frames before the first event in the period are unscored
        lab[a:b][lab[a:b] == "X"] = "X"
    return lab


def mask_ball(arrays, frac=0.4, mean_gap=50, seed=0):
    """Copy of arrays with the ball removed in random gaps (geometric lengths) on ~frac of frames."""
    rng = np.random.RandomState(seed)
    n = len(arrays["bx"])
    mask = np.zeros(n, bool)
    while mask.mean() < frac:
        s = rng.randint(0, n)
        mask[s:s + 1 + rng.geometric(1.0 / mean_gap)] = True
    out = dict(arrays)
    out["bx"], out["by"] = arrays["bx"].copy(), arrays["by"].copy()
    out["bx"][mask] = np.nan
    out["by"][mask] = np.nan
    return out


def synthetic_tags(events):
    """Smart-Tagging-like set-piece windows from SET PIECE events (optimistic: exact timing)."""
    tags = []
    for _, e in events[events["Type"] == "SET PIECE"].iterrows():
        f = int(e["Start Frame"]) - 1
        tags.append({"code": "SET PIECES", "start_frame": f - int(3.4 * FPS),
                     "end_frame": f + int(1.6 * FPS), "tags": {}})
    return tags
