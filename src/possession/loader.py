"""Tactical ATD -> numpy arrays in metres, indexed by video frame (frame_id - 1).

Raw line: "<frame_id>:<x,y,s>;...(46 slots):<bx,by,s>", coordinates normalised 0-1.
Arrays are cached under cache/ (gitignored) because the parse takes about a minute.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
N_SLOTS = 46


def _parse(raw_path, n_frames, pitch):
    length, width = pitch
    px = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    py = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    ps = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                v = group.split(",")
                px[i, slot], py[i, slot], ps[i, slot] = float(v[0]), float(v[1]), float(v[2])
            if parts[2][0] != "N":
                v = parts[2].split(",")
                ball[i] = float(v[0]), float(v[1]), float(v[2])
    px *= length
    py *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return {"px": px, "py": py, "ps": ps, "bx": ball[:, 0], "by": ball[:, 1], "bs": ball[:, 2]}


def load(game=GAME, data_dir=None, use_cache=True):
    """dict of arrays: px, py, ps (n, 46) in metres / source speed units; bx, by, bs (n,)."""
    raw_path = require(["atd_raw"], game, data_dir)[0]
    n_frames = source_frames("tactical", game)
    cache_file = os.path.join(CACHE, "atd_%s.npz" % game)
    if use_cache and os.path.isfile(cache_file):
        with np.load(cache_file) as z:
            return {k: z[k] for k in z.files}
    arrays = _parse(raw_path, n_frames, GAMES[game]["pitch"])
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cache_file, **arrays)
    return arrays


def team_of_slot(game=GAME, data_dir=None):
    """0 = home, 1 = away per track slot, parsed from the ATD metadata XML (first team is home)."""
    import xml.etree.ElementTree as ET
    meta = require(["atd_meta"], game, data_dir)[0]
    root = ET.parse(meta).getroot()
    home = [t.get("id") for t in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], dtype=np.int8)


if __name__ == "__main__":
    a = load()
    n = len(a["bx"])
    print("frames", n, "ball visible %.1f%%" % (100 * np.isfinite(a["bx"]).mean()),
          "mean tracked/frame %.1f" % np.isfinite(a["px"]).sum(1).mean())
