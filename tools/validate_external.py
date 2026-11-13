"""Score the pipeline on Metrica sample games 1-2 with the ball masked ~40% of the time.

    PYTHONPATH=src python tools/validate_external.py [--tags none|synthetic] [--set NAME=VALUE ...]

--set overrides module constants (e.g. control.R_CONTROL=1.2) so settings can be compared on this
EXTERNAL data. Prints aggregates only.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from possession import control, deadball, decoder, external, pipeline  # noqa: E402
import possession_metrics as pm  # noqa: E402

MODS = {"control": control, "deadball": deadball, "decoder": decoder}


def run_game(g, tags_mode, seeds=(0, 1)):
    arrays, team, period, events = external.load_game(g)
    n = len(period)
    gt = external.derive_labels(events, period, n)
    gt_codes = pm.label_codes(gt)
    blocks = []
    for p in (1, 2):
        r = np.flatnonzero(period == p)
        blocks.append((r[0], r[-1], r[0]))
    tags = external.synthetic_tags(events) if tags_mode == "synthetic" else []
    res = []
    for seed in seeds:
        masked = external.mask_ball(arrays, seed=seed)
        labels, _ = pipeline.predict(masked, team, tags, blocks)
        labels[gt == "X"] = "X"
        r = pm.evaluate(gt_codes, pm.label_codes(labels))
        res.append((r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]))
    return np.array(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="none", choices=["none", "synthetic"])
    ap.add_argument("--games", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    for item in a.set:
        key, val = item.split("=")
        mod, name = key.split(".")
        setattr(MODS[mod], name, type(getattr(MODS[mod], name))(float(val)))
    allr = []
    for g in a.games:
        r = run_game(g, a.tags)
        allr.append(r)
        m = r.mean(0)
        print("game %d  macroF1 %.3f  spellF1 %.3f  poss pred %.0f vs derived-gt %.0f" % (g, m[0], m[1], m[2], m[3]))
    m = np.concatenate(allr).mean(0)
    print("MEAN    macroF1 %.3f  spellF1 %.3f" % (m[0], m[1]))


if __name__ == "__main__":
    main()
