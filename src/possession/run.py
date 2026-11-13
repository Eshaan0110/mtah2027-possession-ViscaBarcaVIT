"""Single entry point:  PYTHONPATH=src python -m possession.run --out evaluation/results/<team>.csv

Pipeline: load ATD -> reconstruct ball -> control events -> dead-ball evidence -> Viterbi per half.
Only aggregates are printed (counts, shares), never frame-level values.
"""
import argparse
import os
import sys

import numpy as np

from . import loader, pipeline

ROOT = loader.ROOT
sys.path.insert(0, ROOT)
from games import half_video_blocks, GAME  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

PRE_KICKOFF_DEAD = (110808, 113741)  # graded, all D: FAQ.md (second half is graded before kick-off)


def build_labels():
    arrays = loader.load()
    team = loader.team_of_slot()
    tags = load_smart_tagging.load()
    halves = half_video_blocks(GAME)
    blocks = [(f, l, f if h == 1 else PRE_KICKOFF_DEAD[1] + 1) for h, f, l in halves]
    labels, info = pipeline.predict(arrays, team, tags, blocks, forced_dead=[PRE_KICKOFF_DEAD])
    graded = np.zeros(len(labels), bool)
    for _, first, last in halves:
        graded[first:last + 1] = True
    print("ball known %.1f%% of graded | touches %d | control events %d | restarts %d"
          % (100 * info["known"][graded].mean(), info["touches"], info["events"], info["restarts"]))
    return labels, graded


def summarize(labels, graded):
    g = labels[graded]
    print("graded share  H %.1f%%  A %.1f%%  D %.1f%%  X %.1f%%" % tuple(
        100 * np.mean(g == c) for c in "HADX"))
    n_poss = 0
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        alive = np.isin(seg, ["H", "A"])
        n_poss += int((alive & ~np.r_[False, alive[:-1] & (seg[1:] == seg[:-1])]).sum())
    print("possessions %d, mean duration %.1f s" % (n_poss, np.isin(g, ["H", "A"]).sum() / 25 / max(n_poss, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    labels, graded = build_labels()
    summarize(labels, graded)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines("%d,%s\n" % (i, l) for i, l in enumerate(labels))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
