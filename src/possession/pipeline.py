"""Core prediction shared by the competition run and the external validation harness."""
import numpy as np

from . import ball as ballmod
from . import control, deadball, decoder

LETTERS = np.array(["H", "A", "D"])


def predict(arrays, team, tags, blocks, forced_dead=()):
    """H/A/D/X label array.

    arrays: px, py, ps (n, slots), bx, by (n,) in metres, ps in m/s.
    team:   0/1 per slot.   tags: Smart-Tagging-like dicts (code, start_frame, end_frame).
    blocks: [(first, last, decode_from)] inclusive graded frame ranges; decode starts at decode_from.
    forced_dead: [(first, last)] inclusive ranges labelled D without decoding.
    Returns (labels, info).
    """
    n = len(arrays["bx"])
    ball = ballmod.reconstruct(arrays["bx"], arrays["by"])
    dist, who = control.nearest_team(arrays, team, ball)
    events, touches = control.control_events(dist, who)
    team_ev = control.evidence(events, n)
    L, restarts = deadball.dead_evidence(arrays, ball, team_ev, tags)

    labels = np.full(n, "X", dtype="<U1")
    for first, last, start in blocks:
        a, b = start, last + 1
        labels[a:b] = LETTERS[decoder.decode(team_ev[a:b], L[a:b])]
    for a, b in forced_dead:
        labels[a:b + 1] = "D"
    info = {"known": ball["known"], "touches": len(touches), "events": len(events),
            "restarts": len(restarts)}
    return labels, info
