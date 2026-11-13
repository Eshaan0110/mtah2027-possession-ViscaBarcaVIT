"""Control events: when a team demonstrably controls the ball. Only these may flip possession."""
import numpy as np

R_CONTROL = 1.5      # m: a player this close to the ball is a candidate controller
MERGE_GAP = 5        # frames: tracking dropouts shorter than this do not break a contact
MIN_TOUCH = 3        # frames: shortest contact that counts as a touch
MIN_CONTROL = 12     # frames (0.5 s): one team close to the ball -> control. Chosen on external Metrica games (25 -> 12: spellF1 .456 -> .485)
PASS_GAP = 75        # frames (3 s): two touches by one team this close together -> control


def nearest_team(arrays, team, ball):
    """(dist, team) of the closest tracked player to the (reconstructed) ball, per frame."""
    d = np.hypot(arrays["px"] - ball["x"][:, None], arrays["py"] - ball["y"][:, None])
    d = np.where(np.isfinite(d), d, np.inf).astype(np.float32)
    dh = d[:, team == 0].min(1)
    da = d[:, team == 1].min(1)
    dist = np.minimum(dh, da)
    who = np.where(dh <= da, 0, 1)
    dist[~ball["known"]] = np.inf
    return dist, who.astype(np.int8)


def _runs(mask, gap):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    br = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.concatenate(([idx[0]], idx[br + 1]))
    ends = np.concatenate((idx[br], [idx[-1]]))
    return list(zip(starts.tolist(), (ends + 1).tolist()))


def control_events(dist, who, r=None):
    """[(start, stop, team)] control events, start backdated to the first touch of the stretch."""
    near = dist <= (R_CONTROL if r is None else r)
    touches = []
    for team in (0, 1):
        for a, b in _runs(near & (who == team), MERGE_GAP):
            if b - a >= MIN_TOUCH:
                touches.append((a, b, team))
    touches.sort()
    events = [(a, b, t) for a, b, t in touches if b - a >= MIN_CONTROL]
    for (a1, b1, t1), (a2, b2, t2) in zip(touches[:-1], touches[1:]):
        if t1 == t2 and a2 - b1 <= PASS_GAP and (b1 - a1 < MIN_CONTROL or b2 - a2 < MIN_CONTROL):
            events.append((a1, b2, t1))
    return sorted(events), touches


def evidence(events, n):
    """Per-frame team evidence: -1 none, 0 home, 1 away. Longer events win overlaps."""
    ev = np.full(n, -1, dtype=np.int8)
    for a, b, t in sorted(events, key=lambda e: e[1] - e[0]):
        ev[a:b] = t
    return ev
