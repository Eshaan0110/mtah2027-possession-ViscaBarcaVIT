"""Dead-ball evidence: a per-frame log-likelihood ratio L (positive favours DEAD, negative ALIVE).

Cues: ball outside the pitch; Smart Tagging set pieces as restart times (throw-ins carry Team N/A, so
only the timing is used, never the team); GOALS tags followed by a kick-off; collective player
slowdown; ball stationary. A missing ball is NOT evidence of a dead ball (L stays 0).
The dead stretch before a restart is found by walking back to the last frame with live-play
evidence (ball in bounds and moving, or a control event).
"""
import numpy as np

PITCH = (105.3, 68.0)
OOB_MARGIN = 0.3        # m beyond the line before the ball counts as out
MOVING = 2.0            # m/s: ball faster than this inside the pitch is live-play evidence
STILL = 0.7             # m/s
MAX_LOOKBACK = 90 * 25  # frames: how far back from a restart the dead stretch may begin
SLOW_MS = 1.3           # m/s (ATD speeds are m/s, per DATA.md): walking pace, ~4 km/h
RESTART_SPEED = 4.0     # m/s: the ball leaves the spot faster than this at a restart
W_SET, W_OOB, W_SLOW, W_LIVE = 4.0, 3.0, 1.0, 1.5


def _smooth(x, w):
    return np.convolve(np.nan_to_num(x.astype(float)), np.ones(w) / w, mode="same")


def out_of_bounds(ball):
    x, y = ball["x"], ball["y"]
    L, W = PITCH
    return ball["known"] & ((x < -OOB_MARGIN) | (x > L + OOB_MARGIN) | (y < -OOB_MARGIN) | (y > W + OOB_MARGIN))


def restart_frames(tags, ball):
    """One restart frame per Smart Tagging set piece (incl. kick-offs after goals).

    Inside the tagged ~5 s window the restart is the first frame the ball is struck (known, faster
    than RESTART_SPEED); if the ball is not seen the window midpoint is used.
    """
    out = []
    speed = ball["speed"]
    for e in tags:
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        a, b = e["start_frame"], e["end_frame"]
        w = np.flatnonzero(speed[a:b] > RESTART_SPEED)
        out.append(a + int(w[0]) if len(w) else (a + b) // 2)
    return sorted(out)


def dead_evidence(arrays, ball, team_ev, tags):
    n = len(ball["x"])
    L = np.zeros(n)
    oob = out_of_bounds(ball)
    inb = ball["known"] & ~oob
    live = ((inb & (ball["speed"] > MOVING)) | (team_ev >= 0)) & ~oob

    ps = arrays["ps"]
    med = np.full(n, np.nan)
    ok = np.isfinite(ps).sum(1) > 0
    med[ok] = np.nanmedian(ps[ok], axis=1)
    slow = _smooth(med < SLOW_MS, 75) > 0.8
    still = inb & (ball["speed"] < STILL)

    L[live] -= W_LIVE
    L[oob] += W_OOB
    L[slow & ~live] += W_SLOW
    L[still & slow] += W_SLOW

    restarts = restart_frames(tags, ball)
    for r in restarts:
        lo = max(0, r - MAX_LOOKBACK)
        w = np.flatnonzero(live[lo:r])
        onset = lo + int(w[-1]) + 1 if len(w) else max(lo, r - 15 * 25)
        L[onset:r] += W_SET
        L[r:r + 25] -= W_LIVE   # play is live again just after the restart
    return L, restarts
