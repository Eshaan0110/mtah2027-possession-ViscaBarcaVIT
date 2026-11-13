"""Three-state (H, A, D) Viterbi with minimum durations, replacing majority-vote smoothing.

Each state is a chain of sub-states so it cannot be left before its minimum duration. Team evidence
comes only from control events; an H<->A switch costs SWITCH nats, entering/leaving D costs
DEAD_SWITCH. Frames without evidence are neutral, so possession is sticky through loose balls and
a missing ball.
"""
import numpy as np

H, A, D = 0, 1, 2
W_TEAM = 1.0          # nats per frame of control evidence
SWITCH = 12.0
DEAD_SWITCH = 6.0
MIN_ALIVE = 50        # frames (2 s)
MIN_DEAD = 50         # frames (2 s)
NEG = -1e18


def decode(team_ev, L, min_alive=None, min_dead=None):
    """team_ev: int8 (-1/0/1), L: dead log-likelihood ratio. Returns int8 states 0/1/2."""
    min_alive = MIN_ALIVE if min_alive is None else min_alive
    min_dead = MIN_DEAD if min_dead is None else min_dead
    n = len(L)
    m = (min_alive, min_alive, min_dead)
    M = max(m)
    last = [x - 1 for x in m]
    em = np.zeros((n, 3))
    em[:, H] = np.where(team_ev == 0, W_TEAM, np.where(team_ev == 1, -W_TEAM, 0.0)) - L / 2
    em[:, A] = np.where(team_ev == 1, W_TEAM, np.where(team_ev == 0, -W_TEAM, 0.0)) - L / 2
    em[:, D] = L / 2
    cost = np.full((3, 3), SWITCH)
    cost[D, :] = DEAD_SWITCH
    cost[:, D] = DEAD_SWITCH
    np.fill_diagonal(cost, 0.0)

    V = np.full((3, M), NEG)
    V[:, 0] = em[0]
    bp_entry = np.zeros((n, 3), dtype=np.int8)
    bp_stay = np.zeros((n, 3), dtype=bool)
    for t in range(1, n):
        new = np.full((3, M), NEG)
        exits = [V[p, last[p]] for p in range(3)]
        for s in range(3):
            new[s, 1:m[s]] = V[s, :m[s] - 1]
            if V[s, last[s]] > V[s, last[s] - 1]:
                new[s, last[s]] = V[s, last[s]]
                bp_stay[t, s] = True
            best, arg = NEG, 0
            for p in range(3):
                if p != s and exits[p] - cost[p, s] > best:
                    best, arg = exits[p] - cost[p, s], p
            new[s, 0] = best
            bp_entry[t, s] = arg
        new += em[t][:, None]
        V = new

    s, k = np.unravel_index(np.argmax(V), V.shape)
    s, k = int(s), int(k)
    out = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = s
        if t == 0:
            break
        if k == 0:
            s = int(bp_entry[t, s])
            k = last[s]
        elif k == last[s] and bp_stay[t, s]:
            pass
        else:
            k -= 1
    return out
