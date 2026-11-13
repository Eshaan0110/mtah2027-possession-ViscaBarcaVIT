# ViscaBarcaVIT: ball possession from tactical tracking (FC Barcelona vs Athletic Club)

Team: ViscaBarcaVIT. Phase 1, submission v1.

## 1. Summary

For every video frame we label the match H (home in possession, ball alive), A (away in possession, ball
alive) or D (dead ball). The method is a small classical pipeline with no trained model:

1. rebuild the ball path from the tactical tracking data,
2. detect *control events*, the moments a team demonstrably controls the ball,
3. compute per-frame *dead-ball evidence*,
4. decode the three states with a Viterbi decoder that enforces minimum durations.

Possession is deliberately sticky: it changes team only when the other team demonstrably controls the ball.
This matches the scoring, where one-to-one possession matching rewards clean, correctly bounded spells.
All constants are football priors (distances, speeds, durations). Nothing is fitted to this match.

## 2. Assets and data

| Asset | Use |
|---|---|
| Tactical tracking `BAR-ATH_tactical_atd.txt` (25 fps, 46 player slots + ball) | all player and ball positions and speeds |
| Tactical `atd.xml` | slot to team mapping (slots 1-24 Barcelona/home, 25-46 Athletic/away, parsed in code) |
| Tactical Smart Tagging XML | **only the `SET PIECES` windows' timing**, to locate restarts. The `Team` tag is never used (throw-ins have it N/A) |
| `games.py`, `starter/load_smart_tagging.py`, FAQ-stated half boundaries | frame ranges; frames 110808-113741 are graded and all D (pre-kick-off), forced to D |
| Metrica open sample data (games 1-2), https://github.com/metrica-sports/sample-data | **external validation only** (see 5) |

Not used: broadcast files, homographies, video. Pitch is 105.3 x 68 m; tracking coordinates are normalised
0-1 and converted to metres; `video_frame = frame_id - 1`. Speed channels are m/s (DATA.md), although the
XML labels them kph.

**Training data:** none. No model is trained and no labels (hand-made or otherwise) are used.
The only external data is the Metrica sample, used to compare a handful of settings; its labels are derived
from its event data and are approximate.

## 3. Method

### 3.1 Ball reconstruction
The ball is visible in about 59% of graded frames. Gaps up to 12 frames (0.5 s) are linearly interpolated;
longer gaps stay unknown. Speed and acceleration come from central differences. A missing ball is never
treated as evidence of anything.

### 3.2 Control events (the only team evidence)
- The nearest player within 1.5 m of the ball is a candidate. A run of frames where the same team is nearest
  is a *touch* (gaps up to 5 frames merged, at least 3 frames).
- A touch of at least 12 frames (0.5 s) is a control event.
- Two touches by the same team within 3 s with no opposing touch in between also form a control event
  (a pass between teammates).
- Events are backdated to the first touch of the stretch.

### 3.3 Dead-ball evidence
Each frame gets a log-likelihood ratio L that the ball is dead:
- ball outside the pitch lines (strong),
- collective slowdown: median player speed below 1.1 m/s (walking pace),
- ball still while players are slow,
- live-play evidence (ball moving above 2 m/s inside the pitch, or a control event) pushes towards alive,
- Smart Tagging `SET PIECES` windows give a restart frame (first frame with ball speed above 4 m/s in the
  window, otherwise the window midpoint); the stretch from the last live-play frame before it (at most
  90 s earlier) to the restart is dead.

### 3.4 Decoder
A three-state (H, A, D) Viterbi decoder. Each state is a chain of sub-states so it cannot be left before a
2 s minimum (alive and dead). An H<->A switch costs 12 nats, entering or leaving D costs 6 nats. Control
evidence adds 1 nat per frame for the controlling team; frames without evidence are neutral, so possession
persists through loose balls and missing ball. No `X` is output on graded frames. Each half is decoded
separately.

## 4. Possession rules (as implemented)
- Alive possession belongs to the team that last demonstrably controlled the ball.
- It changes only on a control event by the other team, or after a dead-ball period.
- A dead ball is labelled D from the last live-play frame before a detected restart/stoppage until the restart.
- Frames 110808-113741 are D.

## 5. Evaluation and tuning

- **No tuning on the public window.** The 5-minute public ground truth was used for correctness checks, and
  all constants were set a priori from football priors. One bug was fixed from the units (a speed threshold
  written for km/h applied to m/s data), not from the score.
- Public window, final v1: macro F1 .708, spell F1 .457 (18 predicted possessions vs 17), against
  `baseline_smooth50` .621 / .326. With 17 possessions the noise is about 0.04, so this cannot rank settings.
- **External validation** on Metrica games 1-2 with the ball masked about 40% (random gaps, mean 50 frames,
  two seeds): macro F1 .746, spell F1 .485. Labels there are derived from events (alive team = team of the
  latest pass/shot/recovery/set piece; dead from ball-out/foul/goal until the next set piece), so they are
  approximate. The sweep there chose the 12-frame control minimum (25 to 12 frames raised spell F1 .456 to .485).
- Baselines regenerate byte-identical to the shipped ones; the submission is deterministic (identical file
  hash on rerun).

## 6. Limitations
- Control is proximity-based on tracking only; there is no ball height, so aerial duels and long passes are weak.
- Long stoppages where players walk with the ball (e.g. a ball carried slowly during a break) can be read as
  live play, and stoppages with almost no ball visibility are missed.
- The model under-segments: about 230 predicted possessions against about 330 derived ones on the external
  games, so short flurries of possession are merged.
- Dead-ball onset is heuristic; throw-in restarts depend on the accuracy of the Smart Tagging windows.
- The public window is too small to measure improvements.

## 7. Disclosures
- The tactical `atd.xml` and the first 2,000 lines of the Smart Tagging XML were auto-attached to a hosted
  LLM coding assistant's conversation by an @-mention, and raw rows of the public ground truth and aggregate
  slices of the tracking data were later displayed to it during error analysis. The pipeline itself reads
  the data only through code. No values were hard-coded from those views.
- One threshold change (collective-slowdown 1.1 to 1.3 m/s) was checked against the public window and
  **reverted**; v1 uses 1.1, which was set before any public-window look.
- An LLM coding assistant was used to write and debug the code.

## 8. Reproduction

```bash
pip install -r requirements-solution.txt     # numpy==2.4.2, pandas==2.3.3, scipy==1.18.0
# place the provided data under data/ as described in data/README.md
PYTHONPATH=src python -m possession.run --out evaluation/results/ViscaBarcaVIT.csv
python evaluation/validate_submission.py evaluation/results/ViscaBarcaVIT.csv
```

The command regenerates the submission exactly (199,319 rows). External validation:
`PYTHONPATH=src python tools/validate_external.py` (needs the Metrica sample under `cache/external/metrica/data`).
Code layout: `src/possession/{loader,ball,control,deadball,decoder,pipeline,run,external}.py`.
