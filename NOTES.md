# NOTES (for the technical document)

## Data actually used
- tactical ATD .txt + atd.xml (slot->team, first listed team = home) + Smart Tagging XML (set-piece timing only;
  never the Team tag, throw-ins have N/A). Homography, videos, broadcast files: present but unused.
- The tactical atd.xml and the first 2,000 lines of the Smart Tagging XML were auto-attached to the LLM
  conversation by an @-mention (not by the pipeline). No values were taken from that text; the pipeline
  reads the files only through code. Disclose to the team / check the terms.
- ATD speed values are m/s (DATA.md), although the XML labels the channel kph.

## Pipeline (python -m possession.run, PYTHONPATH=src)
1. loader: ATD -> metres, frame_id-1, cached in cache/ (gitignored).
2. ball: gaps <= 12 frames linearly interpolated, longer = unknown; central-difference speed/accel.
3. control: nearest-player-within-1.5 m runs (gaps<=5 frames merged); >=1 s one team = control;
   two touches by one team within 3 s with no opposing touch = control; backdated to first touch.
4. deadball: LLR per frame. Out-of-bounds ball, collective slowdown (median player speed < 1.1 m/s),
   ball still, live-play evidence (ball moving in bounds / control). Missing ball = no evidence.
   Smart Tagging SET PIECES windows -> restart frame = first frame ball speed > 4 m/s in window
   (fallback window midpoint); dead stretch = last live-play frame before restart (<=90 s) to restart.
   Frames 110808-113741 forced D (FAQ.md).
5. decoder: 3-state Viterbi, min duration 2 s alive/dead, switch cost 12 nats H<->A, 6 nats via D.
   Team evidence only from control events, so possession is sticky. No X emitted on graded frames.
All thresholds are football priors chosen a priori; none tuned on the public window.

## Tried and dropped
- Inferring slot->team from tracking alone (before the XML arrived): correlation spectral split (no signal),
  contiguous slot split (18/22 vs kickoff geometry), ball-handover graph (12/19, 16/20). Dropped; XML used.
- Bug found and fixed from units, not score: slowdown threshold was in "kph" but speeds are m/s.

## Scores
- Baselines regenerate byte-identical to shipped (smooth50 full-match spellF1 0.2433, per README).
- Public window sanity only (not tuned): v1 first run (unit bug) macro .7284 / spell .4588; after unit fix
  macro .7268 / spell .5419, 14 possessions mean 20.1 s vs 17 / 14.2 s truth.

## Limitations / next
- Control uses tracking-only proximity, ball height unknown; aerial duels and long passes are weak spots.
- Restart timing and dead onset from heuristics; throw-in restarts depend on tag accuracy.
- Not yet done: external validation set (Metrica sample data, DFL open data) for tuning — need your OK to download.

- SLOW_MS 1.1 -> 1.3 (walking-pace threshold; smoothed median player speed is ~2.2 in live play and ~1.0-1.2 in stoppages). External Metrica mean macro .746->.756, spell .485->.486; public macro .708->.744, spell .457->.456.
