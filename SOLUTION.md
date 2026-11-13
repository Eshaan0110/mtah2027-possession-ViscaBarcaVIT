# Solution: ball possession from tracking (v1)

Rule-based pipeline plus a 3-state Viterbi decoder. No training, no hand labels, no use of the
public ground truth for fitting or tuning. Details of every choice, and what was tried and dropped,
are in [NOTES.md](NOTES.md).

## Reproduce

Needs the match bundle unpacked in `data/` (or `MTAH_DATA_DIR`): the tactical `*_atd.txt`,
`*_atd.xml` and `*_smart_tagging.xml`. Nothing else is read.

```bash
pip install -r requirements-solution.txt
PYTHONPATH=src python -m possession.run --out evaluation/results/<team_name>.csv
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

(`pip install -e .` also works and removes the need for `PYTHONPATH`.) The first run parses the
tracking file (about a minute) and caches arrays under `cache/`; the output is deterministic.
`data/`, `cache/` and submission CSVs are gitignored.

## Pipeline

1. `loader` - ATD to numpy in metres, indexed by `frame_id - 1`; team per track slot from the XML.
2. `ball` - gaps up to 0.5 s interpolated, longer gaps left unknown; speed and acceleration.
3. `control` - control events: one team within 1.5 m of the ball for 1 s, or two same-team touches
   within 3 s; backdated to the first touch. Only these can flip possession.
4. `deadball` - dead-ball evidence from out-of-bounds, player slowdown, still ball and Smart Tagging
   set-piece timing (never its Team tag). Missing ball is not treated as dead.
5. `decoder` - H/A/D Viterbi with 2 s minimum durations; second-half pre-kick-off frames are `D`.
