# NOTES (for the technical document)

## Setup
- Data actually available: tactical ATD .txt only (+ broadcast ATD xml and homography, not used).
  Missing: tactical atd.xml (slot->team), tactical Smart Tagging, tactical homography, videos.
- Loader (src/possession/loader.py): ATD -> metres, indexed frame_id-1, cached to cache/ (gitignored).
  Verified: 199,319 frames, ball visible 56.9%, 46 slots. Pinned deps in requirements-solution.txt.
- Scorer reproduces README board on public window (smooth125 .4242, smooth50 .3256, smooth25 .2868, raw .0378).

## Tried and dropped: inferring slot->team without the XML (aggregates only, validated on kickoff geometry)
- Spectral split on correlation of x-offset from all-player mean: within-vs-across corr gap ~0.03. No signal.
- Contiguous slot split as in the broadcast XML (slots 1-26 home / 27-51 away): best k=24 on tactical
  agrees with 2nd-half kickoff sides for only 18/22 slots; not supported.
- Ball-handover graph (consecutive nearest-to-ball slots within 1.5 m) spectral split: no eigen-gap,
  kickoff agreement 12/19 and 16/20. Not reliable enough.
Conclusion: team assignment needs the tactical atd.xml; H/A labels are meaningless without it.
