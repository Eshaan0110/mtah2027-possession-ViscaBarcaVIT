#!/bin/bash
# One-at-a-time sweep on the external (Metrica) validation set. Output: cache/sweep2.txt
export PYTHONPATH=src
out=cache/sweep2.txt; : > $out
run() { echo "== $*" >> $out; python tools/validate_external.py --tags none --set "$@" 2>&1 | grep -E "^game|MEAN" >> $out; }
for v in 1.0 2.0 2.5; do run control.R_CONTROL=$v; done
for v in 25 75; do run decoder.MIN_ALIVE=$v; done
for v in 6 9 12 15; do run control.MIN_CONTROL=$v; done
for v in 100 150; do run control.PASS_GAP=$v; done
run control.MIN_CONTROL=12 control.PASS_GAP=125
echo done >> $out
