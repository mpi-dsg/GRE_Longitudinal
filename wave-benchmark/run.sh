#!/bin/sh
# Experiments, in priority order. Each tier is independently useful; stop whenever.
# Completed runs are skipped, so this is safe to interrupt and restart.
#
#   DATA=/path/to/sosd sh run.sh          # runs every tier
#   DATA=/path/to/sosd TIER=1 sh run.sh   # just tier 1
#
# Expects SOSD files named books_800M_uint64 and osm_cellids_800M_uint64.
set -e
: "${DATA:?set DATA to the directory holding the SOSD binaries}"
TIER="${TIER:-all}"
mkdir -p results
B=100000000; TOT=400000000; BS=12500000

run() {  # name index threads seed dataset extra...
  n=$1; idx=$2; T=$3; seed=$4; ds=$5; shift 5
  f="results/${n}_${ds}_t${T}_s${seed}.csv"
  [ -s "$f" ] && { echo "skip $f"; return; }
  echo "run  $f"
  ./bench "$idx" $B $TOT $BS "$seed" "$T" x "$@" "data=$DATA/${ds}_800M_uint64" \
      2>>results/verify.log | grep -E "^x,|^LOCK2," > "$f"
}

# TIER 1 - independent reproduction. This is the one that matters most: same
# configuration, different machine. Roughly 90 minutes.
if [ "$TIER" = 1 ] || [ "$TIER" = all ]; then
  for T in 1 16; do
    run alexol     alexol    $T 1866 books
    run alexolstag alexol    $T 1866 books stagger
    run sali       sali      $T 1866 books
    run artolc     artolc    $T 1866 books
    run btreebulk  btreebulk $T 1866 books bfill=0.70
    run lippol     lippol    $T 1866 books
  done
  echo TIER1_DONE
fi

# TIER 2 - balanced read/write. Our measurements are insert-only; the published
# runs are 50/50. This closes the largest gap in the paper, and it also answers
# whether the waves are visible on the read path at all (the CSV times reads
# separately). Roughly 60 minutes.
if [ "$TIER" = 2 ] || [ "$TIER" = all ]; then
  for T in 1 16; do
    run alexol_rw50     alexol $T 1866 books readpct=50
    run alexolstag_rw50 alexol $T 1866 books stagger readpct=50
    run sali_rw50       sali   $T 1866 books readpct=50
    run artolc_rw50     artolc $T 1866 books readpct=50
  done
  echo TIER2_DONE
fi

# TIER 3 - hard key distribution. OSM is far less uniform than books, so nodes
# fill at different rates and the cohort may decorrelate on its own. If the waves
# survive OSM they survive anything. Roughly 90 minutes.
if [ "$TIER" = 3 ] || [ "$TIER" = all ]; then
  for T in 1 16; do
    run alexol     alexol    $T 1866 osm_cellids
    run alexolstag alexol    $T 1866 osm_cellids stagger
    run sali       sali      $T 1866 osm_cellids
    run artolc     artolc    $T 1866 osm_cellids
  done
  echo TIER3_DONE
fi

# TIER 4 - more seeds, for error bars on tier 1.
if [ "$TIER" = 4 ] || [ "$TIER" = all ]; then
  for seed in 5 72; do
    for T in 1 16; do
      run alexol     alexol $T $seed books
      run alexolstag alexol $T $seed books stagger
      run sali       sali   $T $seed books
    done
  done
  echo TIER4_DONE
fi

# TIER 5 - the design (paper Section 5): side buffer and background expansion against
# unmodified ALEX-OL and the 2048-entry node bound, 400M books keys, one binary for every
# arm. Runs verify every key at the end (FULLVERIFY in results/verify.log). Roughly two hours.
if [ "$TIER" = 5 ] || [ "$TIER" = all ]; then
  for T in 16 1; do
    run d_base    alexol    $T 1866 books
    run d_side    alexol    $T 1866 books side
    run d_bgside  alexol    $T 1866 books bg side
    run d_bgside2 alexol    $T 1866 books bg side bgthreads=2
    run d_ns2048  alexsized $T 1866 books nodebytes=32768
  done
  echo TIER5_DONE
fi
