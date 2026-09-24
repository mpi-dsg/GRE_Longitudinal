#!/bin/bash
# Experiments requested by the 2026-09-24 review panel. Same source as bgx6 (final binary)
# plus driver options: recent reads, insert order, per-batch CPU time. A run whose
# verification fails exits nonzero and is kept only as .failN. Most important first.
cd "$(dirname "$0")"
B=100000000; TOT=400000000; BS=12500000
BK=data/books_800M_uint64
flags() { case $1 in base|xindex|finedex) echo "";; side) echo side;; bgside) echo "bg side";;
  bgside2) echo "bg side bgthreads=2";; bgside4) echo "bg side bgthreads=4";; bg) echo bg;; esac; }
idx() { case $1 in xindex|finedex) echo $1;; *) echo alexol;; esac; }
r() { local f=$1 to=$2 arm=$3; shift 3; [ -s "$f" ] && return
  timeout $to ./bench $(idx $arm) "$@" $(flags $arm) 2>>mq7_verify.log | grep --line-buffered -E "^x,|^CPU,|^LOCK2,|^LOCK3," > "$f.tmp"
  local rc=${PIPESTATUS[0]}; echo "RC $rc $f" >> mq7_verify.log
  [ $rc = 0 ] && mv "$f.tmp" "$f" || mv "$f.tmp" "$f.fail$rc"; }
mkdir -p r7/cmp r7/cmpmix r7/cpu r7/recent r7/skew r7/skew4m
# 1. Comparison table with three seeds for every arm (XIndex and FINEdex were one or two).
for seed in 5 72 1866; do for arm in xindex finedex; do
  r r7/cmp/${arm}_books_t16_s${seed}.csv 2400 $arm $B $TOT $BS $seed 16 x data=$BK; done; done
for seed in 1866 5 72; do for arm in base bgside4 xindex finedex; do
  r r7/cmpmix/${arm}_books_t16_s${seed}.csv 2400 $arm $B $TOT $BS $seed 16 x data=$BK readpct=50; done; done
echo CMP_DONE
# 2. CPU time at equal and unequal core budgets.
for seed in 1866 5 72; do
  r r7/cpu/base_books_t16_s${seed}.csv 2400 base $B $TOT $BS $seed 16 x data=$BK
  r r7/cpu/bgside4_books_t16_s${seed}.csv 2400 bgside4 $B $TOT $BS $seed 16 x data=$BK
  r r7/cpu/bgside4_books_t12_s${seed}.csv 2400 bgside4 $B $TOT $BS $seed 12 x data=$BK
  r r7/cpu/base_books_t1_s${seed}.csv 2400 base $B $TOT $BS $seed 1 x data=$BK
  r r7/cpu/bgside_books_t1_s${seed}.csv 2400 bgside $B $TOT $BS $seed 1 x data=$BK
done
echo CPU_DONE
# 3. Lookups of recently inserted keys, which reach nodes under rebuild and their side buffers.
for seed in 1866 5 72; do for arm in base bgside4 side bgside; do
  [ $seed != 1866 ] && { [ $arm = side ] || [ $arm = bgside ]; } && continue
  r r7/recent/${arm}_books_t16_s${seed}.csv 2400 $arm $B $TOT $BS $seed 16 x data=$BK readpct=50 recent; done; done
r r7/recent/bgside4ls_books_t16_s1866.csv 2400 bgside4 $B $TOT $BS 1866 16 x data=$BK readpct=50 recent lockstats
r r7/recent/sidels_books_t16_s1866.csv 2400 side $B $TOT $BS 1866 16 x data=$BK readpct=50 recent lockstats
echo RECENT_DONE
# 4. Skewed and sorted insert orders.
for seed in 1866 5 72; do for T in 1 16; do for arm in base bgside bgside4; do for o in zipf sorted; do
  [ $T = 1 ] && [ $arm = bgside4 ] && continue
  r r7/skew4m/${arm}_${o}_t${T}_s${seed}.csv 300 $arm 1000000 4000000 100000 $seed $T x order=$o; done; done; done; done
for seed in 1866 5 72; do for arm in base bgside4; do
  r r7/skew/${arm}_zipf_t16_s${seed}.csv 2400 $arm $B $TOT $BS $seed 16 x data=$BK order=zipf; done; done
for arm in base bgside; do r r7/skew/${arm}_zipf_t1_s1866.csv 2400 $arm $B $TOT $BS 1866 1 x data=$BK order=zipf; done
for arm in base bgside4; do r r7/skew/${arm}_sorted_t16_s1866.csv 2400 $arm $B $TOT $BS 1866 16 x data=$BK order=sorted; done
for arm in base bgside; do r r7/skew/${arm}_sorted_t1_s1866.csv 2400 $arm $B $TOT $BS 1866 1 x data=$BK order=sorted; done
echo SKEW_DONE
# 5. XIndex and FINEdex at one thread, seeds 5 and 72 (slow; last).
for seed in 5 72; do for arm in xindex finedex; do
  r r7/cmp/${arm}_books_t1_s${seed}.csv 3600 $arm $B $TOT $BS $seed 1 x data=$BK; done; done
echo MQ7_DONE
