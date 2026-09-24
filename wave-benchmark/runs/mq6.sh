#!/bin/bash
# Final run set on the final binary (all fixes). A run whose verification fails exits
# nonzero and is kept only as .failN. Order: most important first.
cd "$(dirname "$0")"
B=100000000; TOT=400000000; BS=12500000
BK=data/books_800M_uint64; OSM=data/osm_cellids_800M_uint64
flags() { case $1 in base|xindex|finedex) echo "";; side) echo side;; bgside) echo "bg side";;
  bgside2) echo "bg side bgthreads=2";; bgside4) echo "bg side bgthreads=4";; bg) echo bg;;
  ns2048) echo "nodebytes=32768";; esac; }
idx() { case $1 in ns2048) echo alexsized;; xindex|finedex) echo $1;; *) echo alexol;; esac; }
r() { local f=$1 to=$2 arm=$3; shift 3; [ -s "$f" ] && return
  timeout $to ./bench $(idx $arm) "$@" $(flags $arm) 2>>mq6_verify.log | grep --line-buffered -E "^x,|^LOCK2,|^LOCK3," > "$f.tmp"
  local rc=${PIPESTATUS[0]}; echo "RC $rc $f" >> mq6_verify.log
  [ $rc = 0 ] && mv "$f.tmp" "$f" || mv "$f.tmp" "$f.fail$rc"; }
mkdir -p r6/b400 r6/b400ls r6/x400 r6/o400 r6/m4 r6/m4mix r6/m4ls r6/m4soft r6/sat r6/eq
for seed in 1866 5 72; do
  for arm in base side bgside bgside2 bgside4 ns2048; do r r6/b400/${arm}_books_t16_s${seed}.csv 2400 $arm $B $TOT $BS $seed 16 x data=$BK; done
  r r6/eq/bgside4_books_t12_s${seed}.csv 2400 bgside4 $B $TOT $BS $seed 12 x data=$BK
  for arm in base side bgside ns2048; do r r6/b400/${arm}_books_t1_s${seed}.csv 2400 $arm $B $TOT $BS $seed 1 x data=$BK; done
done
echo B400_DONE
for seed in 1866 5 72; do
  for arm in base side bgside bgside4; do r r6/o400/${arm}_osm_t16_s${seed}.csv 2400 $arm $B $TOT $BS $seed 16 x data=$OSM; done
  for arm in base bgside; do r r6/o400/${arm}_osm_t1_s${seed}.csv 3000 $arm $B $TOT $BS $seed 1 x data=$OSM; done
done
echo O400_DONE
for rep in 1 2 3; do for seed in 1866 5 72; do for T in 1 2 4 8 16; do for arm in base ns2048 bg side bgside bgside2 bgside4; do
  r r6/m4/${arm}_t${T}_s${seed}_r${rep}.csv 120 $arm 1000000 4000000 100000 $seed $T x; done; done; done; done
for rep in 1 2 3; do for seed in 1866 5 72; do for T in 1 16; do for arm in base ns2048 side bgside bgside4; do
  r r6/m4mix/${arm}_t${T}_s${seed}_r${rep}.csv 120 $arm 1000000 4000000 100000 $seed $T x readpct=50; done; done; done; done
for seed in 1866 5 72; do for T in 1 16; do for arm in base side bgside bgside4 bg; do
  r r6/m4ls/${arm}_t${T}_s${seed}.csv 120 $arm 1000000 4000000 100000 $seed $T x lockstats; done; done; done
for seed in 1866 5 72; do for T in 1 16; do for sf in 0.85 0.9 0.97; do
  r r6/m4soft/sf${sf}_t${T}_s${seed}.csv 120 bgside 1000000 4000000 100000 $seed $T x bgsoft=$sf; done; done; done
echo M4_DONE
for T in 16 1; do for arm in base bgside bgside4 ns2048; do
  [ $T = 1 ] && [ $arm = bgside4 ] && continue
  r r6/x400/${arm}_books_t${T}_s1866.csv 2400 $arm $B $TOT $BS 1866 $T x data=$BK readpct=50; done; done
for arm in base side bgside bgside4; do r r6/b400ls/${arm}_books_t16_s1866.csv 2400 $arm $B $TOT $BS 1866 16 x data=$BK lockstats; done
for arm in base bgside; do r r6/b400ls/${arm}_books_t1_s1866.csv 2400 $arm $B $TOT $BS 1866 1 x data=$BK lockstats; done
for seed in 1866 5 72; do for arm in base bgside4; do r r6/sat/${arm}_books_t32_s${seed}.csv 2400 $arm $B $TOT $BS $seed 32 x data=$BK; done; done
echo MQ6_DONE
# osm comparison (XIndex and FINEdex), after everything else. XIndex hangs on some inputs,
# so it gets a shorter timeout.
mkdir -p r6/ocmp
for seed in 5 72; do r r6/ocmp/finedex_osm_t16_s${seed}.csv 2700 finedex $B $TOT $BS $seed 16 x data=$OSM; done
for seed in 5 72; do r r6/ocmp/xindex_osm_t16_s${seed}.csv 1500 xindex $B $TOT $BS $seed 16 x data=$OSM; done
r r6/ocmp/finedex_osm_t1_s1866.csv 3600 finedex $B $TOT $BS 1866 1 x data=$OSM
echo OCMP_DONE
