#!/bin/bash
# Sorted-insert runs with restart and rebuild counters, to explain why the design does worse there.
# Waits for the skew runs of mq7.sh, stops mq7.sh before its lowest-value tail (XIndex and FINEdex
# at one thread), runs these, then reruns that tail. Failed runs kept as .failN.
cd "$(dirname "$0")"
while ! grep -q SKEW_DONE mq7.log; do sleep 30; done
P=$(ps -eo pid,args | grep -E "[b]ash mq7.sh" | awk '{print $1}'); [ -n "$P" ] && kill $P
sleep 2
Q=$(ps -eo pid,args | grep -E "[.]/bench (xindex|finedex) 100000000 400000000 12500000 [0-9]+ 1 " | awk '{print $1}'); [ -n "$Q" ] && kill $Q
sleep 5; rm -f r7/cmp/*_t1_*.csv.tmp
echo "STOPPED_MQ7 $(date)" >> mq12_verify.log
B=100000000; TOT=400000000; BS=12500000; BK=data/books_800M_uint64
flags() { case $1 in base|xindex|finedex) echo "";; bgside) echo "bg side";; bgside4) echo "bg side bgthreads=4";; esac; }
idx() { case $1 in xindex|finedex) echo $1;; *) echo alexol;; esac; }
r() { local f=$1 to=$2 arm=$3; shift 3; [ -s "$f" ] && return
  timeout $to ./bench $(idx $arm) "$@" $(flags $arm) 2>>mq12_verify.log | grep --line-buffered -E "^x,|^CPU,|^LOCK2,|^LOCK3," > "$f.tmp"
  local rc=${PIPESTATUS[0]}; echo "RC $rc $f" >> mq12_verify.log
  [ $rc = 0 ] && mv "$f.tmp" "$f" || mv "$f.tmp" "$f.fail$rc"; }
mkdir -p r12/sorted4m r12/sorted400
for seed in 1866 5 72; do
  for arm in base bgside4; do r r12/sorted4m/${arm}_t16_s${seed}.csv 300 $arm 1000000 4000000 100000 $seed 16 x order=sorted lockstats; done
  for arm in base bgside; do r r12/sorted4m/${arm}_t1_s${seed}.csv 300 $arm 1000000 4000000 100000 $seed 1 x order=sorted lockstats; done
done
for arm in base bgside4; do r r12/sorted400/${arm}_t16_s1866.csv 2400 $arm $B $TOT $BS 1866 16 x data=$BK order=sorted lockstats; done
echo SORTED_DONE
for seed in 5 72; do for arm in xindex finedex; do
  r r7/cmp/${arm}_books_t1_s${seed}.csv 3600 $arm $B $TOT $BS $seed 1 x data=$BK; done; done
echo MQ12_DONE
