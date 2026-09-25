#!/bin/bash
# sidealways ablation on volta07, with the baseline and the current design (bg side) in the same
# session so every comparison is on one machine. One run at a time. A run whose verification
# fails exits nonzero and is kept only as .failN. Before each run the script logs uptime and
# checks for other users' running processes; if it finds any, it stops (BUSY_STOP).
cd "$(dirname "$0")"
B=100000000; TOT=400000000; BS=12500000
BK=data/books_800M_uint64
flags() { case $1 in base) echo "";; bgside) echo "bg side";; bgside4) echo "bg side bgthreads=4";;
  sa) echo sidealways;; sabg) echo "sidealways bg";; sabg4) echo "sidealways bg bgthreads=4";; esac; }
# Running (state R) processes of users other than bindsch and root, summed over 5 samples.
others() { local n=0; for i in 1 2 3 4 5; do
  n=$((n + $(ps -eo user:32,stat --no-headers | awk '$1!="bindsch" && $1!="root" && $2 ~ /^R/' | wc -l)))
  sleep 1; done; echo $n; }
guard() { local o=$(others)
  echo "UPTIME $(uptime) others_running=$o before $1" >> mq8_verify.log
  if [ "$o" -gt 0 ]; then
    echo "BUSY_STOP $(date) other users running before $1" >> mq8_verify.log
    ps -eo user:32,pid,stat,pcpu,args --no-headers | awk '$1!="bindsch" && $1!="root" && $3 ~ /^R/' >> mq8_verify.log
    echo BUSY_STOP; exit 3; fi; }
r() { local f=$1 to=$2 arm=$3; shift 3; [ -s "$f" ] && return; ls "$f".fail* >/dev/null 2>&1 && return
  guard "$f"
  timeout $to ./bench alexol "$@" $(flags $arm) 2>>mq8_verify.log | grep --line-buffered -E "^x,|^CPU,|^LOCK2,|^LOCK3," > "$f.tmp"
  local rc=${PIPESTATUS[0]}; echo "RC $rc $f $(date +%H:%M:%S)" >> mq8_verify.log
  [ $rc = 0 ] && mv "$f.tmp" "$f" || mv "$f.tmp" "$f.fail$rc"; }
mkdir -p r8/ins r8/mix r8/recent r8/t1 r8/4m
echo "MQ8_START $(date)"
# 1. 400M books, 16 threads: insert-only, lookups of bulk-loaded keys, lookups of recent keys.
for seed in 1866 5 72; do for arm in base bgside4 sa sabg4; do
  r r8/ins/${arm}_books_t16_s${seed}.csv 2400 $arm $B $TOT $BS $seed 16 x data=$BK; done; done
echo "INS_DONE $(date)"
for seed in 1866 5 72; do for arm in base bgside4 sa sabg4; do
  r r8/mix/${arm}_books_t16_s${seed}.csv 2400 $arm $B $TOT $BS $seed 16 x data=$BK readpct=50; done; done
echo "MIX_DONE $(date)"
for seed in 1866 5 72; do for arm in base bgside4 sa sabg4; do
  r r8/recent/${arm}_books_t16_s${seed}.csv 2400 $arm $B $TOT $BS $seed 16 x data=$BK readpct=50 recent; done; done
echo "RECENT_DONE $(date)"
# 2. 4M dense keys, 3 repetitions, 1 and 16 threads, insert-only and readpct=50.
for rep in 1 2 3; do for seed in 1866 5 72; do for T in 1 16; do
  for arm in base bgside4 sa sabg4; do
    r r8/4m/${arm}_t${T}_s${seed}_r${rep}.csv 120 $arm 1000000 4000000 100000 $seed $T x
    r r8/4m/${arm}_mix_t${T}_s${seed}_r${rep}.csv 120 $arm 1000000 4000000 100000 $seed $T x readpct=50
  done; done; done; done
echo "4M_DONE $(date)"
# 3. 400M books, 1 thread (slowest; last). Timeout 3600 s: sidealways is about 2x slower than base at t1 on 4M keys.
for seed in 1866 5 72; do for arm in base bgside sa sabg; do
  r r8/t1/${arm}_books_t1_s${seed}.csv 3600 $arm $B $TOT $BS $seed 1 x data=$BK; done; done
echo "MQ8_DONE $(date)"
