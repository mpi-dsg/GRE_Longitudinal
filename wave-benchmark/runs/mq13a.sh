#!/bin/bash
# Reruns, on the final binary (fixed ALEX-OL, every key read back), of the older run sets the paper
# still used: 4M thread sweep, randomized density and its variants, restart counters, NUMA pinning,
# B+-tree fill, node-size sweep, and 400M randomized density. Configurations match the original
# scripts (sweep.sh, stag.sh, lockstats.sh, var.sh, pin.sh, bt.sh, ns.sh, q14.sh). One run at a time;
# stops if another user's process is running; failed runs kept as .failN.
cd "$(dirname "$0")"
B=100000000; TOT=400000000; BS=12500000; BK=data/books_800M_uint64
others() { local n=0; for i in 1 2 3; do
  n=$((n + $(ps -eo user:32,stat --no-headers | awk '$1!="bindsch" && $1!="root" && $2 ~ /^R/' | wc -l)))
  sleep 1; done; echo $n; }
guard() { local o tries=0
  while :; do o=$(others); echo "UPTIME $(uptime) others_running=$o before $1" >> mq13_verify.log
    [ "$o" -eq 0 ] && return
    tries=$((tries+1)); echo "BUSY_WAIT $(date) try $tries before $1" >> mq13_verify.log
    ps -eo user:32,pid,stat,pcpu,args --no-headers | awk '$1!="bindsch" && $1!="root" && $3 ~ /^R/' >> mq13_verify.log
    [ $tries -ge 10 ] && { echo "BUSY_STOP $(date) before $1" >> mq13_verify.log; echo BUSY_STOP; exit 3; }
    sleep 60; done; }
r() { local f=$1 to=$2; shift 2; [ -s "$f" ] && return; ls "$f".fail* >/dev/null 2>&1 && return; guard "$f"
  timeout $to ./bench "$@" 2>>mq13_verify.log | grep --line-buffered -E "^x,|^LOCK,|^LOCK2,|^LOCK3," > "$f.tmp"
  local rc=${PIPESTATUS[0]}; echo "RC $rc $f" >> mq13_verify.log
  [ $rc = 0 ] && mv "$f.tmp" "$f" || mv "$f.tmp" "$f.fail$rc"; }
M="1000000 4000000 100000"
mkdir -p r13/sweep r13/ls r13/dens
for rep in 1 2 3; do for seed in 1866 5 72; do for T in 1 2 4 8 16; do
  for idx in alexol lippol sali btreeolc artolc; do r r13/sweep/${idx}_t${T}_s${seed}_r${rep}.csv 300 $idx $M $seed $T x; done
  r r13/sweep/alexolstag_t${T}_s${seed}_r${rep}.csv 300 alexol $M $seed $T x stagger
done; done; done
echo SWEEP_DONE
for rep in 1 2 3; do for seed in 1866 5 72; do for T in 1 16; do
  r r13/ls/base_t${T}_s${seed}_r${rep}.csv 300 alexol $M $seed $T x lockstats
  r r13/ls/stag_t${T}_s${seed}_r${rep}.csv 300 alexol $M $seed $T x stagger lockstats
done; done; done
echo LS_DONE
# node densities right after bulk load (released and randomized), with the measured source
if [ ! -x dens ]; then g++ -O2 -std=c++17 -fopenmp -Ishim -Icompetitor -Icompetitor/alexol -Icompetitor/alexol/src -o dens ../../../bindsch/wave/bgx7/dens.cpp -lpthread -ltbb 2>>mq13_verify.log || g++ -O2 -std=c++17 -fopenmp -Ishim -Icompetitor -Icompetitor/alexol -Icompetitor/alexol/src -o dens dens.cpp -lpthread -ltbb 2>>mq13_verify.log; fi
for seed in 1866 5 72; do ./dens 1000000 $seed > r13/dens/released_s$seed.txt; ./dens 1000000 $seed stagger > r13/dens/stagger_s$seed.txt; done
echo MQ13A_DONE
