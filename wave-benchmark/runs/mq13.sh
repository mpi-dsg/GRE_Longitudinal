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
guard() { local o=$(others); echo "UPTIME $(uptime) others_running=$o before $1" >> mq13_verify.log
  if [ "$o" -gt 0 ]; then echo "BUSY_STOP $(date) before $1" >> mq13_verify.log
    ps -eo user:32,pid,stat,pcpu,args --no-headers | awk '$1!="bindsch" && $1!="root" && $3 ~ /^R/' >> mq13_verify.log
    echo BUSY_STOP; exit 3; fi; }
r() { local f=$1 to=$2; shift 2; [ -s "$f" ] && return; ls "$f".fail* >/dev/null 2>&1 && return; guard "$f"
  timeout $to ./bench "$@" 2>>mq13_verify.log | grep --line-buffered -E "^x,|^LOCK,|^LOCK2,|^LOCK3," > "$f.tmp"
  local rc=${PIPESTATUS[0]}; echo "RC $rc $f" >> mq13_verify.log
  [ $rc = 0 ] && mv "$f.tmp" "$f" || mv "$f.tmp" "$f.fail$rc"; }
M="1000000 4000000 100000"
mkdir -p r13/sweep r13/ls r13/var r13/pin r13/bt r13/ns r13/stag400
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
for rep in 1 2 3; do for seed in 1866 5 72; do for T in 1 4 8 16; do
  r r13/var/d017_t${T}_s${seed}_r${rep}.csv 300 alexol $M $seed $T x stagger delta=0.17
  r r13/var/d008_t${T}_s${seed}_r${rep}.csv 300 alexol $M $seed $T x stagger delta=0.08
  r r13/var/low017_t${T}_s${seed}_r${rep}.csv 300 alexol $M $seed $T x stagger lowonly delta=0.17
  r r13/var/base_t${T}_s${seed}_r${rep}.csv 300 alexol $M $seed $T x
done; done; done
echo VAR_DONE
for rep in 1 2 3; do for seed in 1866 5 72; do for T in 1 4 8 16; do
  r r13/bt/unif07_t${T}_s${seed}_r${rep}.csv 300 btreebulk $M $seed $T x bfill=0.70
  r r13/bt/rand07_t${T}_s${seed}_r${rep}.csv 300 btreebulk $M $seed $T x bfill=0.70 bspread=0.20
  r r13/bt/unif09_t${T}_s${seed}_r${rep}.csv 300 btreebulk $M $seed $T x bfill=0.90
  r r13/bt/rand09_t${T}_s${seed}_r${rep}.csv 300 btreebulk $M $seed $T x bfill=0.90 bspread=0.09
done; done; done
echo BT_DONE
for rep in 1 2 3; do for seed in 1866 5 72; do for T in 1 16; do
  for nb in 512 4096 32768 262144 524288 2097152; do r r13/ns/nb${nb}_t${T}_s${seed}_r${rep}.csv 300 alexsized $M $seed $T x nodebytes=$nb; done
done; done; done
echo NS_DONE
export OMP_PROC_BIND=close OMP_PLACES=cores
for rep in 1 2 3; do for seed in 1866 5 72; do for T in 1 2 4 8 16; do
  r r13/pin/base_t${T}_s${seed}_r${rep}.csv 300 alexol $M $seed $T x
  r r13/pin/stag_t${T}_s${seed}_r${rep}.csv 300 alexol $M $seed $T x stagger
done; done; done
unset OMP_PROC_BIND OMP_PLACES
echo PIN_DONE
for seed in 1866 5 72; do for T in 1 16; do
  r r13/stag400/stag_books_t${T}_s${seed}.csv 2400 alexol $B $TOT $BS $seed $T x stagger data=$BK
  r r13/stag400/base_books_t${T}_s${seed}.csv 2400 alexol $B $TOT $BS $seed $T x data=$BK
done; done
echo MQ13_DONE
