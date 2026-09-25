#!/bin/bash
# (1) Constants test: ALEX-OL rebuilt with kInitDensity_=0.6, kMaxDensity_=0.8, kMinDensity_=0.5.
#     Predicted burst positions were recorded before these runs (results/predictions/).
# (2) Seeds 5 and 72 for every row of the account table (400M books, 16 threads).
# Waits for mq10.sh. One run at a time; stops if another user's process is running.
cd "$(dirname "$0")"
while ps -eo args | grep -qE "[m]q10.sh"; do sleep 30; done
B=100000000; TOT=400000000; BS=12500000; BK=data/books_800M_uint64
others() { local n=0; for i in 1 2 3 4 5; do
  n=$((n + $(ps -eo user:32,stat --no-headers | awk '$1!="bindsch" && $1!="root" && $2 ~ /^R/' | wc -l)))
  sleep 1; done; echo $n; }
guard() { local o=$(others); echo "UPTIME $(uptime) others_running=$o before $1" >> mq11_verify.log
  if [ "$o" -gt 0 ]; then echo "BUSY_STOP $(date) before $1" >> mq11_verify.log
    ps -eo user:32,pid,stat,pcpu,args --no-headers | awk '$1!="bindsch" && $1!="root" && $3 ~ /^R/' >> mq11_verify.log
    echo BUSY_STOP; exit 3; fi; }
flags() { case $1 in bgside4) echo "bg side bgthreads=4";; *) echo "";; esac; }
idx() { case $1 in base|bgside4) echo alexol;; *) echo $1;; esac; }
r() { local bin=$1 f=$2 to=$3 arm=$4; shift 4; [ -s "$f" ] && return; guard "$f"
  timeout $to $bin $(idx $arm) "$@" $(flags $arm) 2>>mq11_verify.log | grep --line-buffered -E "^x,|^CPU," > "$f.tmp"
  local rc=${PIPESTATUS[0]}; echo "RC $rc $f" >> mq11_verify.log
  [ $rc = 0 ] && mv "$f.tmp" "$f" || mv "$f.tmp" "$f.fail$rc"; }
# ---- build the modified-constants binary from the tested sources (same flags as build.sh, XF=0)
if [ ! -x bench_consts ]; then
  rm -rf cbuild && cp -R /local/bindsch/gremini/wave-benchmark cbuild && cd cbuild
  GRE=/local/bindsch/gremini sh build.sh > build.log 2>&1
  H=_c/alexol/src/alex_nodes.h
  sed -i 's/static constexpr double kMaxDensity_ = 0.8;/static constexpr double kMaxDensity_ = 0.8;/; s/^      0.7; \/\/ density of data nodes after bulk loading/      0.6; \/\/ density of data nodes after bulk loading/; s/static constexpr double kMinDensity_ = 0.6;/static constexpr double kMinDensity_ = 0.5;/' $H
  grep -n -A1 "kInitDensity_ =\|kMinDensity_ =\|kMaxDensity_ =" $H >> ../mq11_verify.log
  g++ -O3 -std=c++17 -march=native -fopenmp -DLOCK_STATS -Ishim -I_c -I_c/alexol -I_c/alexol/src -I_c/lippol -I_c/lippol/src \
      -I_c/sali -I_c/sali/src -I_c/btreeolc -I_c/artsync -I. -o ../bench_consts bench.cpp -lpthread -ltbb >> build.log 2>&1
  cd ..
fi
[ -x bench_consts ] || { echo "BUILD_FAIL" >> mq11_verify.log; exit 1; }
mkdir -p r11/consts r11/acct/ins r11/acct/mix r11/acct/read90
for seed in 1866 5 72; do for T in 1 16; do
  r ./bench_consts r11/consts/base_t${T}_s${seed}.csv 300 base 1000000 4000000 100000 $seed $T x; done; done
for T in 1 16; do r ./bench_consts r11/consts/base_books_t${T}_s1866.csv 2400 base $B $TOT $BS 1866 $T x data=$BK; done
echo CONSTS_DONE
# ---- account table: seeds 5 and 72
for seed in 5 72; do
  for a in lippol sali btreebulk artolc; do r ./bench r11/acct/ins/${a}_books_t16_s${seed}.csv 2400 $a $B $TOT $BS $seed 16 x data=$BK; done
  for a in lippol sali btreebulk artolc; do r ./bench r11/acct/mix/${a}_books_t16_s${seed}.csv 2400 $a $B $TOT $BS $seed 16 x data=$BK readpct=50; done
  for a in base bgside4 lippol sali btreebulk artolc xindex finedex; do
    r ./bench r11/acct/read90/${a}_books_t16_s${seed}.csv 2400 $a $B $TOT $BS $seed 16 x data=$BK readpct=90; done
done
echo MQ11_DONE
