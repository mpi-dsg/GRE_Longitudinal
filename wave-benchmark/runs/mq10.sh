#!/bin/bash
# Learned vs classical indexes on the final binary, 400M books, seed 1866: run time, memory (peak
# RSS, since the classical wrappers report none), lookup latency under a balanced and a
# read-heavy mix. Waits for mq9.sh. One run at a time; stops if another user's process is
# running (BUSY_STOP); failed runs kept as .failN.
cd "$(dirname "$0")"
while ps -eo args | grep -qE "[m]q9.sh"; do sleep 30; done
B=100000000; TOT=400000000; BS=12500000; BK=data/books_800M_uint64
flags() { case $1 in bgside4) echo "bg side bgthreads=4";; *) echo "";; esac; }
idx() { case $1 in base|bgside4) echo alexol;; *) echo $1;; esac; }
others() { local n=0; for i in 1 2 3 4 5; do
  n=$((n + $(ps -eo user:32,stat --no-headers | awk '$1!="bindsch" && $1!="root" && $2 ~ /^R/' | wc -l)))
  sleep 1; done; echo $n; }
guard() { local o=$(others); echo "UPTIME $(uptime) others_running=$o before $1" >> mq10_verify.log
  if [ "$o" -gt 0 ]; then echo "BUSY_STOP $(date) before $1" >> mq10_verify.log
    ps -eo user:32,pid,stat,pcpu,args --no-headers | awk '$1!="bindsch" && $1!="root" && $3 ~ /^R/' >> mq10_verify.log
    echo BUSY_STOP; exit 3; fi; }
r() { local f=$1 to=$2 arm=$3; shift 3; [ -s "$f" ] && return; guard "$f"
  timeout $to ./bench $(idx $arm) "$@" $(flags $arm) 2>>mq10_verify.log | grep --line-buffered -E "^x,|^CPU," > "$f.tmp"
  local rc=${PIPESTATUS[0]}; echo "RC $rc $f" >> mq10_verify.log
  [ $rc = 0 ] && mv "$f.tmp" "$f" || mv "$f.tmp" "$f.fail$rc"; }
mkdir -p r10/ins r10/mix r10/read90
for a in artolc btreebulk sali lippol; do r r10/ins/${a}_books_t16_s1866.csv 2400 $a $B $TOT $BS 1866 16 x data=$BK; done
for a in artolc btreebulk sali lippol; do r r10/mix/${a}_books_t16_s1866.csv 2400 $a $B $TOT $BS 1866 16 x data=$BK readpct=50; done
for a in base bgside4 artolc btreebulk sali lippol xindex finedex; do
  r r10/read90/${a}_books_t16_s1866.csv 2400 $a $B $TOT $BS 1866 16 x data=$BK readpct=90; done
for a in artolc btreebulk; do r r10/ins/${a}_books_t1_s1866.csv 2400 $a $B $TOT $BS 1866 1 x data=$BK; done
echo MQ10_DONE
