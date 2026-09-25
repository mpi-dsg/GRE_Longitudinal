#!/bin/bash
# Rerun of the 400M books design arms at 16 threads on an idle server. The final rerun (mq6.sh)
# measured these arms while the machine was slower: the baseline's time there (19.9 s) is 15%
# above the same configuration on the same binary in mq7.sh (17.3 s), so these runs replace the
# 16-thread times. Same binary as mq7.sh (md5 38b1463d). One run at a time; stops if another
# user's process is running (BUSY_STOP), and keeps failed runs as .failN.
cd "$(dirname "$0")"
B=100000000; TOT=400000000; BS=12500000; BK=data/books_800M_uint64
flags() { case $1 in base) echo "";; side) echo side;; bgside) echo "bg side";;
  bgside2) echo "bg side bgthreads=2";; bgside4) echo "bg side bgthreads=4";; ns2048) echo "nodebytes=32768";; esac; }
idx() { case $1 in ns2048) echo alexsized;; *) echo alexol;; esac; }
others() { local n=0; for i in 1 2 3 4 5; do
  n=$((n + $(ps -eo user:32,stat --no-headers | awk '$1!="bindsch" && $1!="root" && $2 ~ /^R/' | wc -l)))
  sleep 1; done; echo $n; }
guard() { local o=$(others); echo "UPTIME $(uptime) others_running=$o before $1" >> mq9_verify.log
  if [ "$o" -gt 0 ]; then echo "BUSY_STOP $(date) before $1" >> mq9_verify.log
    ps -eo user:32,pid,stat,pcpu,args --no-headers | awk '$1!="bindsch" && $1!="root" && $3 ~ /^R/' >> mq9_verify.log
    echo BUSY_STOP; exit 3; fi; }
r() { local f=$1 to=$2 arm=$3; shift 3; [ -s "$f" ] && return; guard "$f"
  timeout $to ./bench $(idx $arm) "$@" $(flags $arm) 2>>mq9_verify.log | grep --line-buffered -E "^x,|^CPU,|^LOCK2,|^LOCK3," > "$f.tmp"
  local rc=${PIPESTATUS[0]}; echo "RC $rc $f" >> mq9_verify.log
  [ $rc = 0 ] && mv "$f.tmp" "$f" || mv "$f.tmp" "$f.fail$rc"; }
mkdir -p r9/b400
for seed in 1866 5 72; do for arm in base side bgside bgside2 bgside4 ns2048; do
  r r9/b400/${arm}_books_t16_s${seed}.csv 2400 $arm $B $TOT $BS $seed 16 x data=$BK; done; done
echo MQ9_DONE
