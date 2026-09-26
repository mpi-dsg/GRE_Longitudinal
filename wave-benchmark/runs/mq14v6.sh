#!/bin/bash
# volta06 (same server as r9, r10, r11; same driver and index code). Puts the whole account table
# on this server: XIndex and FINEdex insert-only, and ALEX-OL, the design, XIndex, and FINEdex at
# 50/50, 16 threads, 3 seeds. Every run reads back all keys.
cd "$(dirname "$0")"
B=100000000; TOT=400000000; BS=12500000; BK=data/books_800M_uint64
L=mq14v6_verify.log
others() { local n=0; for i in 1 2 3; do
  n=$((n + $(ps -eo user:32,stat --no-headers | awk '$1!="bindsch" && $1!="root" && $2 ~ /^R/' | wc -l)))
  sleep 1; done; echo $n; }
guard() { local o tries=0
  while :; do o=$(others); echo "UPTIME $(uptime) others_running=$o before $1" >> $L
    [ "$o" -eq 0 ] && return
    tries=$((tries+1)); echo "BUSY_WAIT $(date) try $tries before $1" >> $L
    ps -eo user:32,pid,stat,pcpu,args --no-headers | awk '$1!="bindsch" && $1!="root" && $3 ~ /^R/' >> $L
    [ $tries -ge 10 ] && { echo "BUSY_STOP $(date) before $1" >> $L; echo BUSY_STOP; exit 3; }
    sleep 60; done; }
flags() { case $1 in side) echo side;; bgside) echo "bg side";; bgside2) echo "bg side bgthreads=2";;
  bgside4) echo "bg side bgthreads=4";; ns2048) echo "nodebytes=32768";; *) echo "";; esac; }
idx() { case $1 in base|side|bgside|bgside2|bgside4) echo alexol;; ns2048) echo alexsized;; *) echo $1;; esac; }
r() { local f=$1 to=$2 arm=$3; shift 3; [ -s "$f" ] && return; ls "$f".fail* >/dev/null 2>&1 && return; guard "$f"
  timeout $to ./bench $(idx $arm) "$@" $(flags $arm) 2>>$L | grep --line-buffered -E "^x,|^CPU,|^LOCK2,|^LOCK3," > "$f.tmp"
  local rc=${PIPESTATUS[0]}; echo "RC $rc $f" >> $L
  [ $rc = 0 ] && mv "$f.tmp" "$f" || mv "$f.tmp" "$f.fail$rc"; }
mkdir -p r14/acct/ins r14/acct/mix
for s in 1866 5 72; do for a in base bgside4; do
  r r14/acct/mix/${a}_books_t16_s$s.csv 2400 $a $B $TOT $BS $s 16 x data=$BK readpct=50; done; done
for s in 1866 5 72; do r r14/acct/ins/finedex_books_t16_s$s.csv 2400 finedex $B $TOT $BS $s 16 x data=$BK
  r r14/acct/mix/finedex_books_t16_s$s.csv 2400 finedex $B $TOT $BS $s 16 x data=$BK readpct=50; done
for s in 1866 5 72; do r r14/acct/ins/xindex_books_t16_s$s.csv 2400 xindex $B $TOT $BS $s 16 x data=$BK
  r r14/acct/mix/xindex_books_t16_s$s.csv 2400 xindex $B $TOT $BS $s 16 x data=$BK readpct=50; done
echo MQ14V6_DONE
