#!/bin/bash
# volta14 (same server as r6, r7, r12, r13; same driver and index code). Removes the last
# earlier-driver runs and puts every 400M comparison outside the account table on this server:
# XIndex/FINEdex at one thread (RQ5), the other indexes at one thread on books and SALI on osm
# (RQ2, Figure 2), and the 16-thread design arms (Table 5, Figure 5). Every run reads back all keys.
cd "$(dirname "$0")"
B=100000000; TOT=400000000; BS=12500000; BK=data/books_800M_uint64; OSM=data/osm_cellids_800M_uint64
L=mq14_verify.log
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
mkdir -p r14/cmp r14/s400 r14/b400
r r14/cmp/xindex_books_t1_s1866.csv 3600 xindex $B $TOT $BS 1866 1 x data=$BK
for s in 1866 72; do r r14/cmp/finedex_books_t1_s$s.csv 3600 finedex $B $TOT $BS $s 1 x data=$BK; done
echo CMP_DONE
for a in lippol sali btreebulk artolc; do r r14/s400/${a}_books_t1_s1866.csv 2400 $a $B $TOT $BS 1866 1 x data=$BK; done
r r14/s400/sali_osm_t1_s1866.csv 3000 sali $B $TOT $BS 1866 1 x data=$OSM
echo S400_1866_DONE
for s in 1866 5 72; do for a in base side bgside bgside2 bgside4 ns2048; do
  r r14/b400/${a}_books_t16_s$s.csv 2400 $a $B $TOT $BS $s 16 x data=$BK; done; done
echo B400_DONE
for s in 5 72; do for a in lippol sali btreebulk artolc; do r r14/s400/${a}_books_t1_s$s.csv 2400 $a $B $TOT $BS $s 1 x data=$BK; done
  r r14/s400/sali_osm_t1_s$s.csv 3000 sali $B $TOT $BS $s 1 x data=$OSM; done
echo MQ14_DONE
