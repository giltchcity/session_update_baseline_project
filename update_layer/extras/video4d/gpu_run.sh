#!/usr/bin/env bash
# One GPU job under runs/GPU.lock (shared with the other sessions): started only while the card holds <= 10 GB, stopped
# if the card passes 14.5 GB (15.5 GB is never passed). gpu_run.sh CMD ARGS...
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
card() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
( flock 9; echo "$(date +%T) lock taken: $*" | cut -c1-200
  while [ $(card) -gt 10240 ]; do sleep 20; done
  setsid "$@" & p=$!; peak=0
  while kill -0 $p 2>/dev/null; do
    c=$(card); [ $c -gt $peak ] && peak=$c
    if [ $c -gt 14848 ]; then kill -TERM -- -$p 2>/dev/null; echo "stopped, card $c MiB > 14.5 GB: $*"; break; fi
    sleep 3
  done
  wait $p; echo "$(date +%T) done, card peak $peak MiB" ) 9>$LOCK
