#!/usr/bin/env bash
# Isaac's ground truth of the four watched moves in close-up, every 1/12 s (as render_all.sh / render_plant.sh made it:
# --history cut --subframes 24), into D:\...\final_state_renders_20261022\d1_<session>_<unit>; each under runs/GPU.lock.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)                 # this folder of update_layer/extras (the scripts)
V=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/analysis/paper_figs_20261022/video4d
IW='D:\3Study\ETH\FT\Synthetic_ABC\final_state_renders_20261022'; I=/mnt/d/3Study/ETH/FT/Synthetic_ABC/final_state_renders_20261022
isaac() { cd $I && timeout 3600 cmd.exe /c "D:\\3Study\\Isaac\\python.bat $IW\\render_video_isaac.py --session $1 --camera $IW\\$2.json --t0 $3 --t1 $4 --step 0.0833333 --output $IW\\$5 --history cut --subframes 24"; }
export -f isaac; export I IW
for e in "b 72 40.0 47.5" "b 164 29.6 34.0" "b 109 110.5 118.5" "a 164 115.0 122.5"; do
  set -- $e; n=$1_$2
  $HERE/../video4d/gpu_run.sh bash -c "isaac $1 cam_$n $3 $4 d1_$n" > /tmp/claude-1000/isaac_$n.log 2>&1
  echo "$(date +%T) d1_$n: $(ls $I/d1_$n/*_rgb.png 2>/dev/null | wc -l) frames, $(grep 'card peak' /tmp/claude-1000/isaac_$n.log)"
done
echo GT_DONE
