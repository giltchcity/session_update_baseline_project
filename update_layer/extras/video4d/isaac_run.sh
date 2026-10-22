#!/usr/bin/env bash
# Isaac ground truth for the 4D video (camera v2): isaac_run.sh SESSION T0 T1 STEP OUT_NAME [--times S,S,...] (render_video_isaac.py on the D drive)
I=/mnt/d/3Study/ETH/FT/Synthetic_ABC/final_state_renders_20261022; IW='D:\3Study\ETH\FT\Synthetic_ABC\final_state_renders_20261022'
cd $I && timeout 7200 cmd.exe /c "D:\\3Study\\Isaac\\python.bat $IW\\render_video_isaac.py --session $1 --camera $IW\\camera_video_v2.json --t0 $2 --t1 $3 --step $4 --output $IW\\$5 ${@:6}"
