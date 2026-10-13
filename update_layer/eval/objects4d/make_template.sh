#!/usr/bin/env bash
# template.4dmap for score_timeline.sh: the layer layout of an official Khronos map, no nodes.
#   make_template.sh ANY_KHRONOS.4dmap
set -euo pipefail
FT=/home/jixian/Desktop/FT
B=$FT/build_artifacts/update_layer_objects4d
python3 -c "import struct,sys; sys.stdout.buffer.write(b'OBJ4D002' + struct.pack('<IQI', 1, 1, 0))" > $B/empty.obj4d  # one empty snapshot
bash -c "source $FT/build_artifacts/official_khronos_63faadde_abc/environment.sh >/dev/null 2>&1; $B/objects4d build $(realpath "$1") $B/empty.obj4d $B/template.4dmap"
rm -f $B/empty.obj4d
