"""P3: official Khronos's label space for a 3RScan room: the main table's (session_update_baseline configs/
nss_ade20k_room_label_space.yaml, room18's 12 object classes) with object_labels set to the room's thing classes (the
physical catalog's semantic classes, STATIC_OTHER 1 included: the adapter files labels without an ADE name, e.g.
trash can, bucket, washer, under it), as our runs take every thing of the scene; nothing else changed. Writes datasets/3rscan/mapping_configs/ROOM/khronos_label_space.yaml.
usage: p3_khronos_label_space.py ROOM
"""
import json
import re
import sys
from pathlib import Path

FT = Path('/home/jixian/Desktop/FT')
BASE = FT / 'session_update_baseline_project/session_update_baseline/configs/nss_ade20k_room_label_space.yaml'
cfg = FT / 'datasets/3rscan/mapping_configs' / sys.argv[1]
classes = sorted({e['semantic_id'] for e in json.load((cfg / 'physical_catalog.json').open())['entities']})
text = BASE.read_text()
block = re.search(r'^object_labels:\n(?:  - \d+\n)+', text, re.M)
assert block, 'object_labels block not found'
text = text[:block.start()] + f'object_labels: {classes}  # the room\'s thing classes (p3_khronos_label_space.py)\n' + text[block.end():]
(cfg / 'khronos_label_space.yaml').write_text(f'# {BASE.name} with object_labels = the thing classes of 3RScan room {sys.argv[1]}\n' + text)
print(sys.argv[1], 'Khronos object classes', classes)
