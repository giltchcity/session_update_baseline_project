"""TODO 4 evidence: what D1 reports frame by frame while GaME takes no keyframe (CPU or GPU; D1 run exactly as in the chain).

D1 is built and fed as in the chain (layer.py:413 config, run.py:201/221/236 frames, every frame of the session except the
held-out frames of the last session). For every processed frame it writes one JSON line: the frame index, whether the chain
took it as a GaME keyframe (the checkpoint's kf_stamp of that session), D1's dynamic pixel count (the mask process() returns)
and every track D1 observed in it (id, dynamic flag, AABB centre, points; physical tracks are keyed by the instance id).
usage: d1_trace.py CKPT DATASET SESSION FIRST_FRAME_TO_WRITE OUT_JSONL
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from update_layer.run import dataset_config, INPUT_STEP                                   # noqa: E402
from update_layer.frames import FlatSession, synthetic_session, real_session             # noqa: E402
from update_layer.core.d1 import D1, D1Config                                            # noqa: E402

ckpt, DATASET, SESSION, FIRST, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
cfg, info, specs = dataset_config(DATASET)
d1 = D1(D1Config(voxel_size=cfg.map_resolution, truncation=cfg.truncation, max_range=cfg.max_range,
                 dynamic_labels=tuple(cfg.dynamic_semantics)))                          # layer.py:413
fs = FlatSession((synthetic_session if DATASET == 'synthetic' else real_session)(SESSION), pixel_step=INPUT_STEP[DATASET])
b = torch.load(ckpt, mmap=True, map_location='cpu', weights_only=False)['backend']
kf_stamps = {int(s) for s in b['kf_stamp'].values()}
indices = list(fs.indices())
final = SESSION == specs[-1].name.split('_')[-1]
held = {n for n in range(len(indices)) if n % 10 == 0 and n != 0} if final else set()   # run.py:221 g8_holdout
with open(OUT, 'w') as out:
    for k, i in enumerate(indices):
        if k in held:
            continue
        frame = fs.load(i)
        dyn = d1.process(frame)
        if i < FIRST:
            continue
        live = d1.live[-1]
        out.write(json.dumps(dict(frame=i, stamp_ns=int(frame.stamp_ns), keyframe=int(frame.stamp_ns) in kf_stamps,
                                  dynamic_pixels=int(dyn.sum()), tracks=live['tracks'])) + '\n')
print('D1_TRACE_DONE', flush=True)
