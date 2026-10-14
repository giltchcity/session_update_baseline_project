"""原理思想：K2 的规则是“一帧带来地图没有的变化，就是关键帧”。这里不跑链，用会话末存档地图按 T1 取该帧时刻还活着的高斯，对链里每个非关键帧跑 K2 的同两项 GaME 变化检测，预先数出 K2 会多提升哪些帧，作为真实链之前的门。来源：GaME 变化检测 baselines/GaME/src/entities/game.py:283-330；本仓库 static_kf 分支 GameBackend.integrate（K2）。
Offline proxy for [K2]: on which frames would a frame of an unmoved camera become a keyframe? (GPU renders, nothing trained)

The stored map of a session (checkpoint_<s>.pt, restored as quick_screen.py does) stands in for the online map: at each frame the
rows alive at that frame's stamp (T1) are rendered, and every frame the chain did NOT take as a keyframe (its stamp is not in the
checkpoint's kf_stamp) goes through the same two tests as GameBackend.integrate on branch static_kf (ae5f24f): GaME's new-geometry
masks (TrackedGaME._find_added_geometry_masks, C5) and conflicting Gaussians (GaME._detect_conflicting_gaussians), with the frame
built by GameBackend._sample. Proxy, not the chain: the stored map has its end-of-session geometry (rows later optimised), and
D1's motion pixels are not available offline (dynamic semantics only). Output: one JSON line per promoted frame + a summary.
usage: k2_offline.py CODE_DIR CKPT DATASET SESSION OUT_JSONL [STRIDE]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE, CKPT, DATASET, SESSION, OUT = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
STRIDE = int(sys.argv[6]) if len(sys.argv) > 6 else 1
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config                                                    # noqa: E402
from update_layer.backends.game.game import GameBackend, Crop, gu                              # noqa: E402
from update_layer.eval.kf_strip import restore                                                 # noqa: E402
from update_layer.frames import FlatSession                                                    # noqa: E402

torch.set_num_threads(2)
_, info, specs = dataset_config(DATASET)
spec = next(sp for sp in specs if sp.name == f'{DATASET}_{SESSION}')
ck = torch.load(CKPT, mmap=True, map_location='cpu', weights_only=False)
be = GameBackend(info, own_update=False)
be.game = be.prior_from_state(restore(ck, DATASET)['backend'])
g = be.game
be.session = FlatSession(spec, pixel_step=1)
be.crop = Crop(be.session.K, be.step)
kf_stamps = {int(s) for s in ck['backend']['kf_stamp'].values()}
indices = list(be.session.indices())
promoted, tested, t0 = 0, 0, time.monotonic()
with open(OUT, 'w') as out:
    for k, i in enumerate(indices):
        stamp = int(be.session.stamp_ns(i))
        if stamp in kf_stamps or k % STRIDE:
            continue
        sample, instance, dyn = be._sample(i)
        pose = sample['pose'].copy()
        kf = {"color": gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0,
              "depth": gu.np2torch(sample["depth"], device="cuda"), "masks": sample["masks"].cuda(),
              "pose": gu.np2torch(pose, device="cuda"), "intrinsics": sample["intrinsics"]}
        with torch.no_grad():
            g.gaussian_model.alive = g.alive_at(stamp)
            added = g._find_added_geometry_masks(kf)
            conflict = None if added else g._detect_conflicting_gaussians(kf)
            g.gaussian_model.alive = None
        tested += 1
        if added or conflict is not None:
            promoted += 1
            # which objects: GameBackend._masks orders the instance masks first (ids ascending, >= min_mask_px pixels), then the
            # connected components of the background semantics; conflicting rows by their identity (I1)
            ids, counts = np.unique(instance[instance > 0], return_counts=True)
            inst_ids = [int(a) for a, c in zip(ids, counts) if c >= be.min_mask_px]
            added_objects = [inst_ids[j] if j < len(inst_ids) else 'background' for j in added]
            conflict_ids = {}
            if conflict is not None:
                u, c = torch.unique(g.identity[conflict], return_counts=True)
                conflict_ids = {int(a): int(b) for a, b in sorted(zip(u.tolist(), c.tolist()), key=lambda t: -t[1])[:6]}
            out.write(json.dumps(dict(frame=i, stamp_ns=stamp, new_geometry_masks=len(added), added_objects=added_objects,
                                      conflicting_rows=0 if conflict is None else int(conflict.sum()),
                                      conflicting_identities=conflict_ids)) + '\n')
        if tested % 500 == 0:
            print(f'{tested} non-keyframes tested, {promoted} promoted, {tested / (time.monotonic() - t0):.1f} frames/s', flush=True)
print(f'K2_OFFLINE {DATASET} {SESSION}: non-keyframes tested {tested} (stride {STRIDE}), promoted {promoted} '
      f'({promoted / max(tested, 1) * 100:.2f} %), chain keyframes {len(kf_stamps)}', flush=True)
