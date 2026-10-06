"""Per-state binding of Gaussians (one object system, as Khronos: fragment -> node -> its own geometry), offline on a
GaME checkpoint, and the re-emission of a session's boundary and final snapshots with it.

  python -m update_layer.eval.state_binding CKPT_PREV CKPT_CUR TIMELINE OUT_TIMELINE SESSION_LETTER

Nodes = the registry's object states, read from the checkpoint as the distinct (state_birth, death_state) of each
identity's Gaussians (T1 already holds them per Gaussian). A node's geometry = the Gaussians its OWN frames bind to it
(README S19: each state only from its own measured geometry; M1a: a label goes to the surface its reading's endpoint
lies on): for every keyframe t and every pixel with a valid reading, the first-echo Gaussian of the map of time t (the
v2 rasterizer, T first <= 0.5) receives one vote for the state of the pixel's identity that is alive at t, when the reading
lies within tau = max(its own 3 sigma, 0.05 m surface tolerance: EvidenceConfig.surface_match_tolerance) of the
first-echo depth. Readings that carry no state evidence do not vote (M1a: likelihood ratio 1): invalid readings,
people and D1-masked pixels, pixels without an instance (as I1: FlashSplat kernel label 0 = no vote), and a pixel
whose identity has more than one state alive at t, unless the Gaussian's own state is one of them (then it votes for
its own state). Each Gaussian takes the state with most votes; a Gaussian without votes keeps its
state (never a first echo: interiors, occluded parts). Its identity, state birth and state end follow its state.
The boundary snapshot (= CKPT_PREV's final map) and the final snapshot (CKPT_CUR) are re-rendered with the first-echo
readout; every other snapshot of TIMELINE stays the run's own.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TOL = 0.05                      # EvidenceConfig.surface_match_tolerance (the evidence's on-surface tolerance)
MAX = torch.iinfo(torch.int64).max


def load_backend(path, info):
    from update_layer.backends.game.game import GameBackend
    ck = torch.load(path, map_location="cpu", weights_only=False)
    to_gpu = lambda x: (torch.nn.Parameter(x.detach().cuda(), requires_grad=x.requires_grad)
                        if isinstance(x, torch.nn.Parameter) else x.cuda() if torch.is_tensor(x)
                        else type(x)(to_gpu(v) for v in x) if isinstance(x, (tuple, list))
                        else {k: to_gpu(v) for k, v in x.items()} if isinstance(x, dict) else x)
    ck["backend"]["model"] = to_gpu(ck["backend"]["model"])
    be = GameBackend(info, own_update=False)
    be.game = be.prior_from_state(ck["backend"])
    return be, int(ck["prev_final"])


@torch.no_grad()
def bind(be, specs) -> dict:
    """Re-bind every Gaussian to a state by first-echo votes of the states' own frames. Returns counts."""
    from update_layer.backends.game.game import Crop, gu, probe_render_fe
    from update_layer.frames import FlatSession, dynamic_mask
    g = be.game
    ident, sb, ds = g.identity.clone(), g.state_birth.clone(), g.death_state.clone()
    # the states: (identity, birth, end); 0 = background
    keys = torch.stack([ident.clamp(min=0), sb, ds], 1)
    obj = (ident > 0) & (sb < MAX)
    states = [(0, -(2 ** 62), MAX)] + sorted({tuple(k) for k in keys[obj].tolist()})
    S = len(states)
    st_id = torch.tensor([s[0] for s in states], device="cuda")
    st_b = torch.tensor([s[1] for s in states], device="cuda")
    st_d = torch.tensor([s[2] for s in states], device="cuda")
    votes = torch.zeros((len(ident), S), dtype=torch.int32, device="cuda")
    # each Gaussian's current state index (0: background or no state)
    lut = {s: j for j, s in enumerate(states)}
    own_state = torch.tensor([lut.get(tuple(k), 0) if o else 0 for k, o in zip(keys.tolist(), obj.tolist())],
                             dtype=torch.int64, device="cuda")
    sessions = {sp.name.split("_")[-1]: FlatSession(sp, pixel_step=1) for sp in specs}   # as GameBackend.start_session
    stamp_index = {}
    for name, fs in sessions.items():
        for i in range(len(fs.ids)):
            stamp_index[fs.stamp_ns(i)] = (name, i)
    sigma3 = 3.0 * g.gaussian_model.get_scaling.detach().max(dim=1).values      # scaled units
    for kid, kf in g.keyframes.items():
        t = g.kf_stamp.get(kid)
        if t is None or t not in stamp_index:
            continue
        name, idx = stamp_index[t]
        be.session = sessions[name]
        be.crop = Crop(be.session.K, be.step)                                            # as GameBackend.start_session
        sample, instance, dyn = be._sample(idx)
        g.gaussian_model.alive = g.alive_at(t)
        view = gu.flashsplat_cam(kf["color"].cuda(), kf["depth"].cuda(), None, kf["intrinsics"], kf["pose"].cpu(), None)
        fe = probe_render_fe(view, g.gaussian_model)
        mi = fe["median_index"].squeeze().long()
        med = fe["median"].squeeze()
        meas = kf["depth"].cuda().reshape(mi.shape)
        inst = torch.as_tensor(instance, device="cuda").long().reshape(mi.shape)
        dmask = torch.as_tensor(dyn, device="cuda").bool().reshape(mi.shape)
        ok = (mi >= 0) & (meas > 0) & ~dmask
        tau = torch.maximum(sigma3[mi.clamp(min=0)], torch.full_like(med, TOL * be.scale))
        ok &= (meas - med).abs() <= tau
        v, u = torch.nonzero(ok, as_tuple=True)
        gid, lab = mi[v, u], inst[v, u]
        objp = lab > 0                                       # no instance: no vote (I1 kernel label 0)
        gid, lab = gid[objp], lab[objp]
        # the state of the pixel's identity alive at t; several alive: the Gaussian's own if among them, else none
        alive_state = (st_b <= t) & (t < st_d)
        cand = (st_id[None, :] == lab[:, None]) & alive_state[None, :]
        n_c = cand.sum(dim=1)
        own = cand[torch.arange(len(gid), device="cuda"), own_state[gid]] & (own_state[gid] > 0)
        sid = torch.where(n_c == 1, torch.argmax(cand.int(), dim=1), torch.full_like(lab, -1))
        sid = torch.where((n_c > 1) & own, own_state[gid], sid)
        keep = sid >= 0
        votes.index_put_((gid[keep], sid[keep]), torch.ones(int(keep.sum()), dtype=torch.int32, device="cuda"),
                         accumulate=True)
    g.gaussian_model.alive = None
    best, arg = votes.max(dim=1)
    voted = best > 0
    old_ident = g.identity.clone()
    g.identity[voted] = st_id[arg[voted]]
    g.state_birth[voted] = torch.where(st_id[arg[voted]] > 0, st_b[arg[voted]], torch.full_like(st_b[arg[voted]], MAX))
    g.death_state[voted] = torch.where(st_id[arg[voted]] > 0, st_d[arg[voted]], torch.full_like(st_d[arg[voted]], MAX))
    return dict(states=S - 1, gaussians=int(len(ident)), voted=int(voted.sum()),
                identity_changed=int((g.identity != old_ident).sum()))


def main():
    prev, cur, tl_path, out, s = sys.argv[1:6]
    from update_layer.run import dataset_config
    from update_layer.eval.scenelist import SceneListTimeline, load_timeline
    _, info, specs = dataset_config("real")
    tl = load_timeline(tl_path)
    stamps = list(tl.stamps())
    scenes = [tl.scene(x) for x in stamps]
    for k, path in ((0, prev), (-1, cur)):
        be, t = load_backend(path, info)
        assert t == stamps[k], f"{path}: final stamp {t} != timeline stamp {stamps[k]}"
        print(path, bind(be, specs), flush=True)
        be.readout = "first_echo"
        scenes[k] = be._render_scene(t)
        del be
        torch.cuda.empty_cache()
    with open(out, "wb") as f:
        pickle.dump(SceneListTimeline(stamps, scenes), f)
    print("wrote", out)


if __name__ == "__main__":
    main()
