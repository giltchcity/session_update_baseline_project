"""What the session-end memory test sees through, from GT-free quantities only (a description, not a verified cause).

  python -m update_layer.eval.g5_characterize DUMP.pt OUT.json [--half METRES]

DUMP.pt: session_<s>/g5_memory_test.pt of a run with --g5 --g5-dump (the stored frames, the tested memory elements and
their evidence). The memory test is run again on the dump with h = --half (default 0: the points backend before h was
ported) and with the dump's own h (must reproduce the run's own counts). For the elements seen through at --half and a
control group (tested, hit by some frame, not seen through), per element over the stored frames it projects into with
a valid reading at its own pixel:
  range q (m); incidence = angle between its normal and the ray (deg; NaN without a normal); distance (px) of its own
  pixel to the nearest depth edge of that frame (a pixel whose 4-neighbour reading differs by more than 5 cm, the
  layer's sensor tolerance; descriptive only); own-pixel residual r = reading - q (cm) and the share of its frames with
  r > tau (the reading lies behind it).
Reported as quantiles over the elements of each group (per element: the median over its frames; for r: over its frames
with r > tau).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from update_layer.core import session_end  # noqa: E402
from update_layer.frames import Intrinsics  # noqa: E402

DEV = session_end.DEV
EDGE = 0.05
Q = (0.1, 0.25, 0.5, 0.75, 0.9)


def edge_distance(rng_mm: np.ndarray) -> np.ndarray:
    r = rng_mm.astype(np.float32) * 1e-3
    ok = r > 0
    e = np.zeros_like(ok)
    for ax in (0, 1):
        a, b = (r[1:], r[:-1]) if ax == 0 else (r[:, 1:], r[:, :-1])
        m = (np.abs(a - b) > EDGE) | ((a > 0) != (b > 0))
        if ax == 0:
            e[1:] |= m; e[:-1] |= m
        else:
            e[:, 1:] |= m; e[:, :-1] |= m
    return distance_transform_edt(~e).astype(np.float32)


def describe(store, d, sel: torch.Tensor, tau_of) -> dict:
    xyz = d["xyz"].to(DEV)[sel]
    nrm = d["normal"].to(DEV)[sel]
    n = len(xyz)
    K = store.K
    H, W = store.rng.shape[1:]
    q_l, inc_l, ed_l, r_l, beh_l, who_l = [], [], [], [], [], []
    for f in range(store.n):
        rng = store.rng[f].to(torch.float32) * 1e-3
        if store.rejected[f] is not None:
            rng = torch.where(store.rejected[f].to(DEV), torch.zeros_like(rng), rng)
        T = store.T[f]
        cam = xyz @ T[:3, :3].T + T[:3, 3]
        z = cam[:, 2]
        ok = z > 0.1
        zs = torch.where(ok, z, torch.ones_like(z))
        u = torch.floor(cam[:, 0] / zs * K.fx + K.cx - K.offset + 0.5).to(torch.int64)
        v = torch.floor(cam[:, 1] / zs * K.fy + K.cy - K.offset + 0.5).to(torch.int64)
        ok &= (u >= 0) & (v >= 0) & (u < W) & (v < H)
        idx = torch.nonzero(ok).squeeze(1)
        if not len(idx):
            continue
        d0 = rng[v[idx], u[idx]]
        good = d0 > 0
        idx, d0 = idx[good], d0[good]
        if not len(idx):
            continue
        c = cam[idx]
        q = torch.linalg.norm(c, dim=1)
        ray_w = (c / q[:, None]) @ T[:3, :3]                     # ray direction in the world frame
        cosang = (nrm[idx] * ray_w).sum(1).abs().clamp(max=1.0)
        inc = torch.rad2deg(torch.arccos(cosang))
        ed = torch.as_tensor(edge_distance(store.rng[f].cpu().numpy()), device=DEV)[v[idx], u[idx]]
        r = d0 - q
        q_l.append(q); inc_l.append(inc); ed_l.append(ed); r_l.append(r); beh_l.append(r > tau_of(q, idx)); who_l.append(idx)
    who = torch.cat(who_l)
    order = torch.argsort(who, stable=True)
    who = who[order]
    cols = {k: torch.cat(v)[order] for k, v in (("q", q_l), ("inc", inc_l), ("edge", ed_l), ("r", r_l), ("behind", beh_l))}
    bounds = torch.searchsorted(who, torch.arange(n + 1, device=DEV))
    out = {}
    per = {k: np.full(n, np.nan, np.float32) for k in ("q", "inc", "edge", "r_behind_cm", "behind_share", "frames")}
    wn = who.cpu().numpy(); b = bounds.cpu().numpy()
    vals = {k: v.cpu().numpy() for k, v in cols.items()}
    for k, src in (("q", "q"), ("inc", "inc"), ("edge", "edge")):
        x = vals[src]
        for i in range(n):                     # per element median over its frames
            s, e = b[i], b[i + 1]
            if e > s:
                per[k][i] = np.nanmedian(x[s:e]) if np.isfinite(x[s:e]).any() else np.nan
    beh, r = vals["behind"], vals["r"]
    for i in range(n):
        s, e = b[i], b[i + 1]
        per["frames"][i] = e - s
        if e > s:
            per["behind_share"][i] = beh[s:e].mean()
            if beh[s:e].any():
                per["r_behind_cm"][i] = 100 * np.median(r[s:e][beh[s:e]])
    for k, x in per.items():
        x = x[np.isfinite(x)]
        out[k] = {str(qq): round(float(np.quantile(x, qq)), 3) for qq in Q} if len(x) else None
    out["elements"] = int(n)
    out["no_normal_share"] = round(float((~torch.isfinite(nrm).all(1)).float().mean()), 4) if n else None
    out["background_share"] = round(float((d["identity"].to(DEV)[sel] == 0).float().mean()), 4) if n else None
    return out


def main():
    args = sys.argv[1:]
    dump, out = Path(args[0]), Path(args[1])
    h0 = float(args[args.index("--half") + 1]) if "--half" in args else 0.0
    d = torch.load(dump, weights_only=False)
    store = SimpleNamespace(n=len(d["T"]), rng=torch.as_tensor(d["rng"].astype(np.int32), device=DEV),
                            T=d["T"].to(DEV), K=Intrinsics(**d["K"]),
                            rejected=[None if r is None else r.to(DEV) for r in d["rejected"]])
    xyz, ext, own_half = d["xyz"].to(DEV), d["extent"].to(DEV), d["half"].to(DEV)
    sigma = d["sigma"]
    res = {"dump": str(dump), "frames": store.n, "tested": len(xyz), "start": d["start"]}
    runs = {}
    for name, half in ((f"half={h0}", torch.full_like(ext, h0)), ("half=dump", own_half)):
        ev = session_end.memory_test(store, xyz, half, ext, sigma, store.rejected)
        # displaced needs the step-2 surface, not in the dump: compared only when error_per_metre = 0 (displaced off)
        seen, hidden, displaced = session_end.decide(ev, xyz, half, sigma, d["error_per_metre"], None)
        runs[name] = (seen, half, ev)
        res[name] = dict(seen_through=int(seen.sum()), hidden=int(hidden.sum()), hit_any=int((ev["hit"] > 0).sum()),
                         through_any=int((ev["through"] > 0).sum()))
    res["dump_own_counts"] = dict(seen_through=int(d["seen_through"].sum()), hidden=int(d["hidden"].sum()))
    seen0, half0, ev0 = runs[f"half={h0}"]
    sig = torch.as_tensor(np.atleast_1d(sigma).astype(np.float32), device=DEV)
    tau_of = lambda q, idx, h=half0: torch.maximum(h[idx], sig[torch.clamp((q / session_end.RANGE_BIN).floor().long(),
                                                                          max=len(sig) - 1)])
    control = (ev0["hit"] > 0) & ~seen0
    res["seen_through_at_half"] = describe(store, d, torch.nonzero(seen0).squeeze(1), tau_of)
    gen = torch.Generator(device="cpu").manual_seed(0)
    ci = torch.nonzero(control).squeeze(1)
    n_seen = int(seen0.sum())
    if len(ci) > n_seen > 0:                      # a random control sample of the same size
        ci = ci[torch.randperm(len(ci), generator=gen)[:n_seen].to(ci.device)]
    res["control_sampled_from"] = int(control.sum())
    res["control_hit_kept"] = describe(store, d, ci, tau_of)
    res["kept_by_dump_half"] = int((seen0 & ~runs["half=dump"][0]).sum())
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
