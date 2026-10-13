"""Store a GaME checkpoint's label sums (I1 label_weight) sparse (CSR), losslessly; prior_from_state densifies.

  python -m update_layer.eval.ckpt_pack CKPT

Writes CKPT.tmp with label_weight as stored by game._label_weight_store, reloads it and checks that the dense label
weights are bitwise equal to the original, then replaces CKPT; everything else is copied unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from update_layer.backends.game.game import _label_weight_store  # noqa: E402


def main():
    p = Path(sys.argv[1])
    ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    lw = ck["backend"].get("label_weight")
    if lw is None or lw.layout != torch.strided:
        print(f"{p}: nothing to pack"); return
    packed = _label_weight_store(lw)
    if packed.layout == torch.strided:
        print(f"{p}: dense is smaller; unchanged"); return
    new = dict(ck); new["backend"] = dict(ck["backend"]); new["backend"]["label_weight"] = packed
    tmp = p.with_name(p.name + ".tmp")
    torch.save(new, tmp)
    del new
    back = torch.load(tmp, map_location="cpu", weights_only=False, mmap=True)
    ok = torch.equal(back["backend"]["label_weight"].to_dense(), lw) and set(back["backend"]) == set(ck["backend"])
    if not ok:
        tmp.unlink(); sys.exit(f"{p}: verification FAILED; unchanged")
    size = p.stat().st_size
    tmp.replace(p)
    print(f"{p}: label_weight {tuple(lw.shape)} -> CSR, verified bitwise; {size / 1e9:.2f} GB -> {p.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
