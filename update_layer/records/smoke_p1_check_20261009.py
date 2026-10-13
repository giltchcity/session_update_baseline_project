# Checks on the P1-on + sparse-table smoke checkpoints: P1 rows present in B, label table CSR -> LabelTable -> CSR bitwise,
# table row count == model rows, every per-row column the same length. Exit 1 on any failure.
import sys, torch
sys.path.insert(0, sys.argv[2])
from update_layer.backends.game.label_table import LabelTable
MAX = torch.iinfo(torch.int64).max
ok = True
for s in "ab":
    ck = torch.load(f"{sys.argv[1]}/checkpoint_{s}.pt", map_location="cpu", weights_only=False); b = ck["backend"]
    n = b["uid"].numel(); lw = b["label_weight"]
    dp = b["death_prune"]; p1 = int((dp < MAX).sum()); fr = int(b["frozen"].sum())
    T = LabelTable.from_tensor(lw, device="cpu"); back = T.to_sparse_csr()
    rt = lw.layout == torch.sparse_csr and torch.equal(back.to_dense(), lw.to_dense()) and T.shape == tuple(lw.shape)
    cols_ok = all(b[k].shape[0] == n for k in ("identity", "created", "death_state", "death_evidence", "state_birth", "death_prune", "frozen", "last_update")) and lw.shape[0] == n
    print(f"{s}: rows {n}, labels {len(b['label_ids'])}, table {lw.layout} nnz {T.nnz()} roundtrip_bitwise={rt} columns_consistent={cols_ok} P1(death_prune set)={p1} frozen={fr}")
    ok &= rt and cols_ok
    if s == "b": ok &= p1 > 0
print("CHECKS", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
