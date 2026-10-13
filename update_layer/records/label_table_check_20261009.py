# Offline exactness check of LabelTable against the dense table on REAL checkpoint tables (synthetic A: 2.97 M x 149,
# real B: 2.58 M x 18): every operation the backend uses, compared bitwise with the dense torch path.
import sys, time, torch
from update_layer.backends.game.label_table import LabelTable
torch.manual_seed(0)
def eq(a, b): return torch.equal(a, b)
for name, f in [("syn A", sys.argv[1]), ("real B", sys.argv[2])]:
    ck = torch.load(f, map_location="cpu", weights_only=False); lw_csr = ck["backend"]["label_weight"]; del ck
    dense = lw_csr.to_dense().cuda(); n, m = dense.shape
    t0 = time.time(); T = LabelTable.from_tensor(lw_csr); torch.cuda.synchronize()
    print(f"== {name}: {n} x {m}, nnz {T.nnz()} ({T.nnz()/n:.2f}/row), table {T.nnz()*20/1e6:.0f} MB vs dense {dense.numel()*4/1e9:.2f} GB; load {time.time()-t0:.1f}s")
    ok = []
    ok.append(("roundtrip dense", eq(T.to_dense(), dense)))
    # max(dim=1): the vote readout
    bd, jd = dense.max(dim=1); bt, jt = T.max(dim=1)
    ok.append(("max values", eq(bd, bt))); ok.append(("max index (incl. ties/empty rows)", eq(jd, jt)))
    print(f"   rows with no entry: {int((bd == 0).sum())}; dense argmax of those: {sorted(set(jd[bd == 0].tolist()))[:5]}")
    # += dense delta shaped like a render's used_count: ~15 % of rows get 1-3 labels, half of them on existing labels
    delta = torch.zeros_like(dense)
    r = torch.nonzero(torch.rand(n, device="cuda") < 0.15).squeeze(1)
    for k in range(3):
        rr = r[torch.rand(len(r), device="cuda") < (1.0 if k == 0 else 0.4)]
        c = torch.randint(0, m, (len(rr),), device="cuda")
        if k == 0:                                      # reuse an existing label of the row for half of them
            ex = jd[rr]; use = torch.rand(len(rr), device="cuda") < 0.5; c = torch.where(use, ex, c)
        delta[rr, c] += torch.rand(len(rr), device="cuda") * 3
    d2 = dense + delta; t0 = time.time(); T2 = T.add_dense(delta); torch.cuda.synchronize(); ta = time.time() - t0
    ok.append((f"+= delta ({int((delta != 0).sum())} entries, {ta*1e3:.0f} ms)", eq(T2.to_dense(), d2)))
    b2, j2 = d2.max(dim=1); bt2, jt2 = T2.max(dim=1); ok.append(("max after +=", eq(b2, bt2) and eq(j2, jt2)))
    # prune: keep ~70 % of rows
    keep = torch.rand(n, device="cuda") < 0.7
    ok.append(("rows[keep]", eq(T2[keep].to_dense(), d2[keep])))
    # densify split: parents with duplicates (N=2 children per parent), then cat rows
    par = torch.nonzero(torch.rand(n, device="cuda") < 0.02).squeeze(1); par = par.repeat(2)
    ok.append(("rows[par] (dup parents)", eq(T2[par].to_dense(), d2[par])))
    ok.append(("append_rows", eq(T2.append_rows(T2[par]).to_dense(), torch.cat([d2, d2[par]]))))
    ok.append(("append_empty_rows", eq(T2.append_empty_rows(1000).to_dense(), torch.cat([d2, d2.new_zeros((1000, m))]))))
    ok.append(("add_columns", eq(T2.add_columns(3).to_dense(), torch.cat([d2, d2.new_zeros((n, 3))], 1))))
    # checkpoint form roundtrip (CSR), bitwise
    csr = T2.to_sparse_csr(); ok.append(("to_sparse_csr -> from_tensor", eq(LabelTable.from_tensor(csr).to_dense(), d2)))
    ok.append(("csr dense equal", eq(csr.to_dense().cuda(), d2)))
    # empty-column table (M3) and (0,1) init
    ok.append(("new_zeros (n,0)", T2.new_zeros((n, 0)).shape == (n, 0) and LabelTable(0, 1).shape == (0, 1)))
    for k, v in ok: print(f"   {'PASS' if v else 'FAIL'} {k}")
    print(f"   peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB (dense copies included)")
    del dense, d2, delta, T, T2; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
