"""LabelTable (sparse I1 label sums) gives the dense table's values: every operation the backend uses, bitwise, on
random tables with the real tables' structure (1-2 non-zeros per row, ties, empty rows). CPU."""
import torch

from update_layer.backends.game.label_table import LabelTable


def _random_dense(n, m, g):
    d = torch.zeros((n, m))
    for _ in range(2):
        r = torch.nonzero(torch.rand(n, generator=g) < 0.8).squeeze(1)
        d[r, torch.randint(0, m, (len(r),), generator=g)] += torch.randint(1, 4, (len(r),), generator=g).float()
    return d


def test_label_table_matches_dense():
    g = torch.Generator().manual_seed(0)
    for n, m in ((500, 7), (3000, 149)):
        d = _random_dense(n, m, g)
        T = LabelTable.from_tensor(d, device="cpu")
        assert torch.equal(T.to_dense(), d) and len(T) == n and T.shape == (n, m)
        bd, jd = d.max(dim=1); bt, jt = T.max(dim=1)
        assert torch.equal(bd, bt) and torch.equal(jd, jt)            # ties -> first column, empty rows -> (0, 0)
        delta = _random_dense(n, m, g) * 0.37
        d2 = d + delta; T2 = T.add_dense(delta)
        assert torch.equal(T2.to_dense(), d2)
        keep = torch.rand(n, generator=g) < 0.6
        assert torch.equal(T2[keep].to_dense(), d2[keep])
        par = torch.nonzero(torch.rand(n, generator=g) < 0.1).squeeze(1).repeat(2)
        assert torch.equal(T2[par].to_dense(), d2[par])
        assert torch.equal(T2.append_rows(T2[par]).to_dense(), torch.cat([d2, d2[par]]))
        assert torch.equal(T2.append_empty_rows(5).to_dense(), torch.cat([d2, d2.new_zeros((5, m))]))
        assert torch.equal(T2.add_columns(2).to_dense(), torch.cat([d2, d2.new_zeros((n, 2))], 1))
        csr = T2.to_sparse_csr()
        assert csr.layout == torch.sparse_csr and torch.equal(LabelTable.from_tensor(csr, device="cpu").to_dense(), d2)
        assert T2.new_zeros((n, 0)).shape == (n, 0)


if __name__ == "__main__":
    test_label_table_matches_dense(); print("ok")
