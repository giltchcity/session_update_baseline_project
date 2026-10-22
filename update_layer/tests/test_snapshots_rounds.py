"""The 4D map is every round's live map (snapshots.py): the map of time t is the map of the latest round at or before
t, bit for bit as it was recorded, across sessions (a tie: the later session), in any query order; none before the
first round."""
import tempfile
from pathlib import Path

import numpy as np
from update_layer.backends.game.snapshots import WIDTH, SnapshotLog, map_at


def test_map_of_time_t_is_the_latest_round_at_or_before_t():
    rng = np.random.default_rng(0)
    rounds = {"s_a": {10: [1, 2, 3, 10], 20: [1, 2, 3, 10, 11], 30: [1, 2, 3, 11, 12]},
              "s_b": {30: [1, 2, 3, 11, 12], 40: [1, 2, 3, 12, 20]}}
    with tempfile.TemporaryDirectory() as d:
        run, kept = Path(d), {}
        for ses, rs in rounds.items():
            log = SnapshotLog(run / "snapshots" / ses)
            for stamp, uid in rs.items():
                uid = np.array(uid)[::-1]                                  # any row order
                p = rng.standard_normal((len(uid), WIDTH)).astype(np.float32)
                ident = uid % 7
                log.record(stamp, uid, p, ident, uid * 0 + stamp, uid * 0)
                kept[(ses, stamp)] = dict(zip(uid.tolist(), p))
        want = {10: ("s_a", 10), 15: ("s_a", 10), 29: ("s_a", 20), 30: ("s_b", 30), 35: ("s_b", 30), 40: ("s_b", 40),
                99: ("s_b", 40)}
        for order in (sorted(want), sorted(want, reverse=True), [35, 99, 30, 15, 40, 10, 29]):
            cache = {}
            for t in order:
                stamp, uid, p, ident = map_at(run, t, cache)
                ses, s = want[t]
                assert stamp == s and sorted(kept[(ses, s)]) == uid.tolist()
                assert all(np.array_equal(p[i].view(np.uint32), kept[(ses, s)][u].view(np.uint32)) for i, u in enumerate(uid.tolist()))
                assert np.array_equal(ident, uid % 7)
        assert map_at(run, 9) is None
