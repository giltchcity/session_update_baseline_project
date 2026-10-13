"""Store an existing post_ref timeline as a tail of its pre_ref timeline (scenelist.TimelineTail), losslessly.

  python -m update_layer.eval.timeline_tail SESSION_DIR

SESSION_DIR/timeline_post_ref.pkl = timeline.pkl's snapshots except its last + the re-rendered final one (run.py F2).
Writes the tail to a temporary file, checks that load_timeline(tail) equals the original post_ref timeline snapshot by
snapshot (stamps equal, every scene's pickle bytes equal), then replaces the original; otherwise leaves it unchanged.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from update_layer.eval.scenelist import SceneListTimeline, TimelineTail, load_timeline  # noqa: E402


def main():
    d = Path(sys.argv[1])
    post_p, pre_p = d / "timeline_post_ref.pkl", d / "timeline.pkl"
    with open(post_p, "rb") as f:
        post = pickle.load(f)
    if isinstance(post, TimelineTail):
        print(f"{post_p}: already a tail"); return
    pre = load_timeline(pre_p)
    k = len(pre.stamps()) - 1
    if not (isinstance(post, SceneListTimeline) and post.stamps()[:k] == pre.stamps()[:k] and len(post.stamps()) == k + 1):
        sys.exit(f"{post_p}: not pre_ref[:-1] + one snapshot; unchanged")
    tail = TimelineTail("timeline.pkl", k, post.stamps()[k:], post.scenes[k:])
    tmp = d / "timeline_post_ref.pkl.tmp"
    tail.save(tmp)
    back = load_timeline(tmp)
    ok = back.stamps() == post.stamps() and all(pickle.dumps(x, protocol=4) == pickle.dumps(y, protocol=4)
                                                for x, y in zip(back.scenes, post.scenes))
    if not ok:
        tmp.unlink()
        sys.exit(f"{post_p}: verification FAILED; unchanged")
    size = post_p.stat().st_size
    tmp.replace(post_p)
    print(f"{post_p}: {len(post.stamps())} snapshots verified identical; {size / 1e9:.2f} GB -> {post_p.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
