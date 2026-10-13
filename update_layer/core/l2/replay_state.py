"""Consistency test of the ported state machine (state.py) against a TSDF run (L2_FINAL2 khronos.log).

  python -m update_layer.core.l2.replay_state KHRONOS_LOG {real|synthetic} [--prev-stats PATH]
        [--init PREVIOUS_FINAL.json] [--dump-final OUT.json]

The port is driven by the calls the C++ made, read from the log, and every line the port writes in the
original's format (MOBILITY_PRIOR, EVIDENCE, OVERLAP_POSTERIOR, EMPTY_INTERVAL_POSTERIOR, INGEST,
INGEST_DECIDE, EXTENT_POSTERIOR, SAME_STATE, MATERIALIZE_POSTERIOR, FINALIZE, *_CLOSE, *_ABSORB, ...)
must equal the line persistent_object_state.cpp wrote, field by field (6 significant digits).

Inputs taken from the log, i.e. not tested here:
  - the round evidence of each measured state (STATE_EVIDENCE_WINDOW) and each look's measured absence
    ratio (rebuilt from ABSENCE_LOOK with the absence arithmetic that replay_absence.py verifies);
  - each segment's interval and motion flag (OBJECT_EXTRACTION_INPUT, STATIC_SURFACE_FRAMES), vertex
    count (INGEST seg_verts) and semantic class (MOBILITY_PRIOR class=);
  - the values of the geometric kernels (shared-space lower bound, extent probability, off-state
    share and effective cells), which need the meshes; the kernels are checked separately.
Everything else (the M2a prior and its population, every decision, every union, close, hand-over,
absorption and fold, the vertex counts of every state slot, the mobility counts) is the port's.
Session B/C start from the previous replay's final registry (--init), as initializeFromObjects does.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict, deque

from .evidence import LOG99, ObjectAbsenceState, finite_count_log_ratio, present_beta, present_log_density, \
    AbsenceModel
from .state import Observation, PersistentObjectState, PhysicalState, SurfaceEvidence

LINE = re.compile(r"(\w+\.cpp):\d+\] ([A-Z][A-Z_0-9]+)(?: (.*))?$")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r"(\w+)=([^ ]+)")
CONFIG = {"real": dict(resolution=0.05, high_mobility=[10, 15, 74, 75, 92, 115, 131, 139]),
          "synthetic": dict(resolution=0.1, high_mobility=[6, 9, 15, 35, 39])}
POSTERIOR_TAGS = ("SAME_STATE", "MATERIALIZE_POSTERIOR", "FINALIZE", "TOP_ABSORB", "SESSION_ABSORB")


def fields(rest: str) -> dict:
    return {k: v for k, v in FIELD.findall(rest or "")}


def parse(path):
    out = []
    for raw in open(path, errors="replace"):
        m = LINE.search(ANSI.sub("", raw.rstrip()))
        if m:
            out.append((m.group(1), m.group(2), m.group(3) or ""))
    return out


class VCount:
    """Stand-in geometry: a vertex count (the replay has no meshes)."""
    def __init__(self, n: int):
        self.n = int(n)

    def __len__(self):
        return self.n


class AbsenceReplay:
    """replay_absence.py's per-record arithmetic, also returning each scored look's measured ratio
    (weight of the strongest candidate x finite-count ratio, as applyObservedAbsence stores it)."""

    def __init__(self, prev_stats):
        self.model = AbsenceModel(store=None)
        if prev_stats and not self.model.load_sensor_statistics(prev_stats):
            raise SystemExit(f"cannot read {prev_stats}")
        self.states, self.latest_record, self.pending = {}, {}, {}
        self.committed_before_record = set()

    @staticmethod
    def _reset(st):
        st.cusum, st.page, st.first_window = 0.0, [], False

    def look(self, d):
        inst, slot, rec = int(d["inst"]), int(d["slot"]), int(d["record"])
        new_record = rec not in self.states
        st = self.states.setdefault(rec, ObjectAbsenceState())
        if new_record and (inst, slot) in self.committed_before_record:
            self.committed_before_record.discard((inst, slot))
            self._reset(st)
        if self.pending.get(rec) and st.cusum > LOG99:
            self._reset(st)
        k, n, reliable = int(d["k"]), int(d["n"]), int(d["reliable"])
        f = k / n
        beta = present_beta(*self.model.present_moments(st))
        llr = -present_log_density(beta, f)
        cum = [int(x) for x in d["first_since"].split(",")]
        t = len(st.page)
        weight = lambda s: min(1.0, cum[s] / max(1, reliable))
        strongest = t
        for s in range(t):
            if st.page[s] > (0.0 if strongest == t else st.page[strongest]):
                strongest = s
        st.page.append(0.0)
        for s in range(t + 1):
            st.page[s] += weight(s) * llr
        st.cusum = max(0.0, max(st.page))
        w = weight(strongest)
        lik = (w * finite_count_log_ratio(float(k), float(n), beta), True, True) if w > 0.0 else (0.0, False, True)
        if d["identified"] == "1":
            self.model.learn_in_place_look(st, f)
        self.latest_record[(inst, slot)] = rec
        self.pending[rec] = True
        return lik

    def commit(self, d):
        inst, slot = int(d["inst"]), int(d["slot"])
        rec = self.latest_record.get((inst, slot))
        committed = d["committed"] == "1"
        if rec is not None and self.pending.get(rec):
            if committed:
                self._reset(self.states[rec])
            self.pending[rec] = False
        elif committed:
            if rec is not None:
                self._reset(self.states[rec])
            else:
                self.committed_before_record.add((inst, slot))


class ReplayState(PersistentObjectState):
    """The port with the geometric kernels answered from the log (the next logged result)."""

    def __init__(self, oracle, **kw):
        super().__init__(**kw)
        self.oracle = oracle

    def _shared_space(self, current, obs, decision):
        return float(self.oracle("OVERLAP_POSTERIOR", obs.identity)["shared_probability_lower_bound"])

    def _extent_same_site(self, current, candidate):
        return float(self.oracle("EXTENT_POSTERIOR", None)["same_site_probability"])

    def _off_state_share(self, copy, reference):
        d = self.oracle(POSTERIOR_TAGS, None)
        return float(d["off_share"]), float(d["effective_cells"])

    @staticmethod
    def _copy_geometry(obs):
        return VCount(len(obs.points)), None, None

    @staticmethod
    def _append(target, points, normals, elements):
        target.points = VCount(len(target.points) + len(points))


def close(a: float, b: float, rel: float = 5e-6) -> bool:
    if math.isinf(a) or math.isinf(b) or math.isnan(a) or math.isnan(b):
        return str(a) == str(b) or (math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0))
    return abs(a - b) <= rel * max(1.0, abs(b)) + 1e-9


def same_line(port: str, logged: str):
    """None if equal field by field; else the differing fields."""
    tp, _, rp = port.partition(" ")
    tl, _, rl = logged.partition(" ")
    if tp != tl:
        return [("tag", tp, tl)]
    fp, fl = fields(rp), fields(rl)
    diff = []
    for k in set(fp) | set(fl):
        a, b = fp.get(k), fl.get(k)
        if a == b:
            continue
        try:
            fa, fb = float(a.rstrip("sv")), float(b.rstrip("sv"))
        except (AttributeError, ValueError):
            diff.append((k, a, b))
            continue
        # factors computed from a logged (rounded) off share: the logged share has 6 digits
        rel = 1e-3 if k in ("bayes_factor_or_bound", "geometry_factor_or_bound") else 5e-6
        if not close(fa, fb, rel):
            diff.append((k, a, b))
    return diff or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("dataset", choices=sorted(CONFIG))
    ap.add_argument("--prev-stats", default="")
    ap.add_argument("--init", default="")
    ap.add_argument("--dump-final", default="")
    ap.add_argument("--max-report", type=int, default=25)
    args = ap.parse_args()
    lines = parse(args.log)
    cfg = CONFIG[args.dataset]

    expected = [(i, f"{tag} {rest}") for i, (src, tag, rest) in enumerate(lines)
                if src == "persistent_object_state.cpp"]
    exp_pos = {i: j for j, (i, _) in enumerate(expected)}       # log line index -> expected index
    ptr = [0]                                                   # next expected line to match

    def oracle(tags, inst):
        drain()                        # align with the logged lines the port has written so far
        tags = (tags,) if isinstance(tags, str) else tags
        for j in range(ptr[0], len(expected)):
            text = expected[j][1]
            tag = text.split(" ", 1)[0]
            if tag in tags:
                d = fields(text.split(" ", 1)[1])
                if d.get("off_share") == "unmeasured":
                    continue
                if inst is not None and d.get("inst") != str(inst):
                    continue
                return d
        raise RuntimeError(f"oracle: no logged {tags} for inst {inst} after expected #{ptr[0]}")

    reg = ReplayState(oracle, map_resolution=cfg["resolution"], high_mobility_semantics=cfg["high_mobility"])
    absence = AbsenceReplay(args.prev_stats)

    # semantic class of each identity (the class of its CURRENT in the MOBILITY_PRIOR lines)
    label_of = {}
    for src, tag, rest in lines:
        if tag == "MOBILITY_PRIOR":
            d = fields(rest)
            label_of.setdefault(int(d["inst"]), int(d["class"]))

    # the terminal round: two verify passes at the drain stamp, each followed by finalizePendingAbsences
    terminal = None
    for src, tag, rest in lines:
        if tag == "TERMINAL_STATE_DRAIN":
            terminal = int(fields(rest)["stamp"])
    finalize_after = set()
    if terminal is not None:
        blocks, block, last_inst, stamp = [], [], None, None
        for i, (src, tag, rest) in enumerate(lines):
            if tag == "STATE_EVIDENCE_WINDOW":
                stamp = int(fields(rest)["through"])
            elif tag == "EVIDENCE" and stamp == terminal:
                inst = int(fields(rest)["inst"])
                if block and (last_inst is not None and inst <= last_inst):
                    blocks.append(block)
                    block = []
                block.append(i)
                last_inst = inst
            elif tag in ("INGEST", "MATERIALIZE_POSTERIOR") and stamp == terminal and block:
                blocks.append(block)
                block, last_inst = [], None
        if block:
            blocks.append(block)
        # a regular round may precede at the same stamp (finishProcessing uses the last stamp received)
        finalize_after = {b[-1] for b in blocks[-2:]}
        if len(blocks) < 2:
            print(f"warning: {len(blocks)} terminal verify blocks found")

    # session start (initializeFromObjects of the previous session's final registry)
    first_after = {}
    for src, tag, rest in lines:
        if tag == "STATE_EVIDENCE_WINDOW":
            d = fields(rest)
            first_after.setdefault(int(d["inst"]), int(d["after"]))
    if args.init:
        objs = json.load(open(args.init))
        for o in objs:
            o["points"] = VCount(o["verts"])
            o["last"] = first_after.get(o["identity"], o["last"])
            o["semantic"] = label_of.get(o["identity"], o["semantic"])   # a class the previous log never printed
        reg.initialize_from_objects(objs)

    bad = []
    n_compared = [0]

    def drain():
        """compare the port's new lines with the logged ones, in order"""
        while reg.log:
            port_line = reg.log.pop(0)
            n_compared[0] += 1
            if ptr[0] >= len(expected):
                bad.append(("extra port line at end", port_line))
                continue
            logged = expected[ptr[0]][1]
            diff = same_line(port_line, logged)
            if diff and diff[0][0] == "tag":
                # resynchronise: the port missed logged lines, or wrote one the C++ did not
                tag_p = port_line.split(" ", 1)[0]
                inst_p = fields(port_line).get("inst")
                for j in range(ptr[0] + 1, min(len(expected), ptr[0] + 8)):
                    t = expected[j][1]
                    if t.split(" ", 1)[0] == tag_p and fields(t).get("inst") == inst_p:
                        for k in range(ptr[0], j):
                            bad.append(("missing in port", lines[expected[k][0]][1], expected[k][1][:200]))
                        ptr[0] = j
                        logged = t
                        diff = same_line(port_line, logged)
                        break
                else:
                    bad.append(("extra in port", port_line[:200], "next logged: " + logged[:200]))
                    continue
            if diff:
                bad.append(("differs", port_line[:240], logged[:240], diff))
            ptr[0] += 1

    segments = defaultdict(deque)       # inst -> extracted segments not yet ingested
    extraction = {}                     # inst -> pending OBJECT_EXTRACTION_INPUT fields
    windows = defaultdict(list)         # inst -> evidence of this verify call
    lik_pending = {}
    verify_stamp = None
    started = False

    for i, (src, tag, rest) in enumerate(lines):
        d = fields(rest)
        if tag == "ABSENCE_LOOK":
            lik_pending[int(d["inst"])] = absence.look(d)
        elif tag == "ABSENCE_COMMIT":
            absence.commit(d)
        elif tag == "OBJECT_EXTRACTION_INPUT":
            extraction[int(d["inst"])] = d
        elif tag == "STATIC_SURFACE_FRAMES":
            inst = int(d["inst"])
            e = extraction.pop(inst, None)
            if e is None:
                bad.append(("frames without extraction input", rest[:120]))
                continue
            moved = e["motion_history"] == "1" and e["static_fallback"] == "0"
            segments[inst].append(dict(first=int(d["first"]), last=max(int(e["last"]), int(d["last"])),
                                       moved=moved, track_first=int(e["first"]),
                                       frames=int(d["selected"])))
        elif tag == "STATE_EVIDENCE_WINDOW":
            inst, stamp = int(d["inst"]), int(d["through"])
            if not reg.event_open or stamp != reg.event_stamp:
                reg.begin_observation_event(stamp)
            verify_stamp = stamp
            cur, scur = reg.current_fragment(inst), reg.session_current_fragment(inst)
            slot = 0 if (not windows[inst] and cur is not None and cur.num_vertices) else 1
            prior = reg.change_prior_log_odds(inst, slot)
            drain()
            if not close(prior, float(d["prior_log_odds"])):
                bad.append(("prior_log_odds", inst, prior, d["prior_log_odds"]))
            lr, measured, calibrated = lik_pending.pop(inst, (0.0, False, True))
            ev = SurfaceEvidence(support_rays=int(d["support"]), contradiction_rays=int(d["contradiction"]),
                                 surface_samples=int(d["total_samples"]),
                                 absence_coverage_sufficient=d["absence_coverage_sufficient"] == "1",
                                 latest_support_stamp=int(d["latest_measured_support"]),
                                 reliable_in_view=int(d["reliable_in_view"]),
                                 reliable_seen_through=int(d["reliable_seen_through"]),
                                 reliable_samples=int(d["reliable"]), measured_absence_log_ratio=lr,
                                 has_measured_absence_likelihood=measured,
                                 has_calibrated_absence_source=calibrated)
            windows[inst].append((slot, ev))
        elif tag == "EVIDENCE" and src == "persistent_object_state.cpp":
            inst = int(d["inst"])
            inh, ses = SurfaceEvidence(), SurfaceEvidence()
            for slot, ev in windows.pop(inst, []):
                if slot == 0:
                    inh = ev
                else:
                    ses = ev
            reg.resolve_current_evidence(inst, inh, ses, verify_stamp)
            drain()
            if i in finalize_after:
                reg.finalize_pending_absences(terminal)
                drain()
        elif tag == "INGEST" and src == "persistent_object_state.cpp":
            j = exp_pos[i]
            prev = expected[j - 1][1] if j > 0 else ""
            inst = int(d["inst"])
            if prev.startswith(f"INGEST_DECIDE inst={inst} inherited_session_deferred=true"):
                continue                                         # the recursive call: the port writes it
            if not started:
                started = True
                if not reg.event_open:
                    reg.begin_observation_event(0)               # session start: empty history
            first_s = int(d["first"].rstrip("s"))
            q = segments[inst]
            seg = None
            for k, s in enumerate(q):
                if s["first"] // 1000000000 == first_s:
                    seg = s
                    del q[k]
                    break
            if seg is None:
                bad.append(("no extracted segment for INGEST", rest))
                seg = dict(first=first_s * 1000000000, last=first_s * 1000000000, moved=False, track_first=0,
                           frames=0)
            obs = Observation(identity=inst, points=VCount(int(d["seg_verts"])), first=seg["first"],
                              last=seg["last"], semantic=label_of.get(inst, -1), moved=seg["moved"],
                              reconstruction_frames=seg["frames"], track_first_seen=seg["track_first"])
            state = reg.states.setdefault(inst, PhysicalState())
            reg.ingest_observation(state, obs, inst)
            drain()
        elif tag == "MATERIALIZE_POSTERIOR":
            reg.materialize(int(d["inst"]))
            drain()
        if reg.event_open and reg.event_stamp == 0 and tag == "MOBILITY_PRIOR":
            reg.event_stamp = int(d["event_stamp"])              # the first event's stamp, known late
    drain()
    for k in range(ptr[0], len(expected)):
        bad.append(("missing in port (end)", expected[k][1][:200]))

    print(f"{args.log}: expected lines {len(expected)}, port lines compared {n_compared[0]}, "
          f"mismatches {len(bad)}")
    by_kind = defaultdict(int)
    for b in bad:
        by_kind[b[0] if b[0] != "differs" else "differs " + b[1].split(' ', 1)[0]] += 1
    for k, v in sorted(by_kind.items()):
        print(f"  {k}: {v}")
    for b in bad[:args.max_report]:
        print("  ", b)

    if args.dump_final:
        objs = []
        for inst in reg.tracked_ids():
            st = reg.states[inst]
            if not st.fragments:
                continue
            m = reg.materialize(inst)
            reg.log.clear()
            verts = sum(f.num_vertices for f in m.fragments) if m and m.present else 0
            cur = st.fragments[st.current] if st.current is not None else st.fragments[-1]
            objs.append(dict(identity=inst, verts=verts, first=min(f.birth_time for f in st.fragments),
                             last=cur.last_support_time, semantic=cur.semantic_label,
                             dynamic=st.has_dynamic_history, mobility_changes=st.mobility_changes,
                             mobility_continuations=st.mobility_continuations))
        json.dump(objs, open(args.dump_final, "w"), indent=1)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
