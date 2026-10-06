"""Consistency test of the ported observed-absence arithmetic against a TSDF run (L2_FINAL2 khronos.log).

  python -m update_layer.core.l2.replay_absence KHRONOS_LOG [PREVIOUS_SENSOR_STATISTICS]

Every scored look of the run (ABSENCE_LOOK: k, n, reliable, the per-candidate first-judgement counts, whether
the object stood in place) is fed in log order through the port (present_moments, present_beta,
present_log_density, Page's update with candidate weights, in-place learning into the pooled population,
commitment reset); the logged llr and cusum_before, and the ABSENCE_COMMIT before_session / page_statistic /
committed fields, must be reproduced. Frame classification (which samples a look judged) is not replayed here:
its inputs, the stored pixels, are not in the log.
"""
from __future__ import annotations

import math
import re
import sys

from .evidence import LOG99, AbsenceModel, ObjectAbsenceState, present_beta, present_log_density

FIELD = re.compile(r"(\w+)=([^ ]+)")


def parse(line: str) -> dict:
    return {k: v for k, v in FIELD.findall(line.rstrip())}


def close(a: float, b: float) -> bool:
    """Logged values carry 6 significant digits (glog's default stream precision)."""
    return abs(a - b) <= 5e-6 * max(1.0, abs(b)) + 1e-9


def main():
    log_path = sys.argv[1]
    model = AbsenceModel(store=None)
    if len(sys.argv) > 2:
        model.load_sensor_statistics(sys.argv[2])
    states = {}
    latest_record = {}
    pending = {}
    committed_before_record = set()      # (inst, slot) committed by the prior alone before its first scored look
    n_look = n_commit = 0
    bad = []

    def reset(st):
        st.cusum = 0.0
        st.page = []
        st.first_window = False

    for line in open(log_path, errors="replace"):
        if "] ABSENCE_LOOK " in line:
            d = parse(line)
            inst, slot, rec = int(d["inst"]), int(d["slot"]), int(d["record"])
            new_record = rec not in states
            st = states.setdefault(rec, ObjectAbsenceState())
            if new_record and (inst, slot) in committed_before_record:
                committed_before_record.discard((inst, slot))
                reset(st)
            if pending.get(rec) and st.cusum > LOG99:     # previous call had prior 0: committed by the Page statistic
                reset(st)
            k, n, reliable = int(d["k"]), int(d["n"]), int(d["reliable"])
            f = k / n
            beta = present_beta(*model.present_moments(st))
            llr = -present_log_density(beta, f)
            if not close(llr, float(d["llr"])) or not close(st.cusum, float(d["cusum_before"])):
                bad.append((line.split("] ")[0][-30:], inst, rec, "llr", llr, d["llr"], "cusum", st.cusum,
                            d["cusum_before"]))
            cum = [int(x) for x in d["first_since"].split(",")]
            t = len(st.page)
            if len(cum) != t + 1:
                bad.append(("first_since length", inst, rec, len(cum), t + 1))
            weight = lambda s: min(1.0, cum[s] / max(1, reliable))
            st.page.append(0.0)
            for s in range(t + 1):
                st.page[s] += weight(s) * llr
            st.cusum = max(0.0, max(st.page))
            if d["identified"] == "1":
                model.learn_in_place_look(st, f)
            latest_record[(inst, slot)] = rec
            pending[rec] = True
            n_look += 1
        elif "] ABSENCE_COMMIT " in line:
            d = parse(line)
            inst, slot = int(d["inst"]), int(d["slot"])
            rec = latest_record.get((inst, slot))
            committed = d["committed"] == "1"
            if rec is not None and pending.get(rec):
                st = states[rec]
                prior = float(d["prior_log_odds"])
                before = (prior + (st.page[0] if st.page else 0.0)) if st.first_window else -math.inf
                # before_session uses the logged (6-digit) prior: allow its rounding
                if not close(st.cusum, float(d["page_statistic"])) or \
                        (math.isfinite(before) and abs(before - float(d["before_session"])) > 2e-5 * max(1.0, abs(before))):
                    bad.append(("commit", inst, rec, st.cusum, d["page_statistic"], before, d["before_session"]))
                if committed != (max(before, st.cusum) > LOG99):
                    bad.append(("committed flag", inst, rec, committed, max(before, st.cusum)))
                if committed:
                    reset(st)
                pending[rec] = False
                n_commit += 1
            elif committed:                                # an unscored round committed by the prior alone
                if rec is not None:
                    reset(states[rec])
                else:
                    committed_before_record.add((inst, slot))
    print(f"{log_path}: looks {n_look}, commit lines matched {n_commit}, mismatches {len(bad)}")
    for b in bad[:15]:
        print("  ", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
