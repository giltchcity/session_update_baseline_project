# Log-level replay of the observed-absence gate under alternative present-share scales (no run, no GT selection).
# Per object/record: looks (k, n, reliable) and windows (support, contradiction) from the layer log; the gate is the page
# statistic of -log density(f | present beta) weighted by the judged fraction, > LOG99; a closure = gate open at a
# window with contradiction > support (resolve_support_dominance). Validation: the current scale must reproduce the
# logged llr and the logged committed flags. GT is used only at the end to label predicted closures.
import re, math, sys, csv
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.core.l2 import evidence as E
R = "/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"
LOG99 = math.log(99.0)
look_re = re.compile(r"^ABSENCE_LOOK inst=(\d+) record=(\d+) stamp=(\d+) k=(\d+) n=(\d+) reliable=(\d+) own=(\d+) identified=(\d) inherited=\d llr=([-\d.e+]+) cusum_before=([-\d.e+]+)")
win_re = re.compile(r"^STATE_EVIDENCE_WINDOW inst=(\d+) slot=(\d+) after=(\d+) through=(\d+) support=(\d+) contradiction=(\d+) .*absence_llr=([-\d.e+]+) prior_log_odds=([-\d.e+]+) .*absence_coverage_sufficient=(\d)")
close_re = re.compile(r"^(TOP|INHERITED)_CLOSE inst=(\d+) .*by_observed_absence=(\d)")

def pooled_stats(lines):
    hist, pooled = {}, []
    for line in lines:
        m = look_re.match(line)
        if not m or m.group(8) != "1": continue
        key = (m.group(1), m.group(2)); f = int(m.group(4)) / int(m.group(5))
        h = hist.setdefault(key, []); h.append(f)
        if len(h) == 3: pooled.append(sum(h) / 3)
    n = len(pooled); mean = sum(pooled) / n
    med = sorted(pooled)[n // 2]
    mad = sorted(abs(x - med) for x in pooled)[n // 2]
    meanad = sum(abs(x - med) for x in pooled) / n
    var = sum((x - mean) ** 2 for x in pooled) / n
    return dict(n=n, mean=mean, mad_sd=1.4826 * mad, meanad=meanad, sd=math.sqrt(var))

def beta_from(mean, v):
    m = min(0.999, max(0.001, mean)); v = max(1e-12, v)
    c = max(E.K_UNIFORM_PRIOR_PSEUDO_COUNT, m * (1 - m) / v - 1)
    return E.PresentBeta(m, m * c + 1e-3, (1 - m) * c + 1e-3)

def scales(st):
    v = max(E.K_SHARE_VARIANCE_FLOOR, st["mad_sd"] ** 2)
    sd_floor = math.sqrt(E.K_SHARE_VARIANCE_FLOOR)
    return {"current(MAD+floor)": E.present_beta(st["mean"], st["mean"] ** 2 + v),
            "noise-level floor on the mean (m>=1pp)": E.present_beta(max(st["mean"], sd_floor), max(st["mean"], sd_floor) ** 2 + v),
            "full sd": beta_from(st["mean"], st["sd"] ** 2)}

def replay(lines, beta):
    """Returns per (inst, record): list of (stamp, k, n, reliable, llr_logged, llr_new, cusum_logged_before)."""
    looks = {}
    for line in lines:
        m = look_re.match(line)
        if not m: continue
        key = (int(m.group(1)), int(m.group(2))); k, n, rel = int(m.group(4)), int(m.group(5)), int(m.group(6))
        f = k / n; llr_new = -E.present_log_density(beta, f)
        looks.setdefault(key, []).append((int(m.group(3)), k, n, rel, float(m.group(9)), llr_new, float(m.group(10))))
    return looks

def page_cusum(seq):
    """CUSUM with weight = judged/reliable (the page statistic's weight approximated by the judged fraction)."""
    out = []; pages = []
    for stamp, k, n, rel, llr_old, llr_new, cb in seq:
        w = min(1.0, n / max(1, rel))
        pages.append(0.0)
        pages = [p + w * llr_new for p in pages]
        out.append((stamp, max(0.0, max(pages))))
    return out

def run(name, log_path, gt):
    lines = open(log_path).read().splitlines()
    st = pooled_stats(lines)
    actual = [(int(m.group(2)), m.group(1)) for m in map(close_re.match, lines) if m and m.group(3) == "1"]
    wins = {}
    for line in lines:
        m = win_re.match(line)
        if m: wins.setdefault(int(m.group(1)), []).append((int(m.group(4)), int(m.group(5)), int(m.group(6)), float(m.group(7)), float(m.group(8)), int(m.group(9))))
    print(f"\n=== {name}: pooled n={st['n']} mean={st['mean']:.5f} MAD-sd={st['mad_sd']:.5f} meanAD={st['meanad']:.5f} sd={st['sd']:.5f}; actual absence closures {len(actual)}: {sorted(set(i for i,_ in actual))}")
    for sname, beta in scales(st).items():
        looks = replay(lines, beta)
        # validation of the llr reconstruction (current scale only): median |llr_new - llr_logged| over scored looks
        if sname.startswith("current"):
            diffs = sorted(abs(x[5] - x[4]) for seq in looks.values() for x in seq)
            print(f"  [check] current-scale llr reconstruction: median abs diff {diffs[len(diffs)//2]:.2f} nats over {len(diffs)} looks (own-history blending ignored)")
        closures = {}
        for (inst, rec), seq in looks.items():
            cs = page_cusum(seq)
            # a closure: first window (through stamp >= look stamp) where the cusum so far > LOG99 (or prior+page) and contradiction > support
            for stamp, cu in cs:
                for (through, sup, con, allr, prior, suff) in wins.get(inst, []):
                    if through == stamp and max(cu, prior + cu if prior else -1) > LOG99 and con > sup:
                        closures.setdefault(inst, (stamp, cu, con, sup)); break
                if inst in closures: break
        pred = sorted(closures)
        gt_right = [i for i in pred if i in gt]; gt_wrong = [i for i in pred if i not in gt]
        act_set = set(i for i, _ in actual)
        print(f"  {sname:20s} beta a={beta.a:.3f} b={beta.b:.1f}: predicted absence closures {len(pred)} -> GT-change objects {len(gt_right)}, GT-static objects {len(gt_wrong)} {gt_wrong[:12]}; same as actual {len(set(pred)&act_set)}, new {sorted(set(pred)-act_set)[:12]}, lost {sorted(act_set-set(pred))[:12]}")

def gt_real(stage):
    gt = set()
    for s in ("b", "c"):
        p = f"{R}/real_row4k/eval_real/changes/{s}/STATE_CHANGE_DETAILS.csv"
        for row in csv.DictReader(open(p)):
            if row["session"].lower() == stage and row["change_type"] != "unchanged":
                gt.add(int(row["object_id"]))
    return gt
def gt_syn(sess):
    gt = set()
    for row in csv.DictReader(open("/home/jixian/Desktop/FT/datasets/synthetic_ab/GT_ANIMATION_INTERVALS.csv")):
        if row["session"] == sess: gt.add(int(row["instance_id"]))
    return gt
print("GT change objects: real B", sorted(gt_real("b")), "real C", sorted(gt_real("c")), "syn A", sorted(gt_syn("a")), "syn B", sorted(gt_syn("b")))
for name, path, gt in (("real A new", f"{R}/real_row4k/session_a/layer_log.txt", set()), ("real B new", f"{R}/real_row4k/session_b/layer_log.txt", gt_real("b")),
                       ("real C new", f"{R}/real_row4k/session_c/layer_log.txt", gt_real("c")), ("syn A new", f"{R}/syn_row4i/session_a/layer_log.txt", gt_syn("a")),
                       ("syn B new", f"{R}/syn_row4i/session_b/layer_log.txt", gt_syn("b"))):
    run(name, path, gt)
