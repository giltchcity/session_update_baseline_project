"""Offline replay of the absence test (Eq. 2, projected_physical_evidence.cpp addLook + the commit rule) on the
recorded evidence of a mapping run: every scored round is one ABSENCE_LOOK line (k, n, reliable, llr, the cumulative
first-judged counts per candidate change time) and, from the second visit on, an ABSENCE_COMMIT line of the same
thread (the cross-visit prior log-odds). The replay is open loop: a decision does not feed back into later evidence,
so it is exact for each state up to its first commitment under the replayed variant.

Variants (all else as the method):
  eta=X            commit threshold log(X) instead of log(99)
  weights=one      w_i(j) = 1 (plain Page CUSUM over rounds)
  single           no accumulation: a round commits alone (llr + prior on the first window)
  beta=fixed       the present model fixed at the cold start Beta(0.1, 1.9) instead of the learned one
  prior=0          no cross-visit prior (q = 1/2)

usage: replay.py LOG [variant ...]   -> JSON lines, one per scored round, with the replayed and logged decisions
       replay.py --check LOG         -> exactness check of the method itself against the log"""
import json, math, re, sys
from collections import defaultdict

LOOK = re.compile(r'^\S+ \S+ (\d+) .*ABSENCE_LOOK inst=(\d+) slot=(-?\d+) record=(\d+) stamp=(\d+) k=(\d+) n=(\d+) '
                  r'reliable=(\d+) own=(\d+) own_latest=(\d+) identified=(\d) inherited=(\d) llr=(\S+) cusum_before=(\S+) '
                  r'first_since=([\d,]+)')
WINDOW = re.compile(r'^\S+ \S+ (\d+) .*STATE_EVIDENCE_WINDOW inst=(\d+) .*through=(\d+) .* support=(\d+) contradiction=(\d+) '
                    r'.*absence_llr=(\S+) ')
COMMIT = re.compile(r'^\S+ \S+ (\d+) .*ABSENCE_COMMIT inst=(\d+) slot=(-?\d+) prior_log_odds=(\S+) before_session=(\S+) '
                    r'page_statistic=(\S+) committed=(\d)')


def rounds(path):
    """Scored rounds in log order, each with its prior (0 when the run logs none, i.e. the first visit)."""
    pending, window, out = {}, {}, []
    for line in open(path, errors='replace'):
        m = LOOK.match(line)
        if m:
            th, inst, slot, rec, stamp, k, n, rel, own, own_l, ident, inh, llr, cb, since = m.groups()
            r = dict(thread=th, inst=int(inst), slot=int(slot), record=int(rec), stamp=int(stamp), k=int(k), n=int(n),
                     reliable=int(rel), own=int(own), identified=int(ident), inherited=int(inh), llr=float(llr),
                     cusum_before=float(cb), since=[int(x) for x in since.split(',')], prior=0.0, logged_commit=None)
            out.append(r); pending[(th, r['inst'], r['slot'])] = r; window[(th, r['inst'])] = r
            continue
        m = WINDOW.match(line)
        if m:   # the round's ray evidence, logged right after its look: support S, contradiction C (ungated)
            th, inst, through, sup, con, llr = m.groups()
            r = window.pop((th, int(inst)), None)
            if r is not None and int(through) == r['stamp']:
                r['support'], r['contradiction'], r['window_cusum'] = int(sup), int(con), float(llr)
            continue
        m = COMMIT.match(line)
        if m:
            th, inst, slot, prior, before, page, committed = m.groups()
            r = pending.pop((th, int(inst), int(slot)), None)
            if r is not None:
                r['prior'] = float(prior); r['logged_commit'] = int(committed); r['logged_page'] = float(page)
    return out


def beta_llr(k, n, a=0.1, b=1.9):
    f = min(0.995, max(max(0.005, a / (a + b)), k / n))
    return -(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + (a - 1) * math.log(f) + (b - 1) * math.log(1 - f))


def replay(rs, eta=99.0, weights='method', single=False, beta='learned', prior=True, off=False):
    # The logged first-judged counts are per candidate of the method's own window (it restarts at its commitments).
    # Candidate j of the variant's window is round g = start + j of the record; the method's window starts at round
    # g0 = n_round - (len(since) - 1); for g >= g0 the count is since[g - g0] exactly (it depends only on the looks from
    # g on). Candidates older than the method's window have no logged count: the variant restarts its candidates there
    # (not a commitment; it can only delay the variant's decisions).
    state = defaultdict(lambda: dict(page=[], cusum=0.0, first=True, start=0, n=0))
    for r in rs:
        s = state[r['record']]
        r['replay_cusum_before'] = s['cusum']
        llr = beta_llr(r['k'], r['n']) if beta == 'fixed' else r['llr']
        g0 = s['n'] - (len(r['since']) - 1)
        if s['start'] < g0:
            drop = g0 - s['start']
            if drop >= len(s['page']): s['page'] = []
            else: s['page'] = s['page'][drop:]
            s['start'] = g0; s['first'] = False
            s['cusum'] = max(0.0, max(s['page'])) if s['page'] else 0.0
        t = len(s['page'])
        w = [1.0 if weights == 'one' else min(1.0, r['since'][s['start'] - g0 + j] / max(1, r['reliable']))
             for j in range(t + 1)]
        s['n'] += 1
        if single:
            s['page'] = [w[t] * llr]
        else:
            s['page'].append(0.0)
            for j in range(t + 1): s['page'][j] += w[j] * llr
        s['cusum'] = max(0.0, max(s['page']))
        before = (r['prior'] if prior else 0.0) + s['page'][0] if s['first'] and s['page'] else -math.inf
        r['commit'] = max(before, s['cusum']) > math.log(eta) and not off
        r['close'] = r['commit'] and r.get('contradiction', 0) > r.get('support', 0)   # the absence path of a close
        if r['commit']:
            s['page'], s['cusum'], s['first'], s['start'] = [], 0.0, False, s['n']
        elif single:
            s['start'] = s['n'] - 1
    return rs


if __name__ == '__main__':
    if sys.argv[1] == '--check':
        rs = replay(rounds(sys.argv[2]))
        bad_c = sum(abs(r['replay_cusum_before'] - r['cusum_before']) > 1e-3 * max(1, abs(r['cusum_before'])) for r in rs)
        paired = [r for r in rs if r['logged_commit'] is not None]
        bad_d = sum(int(r['commit']) != r['logged_commit'] for r in paired)
        print(json.dumps(dict(log=sys.argv[2], rounds=len(rs), cusum_mismatch=bad_c, paired_decisions=len(paired),
                              decision_mismatch=bad_d, replayed_commits=sum(r['commit'] for r in rs))))
    else:
        opts = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
        rs = replay(rounds(sys.argv[1]), eta=float(opts.get('eta', 99)), weights=opts.get('weights', 'method'),
                    single='single' in sys.argv[2:], beta=opts.get('beta', 'learned'), prior=opts.get('prior') != '0')
        for r in rs: print(json.dumps({k: v for k, v in r.items() if k != 'since'}))
