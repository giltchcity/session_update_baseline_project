"""In-visit moves of identified objects from the core's own state decisions (the STATE_CHANGE lines the present
surface writes whenever an identity's state changes). Per identity, in the order the changes were decided: a current
state followed by a current state of the same identity that starts later, inside the visit, with its surface centre
more than 0.10 m (twice the 5 cm resolution, the successor rule's site distance) from the earlier one is one move.
Output only; nothing in the mapper reads it.
  usage: state_moves.py MAPPER_LOG VISIT_START_NS
Prints: identity, start of the earlier state, start of the new state (ns), distance (m)."""
import math, re, sys
PAT = re.compile(r'STATE_CHANGE inst=(\d+) at=(\d+) before_current=(\d) before_start=(\d+) after_current=(\d) '
                 r'after_start=(\d+) before_centre=([-\w.]+),([-\w.]+),([-\w.]+) after_centre=([-\w.]+),([-\w.]+),([-\w.]+)')
def moves(log, visit_start):
    rows = sorted({m.groups() for m in PAT.finditer(open(log, errors='ignore').read())}, key=lambda g: (int(g[1]), int(g[0])))
    last, out = {}, []
    for g in rows:
        i = int(g[0]); bc, bs, ac, as_ = g[2] == '1', int(g[3]), g[4] == '1', int(g[5])
        b = [float(x) for x in g[6:9]]; a = [float(x) for x in g[9:12]]
        if bc and i not in last and not any(map(math.isnan, b)):
            last[i] = (bs, b)
        if ac and not any(map(math.isnan, a)):
            if i in last and as_ > last[i][0] and as_ >= visit_start and math.dist(a, last[i][1]) > 0.10:
                out.append((i, last[i][0], as_, math.dist(a, last[i][1])))
            last[i] = (as_, a)
    return out
if __name__ == '__main__':
    for i, s0, s1, d in moves(sys.argv[1], int(sys.argv[2])):
        print(i, s0, s1, f'{d:.3f}')
