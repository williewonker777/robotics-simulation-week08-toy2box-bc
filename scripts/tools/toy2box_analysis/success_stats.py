"""Success rate mean ± std (plus SE and Wilson 95% CI) from play.py stdout logs ("[INFO] Trial N: True/False")."""

import math
import re
import sys

PAT = re.compile(r"^\[INFO\] Trial (\d+): (True|False)\s*$")


def outcomes(path):
    res = {}
    for line in open(path, errors="replace"):
        m = PAT.match(line)
        if m:
            res[int(m.group(1))] = m.group(2) == "True"
    assert sorted(res) == list(range(len(res))), f"{path}: trial indices are not 0..n-1"
    return [res[i] for i in range(len(res))]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


runs = {}
for path in sys.argv[1:]:
    v = outcomes(path)
    n, k = len(v), sum(v)
    name = path.rsplit("/", 1)[-1]
    runs[name] = v
    if n < 2:
        print(f"{name}: n={n}, too few trials")
        continue
    mean = k / n
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1))  # sample std over per-rollout 0/1 outcomes
    lo, hi = wilson(k, n)
    print(f"{name}: n={n} success={k} -> {mean:.1%} ± {std:.1%} (mean ± sample std over rollouts)")
    print(f"  SE {std / math.sqrt(n):.1%}, Wilson 95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  success trials: {[i for i, x in enumerate(v) if x]}")

# outcome-vector agreement against the first log
names = list(runs)
for other in names[1:]:
    a, b = runs[names[0]], runs[other]
    m = min(len(a), len(b))
    diff = [i for i in range(m) if a[i] != b[i]]
    print(f"{other} vs {names[0]}: {m - len(diff)}/{m} trials identical, differ at {diff}")
