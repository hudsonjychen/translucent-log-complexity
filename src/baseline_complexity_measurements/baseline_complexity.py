from collections import Counter


def variety(traces):
    acts = set()
    for t in traces:
        acts.update(t)
    return len(acts)

def structure(traces):
    return sum(len(set(t)) for t in traces) / len(traces)

def _df_pairs(seq):
    return frozenset(zip(seq[:-1], seq[1:]))

def affinity(traces):
    freq = Counter(traces)
    N = sum(freq.values())
    if N < 2:
        return float("nan")
    variants = list(freq)
    dfset = {v: _df_pairs(v) for v in variants}
    total = 0.0
    for i, v1 in enumerate(variants):
        f1, s1 = freq[v1], dfset[v1]
        total += f1 * (f1 - 1)
        for j in range(i + 1, len(variants)):
            v2 = variants[j]
            union = len(s1 | dfset[v2])
            if union == 0:
                continue
            jac = len(s1 & dfset[v2]) / union
            total += jac * f1 * freq[v2] * 2
    return total / (N * (N - 1))