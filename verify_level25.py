"""Verify the two identities of the middle layer at the level-2.5 rate.

For each edge (i,j) with w=w_ij, w'=w_ji, the edge rate function is

    I_edge(j) = j*ln((j+s)/(2w)) - s + w + w',      s = sqrt(j^2 + 4 w w').

Two checks:
  (A) I_edge(0)    = (sqrt(w) - sqrt(w'))^2      (paper, Sec. 4.2)
  (B) I_edge(j_ss) = 0,  j_ss = w - w'           (typical flow)
and, summing over edges,
      sum_{i<j} I_edge(0) = H(pi).
"""
import sys, math
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

rng = np.random.default_rng(20261121)


def stat(T):
    m = T.shape[0]
    M = np.vstack([T.T - np.eye(m), np.ones(m)])
    b = np.zeros(m + 1); b[-1] = 1
    p, *_ = np.linalg.lstsq(M, b, rcond=None)
    p = np.clip(p, 1e-15, None)
    return p / p.sum()


def rate(j, w, wp):
    s = math.sqrt(j * j + 4.0 * w * wp)
    t = -s + w + wp
    arg = j + s
    if j != 0.0 and arg > 0.0 and w > 0.0:
        t += j * math.log(arg / (2.0 * w))
    return t


def check(T):
    p = stat(T); w = p[:, None] * T; m = T.shape[0]
    H = 0.0; S0 = 0.0; Sss = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            a = float(w[i, j]); b = float(w[j, i])
            if a < 1e-300 and b < 1e-300:
                continue
            H += (math.sqrt(max(a, 0.0)) - math.sqrt(max(b, 0.0))) ** 2
            S0 += rate(0.0, a, b)
            Sss += rate(a - b, a, b)
    return H, S0, Sss


def ring(m, r):
    q = r / (1 + r); T = np.zeros((m, m))
    for i in range(m):
        T[i, (i + 1) % m] = q
        T[i, (i - 1) % m] = 1 - q
    return T


print("=" * 96)
print("Identity check:  H(pi) = sum I_edge(0)   and   I_edge(j_ss) = 0")
print("=" * 96)
print("  %-24s %16s %16s %16s %11s %11s" %
      ("chain", "H(pi)", "sum I_edge(0)", "sum I_edge(jss)", "|A-H|", "|B|"))
cases = [("ring m=%d r=%g" % (m, r), ring(m, r)) for (m, r) in [(3, 1.5), (3, 3.0), (4, 10.0), (5, 2.0)]]
for _ in range(5):
    cases.append(("random m=4", rng.dirichlet([1.0] * 4, size=4)))
worstA = 0.0; worstB = 0.0
for name, T in cases:
    H, S0, Sss = check(T)
    dA = abs(H - S0); dB = abs(Sss)
    worstA = max(worstA, dA); worstB = max(worstB, dB)
    print("  %-24s %16.10f %16.10f %16.10e %11.2e %11.2e" % (name, H, S0, Sss, dA, dB))
print()
print("  max |H - sum I_edge(0)| = %.3e" % worstA)
print("  max |sum I_edge(j_ss)|  = %.3e" % worstB)
