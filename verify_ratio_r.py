"""Check the single-cycle degeneration law  H/Deltas = r(a) = tanh(a/4)/a.

Model: uniform ring of m states, clockwise rate q = r/(1+r), counter-clockwise
1-q, so that every directed edge has affinity a = ln r.  The stationary
distribution is uniform and all edges carry the same flow pair
    w = W exp(a/2),  w' = W exp(-a/2).

Checks:
  (1) per-edge identity:  H_edge/Deltas_edge = tanh(a/4)/a  exactly;
  (2) chain-level ratio H/Deltas equals the same value for every m
      (all edges have equal affinity, so the ratio is m-independent);
  (3) JS/Deltas <= H/Deltas with equality only at a = 0;
  (4) small-a behaviour: H/Deltas -> 1/4 + O(a^2).

Note: Deltas, H and JS depend only on the transition matrix T and on pi.
The size of the emission alphabet does not enter, so no claim of
"independence of K" is tested here (it would be vacuous).
"""
import sys, math
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np


def stat(T):
    m = T.shape[0]
    M = np.vstack([T.T - np.eye(m), np.ones(m)])
    b = np.zeros(m + 1); b[-1] = 1
    p, *_ = np.linalg.lstsq(M, b, rcond=None)
    p = np.clip(p, 1e-15, None)
    return p / p.sum()


def kl(x, y):
    x = np.clip(x, 1e-300, None); y = np.clip(y, 1e-300, None)
    return float(np.sum(x * np.log(x / y)))


def quant(T):
    m = T.shape[0]; p = stat(T); w = p[:, None] * T
    DS = 0.0
    for i in range(m):
        for j in range(m):
            a = float(w[i, j]); b = float(w[j, i])
            if a > 1e-300 and b > 1e-300:
                DS += a * math.log(a / b)
    H = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            H += (math.sqrt(max(w[i, j], 0.0)) - math.sqrt(max(w[j, i], 0.0))) ** 2
    JS = 0.0
    for i in range(m):
        a = T[i, :]; bb = (p * T[:, i]) / max(p[i], 1e-300); mm = (a + bb) / 2
        JS += 0.5 * p[i] * (kl(a, mm) + kl(bb, mm))
    return DS, H, JS


def ring(m, r):
    q = r / (1 + r); T = np.zeros((m, m))
    for i in range(m):
        T[i, (i + 1) % m] = q
        T[i, (i - 1) % m] = 1 - q
    return T


def r_exact(a):
    if abs(a) < 1e-12:
        return 0.25
    return math.tanh(a / 4.0) / a


print("=" * 100)
print("Single-cycle degeneration:  H/DeltaS = r(a) = tanh(a/4)/a,  a = ln r")
print("=" * 100)
print("  %4s %7s %12s %12s %12s %12s %11s %9s" %
      ("m", "r", "a", "DeltaS", "H/DeltaS", "r(a)", "rel.dev", "JS<=H"))
for m in (3, 4, 5, 6):
    for r in (1.5, 3.0, 10.0):
        DS, H, JS = quant(ring(m, r))
        a = math.log(r)
        got = H / DS if DS > 1e-14 else float('nan')
        th = r_exact(a)
        print("  %4d %7.2f %12.6f %12.8f %12.8f %12.8f %11.2e %9s" %
              (m, r, a, DS, got, th, abs(got - th) / th, "OK" if JS <= H + 1e-12 else "FAIL"))
print()
print("  --- small-a limit (expect 1/4 + O(a^2)) ---")
for r in (1.0001, 1.001, 1.01, 1.1):
    DS, H, JS = quant(ring(5, r))
    a = math.log(r)
    print("    r-1 = %.4e   a = %.3e   H/DeltaS = %.10f   r(a) = %.10f" %
          (r - 1, a, H / DS, r_exact(a)))
print()
print("  a = 0.5 (r = e^0.5 = 1.648721):")
DS, H, JS = quant(ring(5, math.exp(0.5)))
print("    H/DeltaS = %.6f    r(0.5) = %.6f    JS/DeltaS = %.6f" % (H / DS, r_exact(0.5), JS / DS))
