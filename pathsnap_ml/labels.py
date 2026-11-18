"""Reference labels from .route (TRAIN / DEV ONLY - never a feature).
.route is a per-trip arc sequence, not per-observation, so each GPS fix is assigned a route arc by a monotone
(non-decreasing route position) dynamic-programming alignment minimising point-to-arc distance (+lam metres per skipped arc).
Same alignment as the legacy ML package (decision retained). Uses Sakshi's projected coordinates."""
import numpy as np

def pt_seg(P, A, B):
    AB = B - A; L2 = (AB ** 2).sum(1); L2s = np.where(L2 == 0, 1, L2)
    f = np.clip(((P - A) * AB).sum(1) / L2s, 0, 1)
    return np.linalg.norm(P - (A + f[:, None] * AB), axis=1)

def align_reference(gps_xy, arc_start, arc_end, route, lam=0.5):
    A = arc_start[route]; B = arc_end[route]
    N, M = len(gps_xy), len(route)
    D = np.empty((N, M), dtype=np.float32)
    for j in range(M):
        D[:, j] = pt_seg(gps_xy, np.broadcast_to(A[j], gps_xy.shape), np.broadcast_to(B[j], gps_xy.shape))
    idx = np.arange(M, dtype=np.float32)
    cost = D[0].copy(); back = np.zeros((N, M), dtype=np.int32); ar = np.arange(M)
    for t in range(1, N):
        g = cost - lam * idx; run = np.minimum.accumulate(g)
        am = np.maximum.accumulate(np.where(g <= run, ar, -1))
        adv = np.full(M, np.inf, dtype=np.float32); adv[1:] = run[:-1] + lam * idx[1:]
        use = adv < cost
        back[t] = np.where(use, np.r_[0, am[:-1]], ar)
        cost = np.where(use, adv, cost) + D[t]
    j = int(np.argmin(cost)); path = np.empty(N, dtype=np.int64)
    for t in range(N - 1, -1, -1):
        path[t] = j; j = back[t, j]
    return route[path], D[np.arange(N), path]
