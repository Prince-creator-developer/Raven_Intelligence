"""
Curvature-aware lip feature extraction.

Input  : (T, 44, 2|3) lip-contour landmarks (MediaPipe FaceMesh, pixel units), or legacy (T, 12).
Output : (T, 64) = 32 static features + 32 first-derivatives.

Landmark order (44 pts, always the same):
    0..10  outer upper  (61 -> 291)
    11..21 outer lower  (61 -> 291)
    22..32 inner upper  (78 -> 308)
    33..43 inner lower  (78 -> 308)
"""
import numpy as np

OUTER_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
OUTER_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_LOWER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LIP_LANDMARKS = OUTER_UPPER + OUTER_LOWER + INNER_UPPER + INNER_LOWER  # 44 indices

N_PTS = 44
LEGACY_DIM = 12
STATIC_DIM = 32
FEATURE_DIM = 64
EPS = 1e-6


def detect_format(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == LEGACY_DIM:
        return "legacy12"
    if arr.ndim == 3 and arr.shape[1] == N_PTS and arr.shape[2] in (2, 3):
        return "landmarks44"
    return None


def feature_dim(fmt):
    return LEGACY_DIM if fmt == "legacy12" else FEATURE_DIM


def resample_time(arr, T):
    arr = np.asarray(arr, dtype=np.float32)
    n = arr.shape[0]
    if n == T:
        return arr
    if n == 1:
        return np.repeat(arr, T, axis=0)
    src = np.linspace(0, n - 1, T)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo).astype(np.float32).reshape((T,) + (1,) * (arr.ndim - 1))
    return arr[lo] * (1 - w) + arr[hi] * w


def _smooth_time(x, k=3):
    if x.shape[0] < k:
        return x
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)], 0)
    return sum(xp[i:i + x.shape[0]] for i in range(k)) / k


def normalize_landmarks(pts):
    """Center on mouth corners, de-roll, scale by the sequence's median mouth width."""
    pts = np.asarray(pts, np.float32)
    if pts.shape[-1] == 2:
        pts = np.concatenate([pts, np.zeros(pts.shape[:-1] + (1,), np.float32)], -1)
    pts = _smooth_time(pts)
    L, R = pts[:, 0], pts[:, 10]
    center = (L + R) / 2
    p = pts - center[:, None, :]
    d = (R - L)[:, :2]
    ang = np.arctan2(d[:, 1], d[:, 0])
    c, s = np.cos(-ang)[:, None], np.sin(-ang)[:, None]
    x = p[..., 0] * c - p[..., 1] * s
    y = p[..., 0] * s + p[..., 1] * c
    ref = max(float(np.median(np.linalg.norm(d, axis=1))), EPS)
    return np.stack([x, y, p[..., 2]], -1) / ref


def _parabola_curvature(curve):
    """Least-squares y = a x^2 + b x + c per frame -> curvature 2a/(1+b^2)^1.5 (robust global lip curve)."""
    x, y = curve[..., 0], curve[..., 1]
    A = np.stack([x * x, x, np.ones_like(x)], -1)
    AtA = np.einsum("tni,tnj->tij", A, A) + 1e-6 * np.eye(3, dtype=np.float32)
    Aty = np.einsum("tni,tn->ti", A, y)
    coef = np.linalg.solve(AtA, Aty[..., None])[..., 0]
    a, b = coef[:, 0], coef[:, 1]
    return np.clip(2 * a / np.power(1 + b * b, 1.5), -10, 10)


def _menger(curve, stride=2):
    """Signed local curvature of point triplets (i-s, i, i+s) -> (T, N-2s)."""
    a, b, d = curve[:, :-2 * stride], curve[:, stride:-stride], curve[:, 2 * stride:]
    ab, bd, ad = b - a, d - b, d - a
    cross = ab[..., 0] * bd[..., 1] - ab[..., 1] * bd[..., 0]
    den = np.linalg.norm(ab, axis=-1) * np.linalg.norm(bd, axis=-1) * np.linalg.norm(ad, axis=-1) + 1e-4
    return np.clip(2 * cross / den, -10, 10)


def _area(poly):
    x, y = poly[..., 0], poly[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1))


def landmarks_to_features(pts):
    p = normalize_landmarks(pts)
    xy = p[..., :2]
    OU, OL, IU, IL = xy[:, 0:11], xy[:, 11:22], xy[:, 22:33], xy[:, 33:44]
    dist = lambda a, b: np.linalg.norm(a - b, axis=-1)

    inner_c = dist(IU[:, 5], IL[:, 5])
    inner_l = (dist(IU[:, 2], IL[:, 2]) + dist(IU[:, 3], IL[:, 3])) / 2
    inner_r = (dist(IU[:, 7], IL[:, 7]) + dist(IU[:, 8], IL[:, 8])) / 2
    outer_c = dist(OU[:, 5], OL[:, 5])
    width = dist(xy[:, 0], xy[:, 10])
    aspect = inner_c / (width + EPS)
    inner_area = _area(np.concatenate([IU, IL[:, 9:0:-1]], 1))
    outer_area = _area(np.concatenate([OU, OL[:, 9:0:-1]], 1))
    protrusion = (p[:, 5, 2] + p[:, 16, 2]) / 2 - (p[:, 0, 2] + p[:, 10, 2]) / 2
    upper_thick = dist(OU[:, 5], IU[:, 5])
    lower_thick = dist(OL[:, 5], IL[:, 5])
    center_y = (OU[:, 5, 1] + OL[:, 5, 1] + IU[:, 5, 1] + IL[:, 5, 1]) / 4

    feats = [inner_c, inner_l, inner_r, outer_c, width, aspect, inner_area, outer_area,
             protrusion, upper_thick, lower_thick, center_y]               # 12
    sag = []
    for c in (OU, OL, IU, IL):                                              # 4 x 4 = 16 curvature
        m = _menger(c)
        feats += [_parabola_curvature(c), m[:, 3], m[:, :3].mean(1), m[:, 4:].mean(1)]
        sag.append(c[:, 5, 1] - (c[:, 0, 1] + c[:, 10, 1]) / 2)             # 4 sagitta
    feats += sag
    static = np.stack(feats, 1).astype(np.float32)                          # (T, 32)
    delta = np.gradient(static, axis=0) if static.shape[0] > 1 else np.zeros_like(static)
    out = np.concatenate([static, delta], 1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def prepare_features(arr, fmt, seq_len=30):
    """Raw recorded array -> (seq_len, D) model input."""
    arr = resample_time(arr, seq_len)
    if fmt == "legacy12":
        return np.nan_to_num(arr).astype(np.float32)
    if fmt == "landmarks44":
        return landmarks_to_features(arr)
    raise ValueError(f"unknown format {fmt!r}")
