"""Multi-class target encoding yang OOF-safe.

`fit_te(keys, y, m)` → tabel (n_level × K) proba kelas tersmoothing.
`oof_te(...)`       → encoding untuk baris train memakai inner-fold
                      (baris tidak pernah melihat labelnya sendiri).
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def fit_te(codes: np.ndarray, y: np.ndarray, n_levels: int, k: int, m: float, prior: np.ndarray) -> np.ndarray:
    cnt = np.zeros((n_levels, k), dtype=np.float64)
    np.add.at(cnt, (codes, y), 1.0)
    n = cnt.sum(1, keepdims=True)
    return ((cnt + m * prior) / (n + m)).astype(np.float32), n.ravel()


def oof_te(codes: np.ndarray, y: np.ndarray, n_levels: int, k: int, m: float, n_splits: int = 5, seed: int = 0):
    """OOF encoding untuk baris (codes, y). Return (n, k) proba + (n,) count."""
    prior = np.bincount(y, minlength=k) / len(y)
    out = np.zeros((len(y), k), dtype=np.float32)
    cnt = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(codes, y):
        tab, n = fit_te(codes[tr], y[tr], n_levels, k, m, prior)
        out[va] = tab[codes[va]]
        cnt[va] = n[codes[va]] * n_splits / (n_splits - 1)  # samakan skala dgn TE penuh
    return out, cnt


def fit_resid(codes: np.ndarray, R: np.ndarray, n_levels: int, m: float):
    """Rata-rata residual (one-hot(y) − proba model tanpa info entitas) per level, shrink ke 0 dengan m."""
    s = np.stack([np.bincount(codes, weights=R[:, j], minlength=n_levels) for j in range(R.shape[1])], 1)
    n = np.bincount(codes, minlength=n_levels).astype(np.float64)
    return (s / (n + m)[:, None]).astype(np.float32), n


def encode_fold_resid(codes_all: np.ndarray, R_all: np.ndarray, tr_idx: np.ndarray, apply_idx: list[np.ndarray],
                      n_levels: int, m: float, n_splits: int = 5, seed: int = 0):
    """Residual TE (efek acak entitas ala mixed-effects): OOF inner-fold untuk tr_idx, tabel penuh untuk set lain."""
    from sklearn.model_selection import KFold
    enc_tr = np.zeros((len(tr_idx), R_all.shape[1]), dtype=np.float32)
    n_tr = np.zeros(len(tr_idx), dtype=np.float32)
    for a, b in KFold(n_splits, shuffle=True, random_state=seed).split(tr_idx):
        tab, n = fit_resid(codes_all[tr_idx[a]], R_all[tr_idx[a]], n_levels, m)
        enc_tr[b] = tab[codes_all[tr_idx[b]]]
        n_tr[b] = n[codes_all[tr_idx[b]]] * n_splits / (n_splits - 1)
    tab, n = fit_resid(codes_all[tr_idx], R_all[tr_idx], n_levels, m)
    applied = [(tab[codes_all[i]], n[codes_all[i]].astype(np.float32)) for i in apply_idx]
    return (enc_tr, n_tr), applied, (tab, n.astype(np.float32))


def encode_fold(codes_all: np.ndarray, y_all: np.ndarray, tr_idx: np.ndarray, apply_idx: list[np.ndarray],
                n_levels: int, k: int, m: float, seed: int = 0):
    """Untuk satu fold model: OOF-TE di bagian train, TE penuh (dari tr_idx) untuk set lain (valid/test).

    Return: (enc_train, [enc_apply...], table) — table disimpan untuk inference.
    """
    y_tr = y_all[tr_idx]
    enc_tr, n_tr = oof_te(codes_all[tr_idx], y_tr, n_levels, k, m, seed=seed)
    prior = np.bincount(y_tr, minlength=k) / len(y_tr)
    tab, n = fit_te(codes_all[tr_idx], y_tr, n_levels, k, m, prior)
    applied = [(tab[codes_all[i]], n[codes_all[i]].astype(np.float32)) for i in apply_idx]
    return (enc_tr, n_tr), applied, (tab, n.astype(np.float32))
