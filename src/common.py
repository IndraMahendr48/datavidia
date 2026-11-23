"""Utilitas bersama: matriks fitur, folds, metrik, optimasi bobot kelas."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from features import CAT_COLS, CLASSES, ID_COLS, TARGET
from te import encode_fold

K = len(CLASSES)

# kombinasi kolom untuk target encoding (ID kardinalitas tinggi + interaksi)
TE_KEYS = {  # nama: (kolom, smoothing m)
    "customer_id": (["customer_id"], 2.0),
    "courier_id": (["courier_id"], 10.0),
    "hub_id": (["hub_id"], 30.0),
    "route_pair": (["route_pair"], 30.0),
    "hub_pkg": (["hub_id", "package_id"], 30.0),
    "hub_goods": (["hub_id", "goods_category"], 30.0),
    "kota_tujuan": (["kota_tujuan"], 30.0),
}


def te_codes(df: pl.DataFrame, keys: dict = TE_KEYS) -> dict[str, tuple[np.ndarray, int]]:
    out = {}
    for name, (cols, m) in keys.items():
        c = df.select(pl.struct(cols).rank("dense").cast(pl.Int64) - 1).to_series().to_numpy()
        out[name] = (c, int(c.max()) + 1, m)
    return out


def to_pandas_X(df: pl.DataFrame, drop: list[str] | None = None) -> pd.DataFrame:
    drop = set((drop or []) + ID_COLS + [TARGET, "is_test"])
    cols = [c for c in df.columns if c not in drop]
    pdf = df.select(cols).to_pandas()
    for c in cols:
        if c in CAT_COLS:
            pdf[c] = pdf[c].astype("category")
        elif pdf[c].dtype == bool:
            pdf[c] = pdf[c].astype(np.int8)
        else:
            pdf[c] = pdf[c].astype(np.float32)
    return pdf


def make_folds(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    f = np.zeros(len(y), dtype=np.int8)
    for i, (_, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(y, y)):
        f[va] = i
    return f


def macro_f1(y, proba, w=None) -> float:
    p = proba if w is None else proba * w
    return f1_score(y, p.argmax(1), average="macro")


def optimize_class_weights(proba, y, n_iter: int = 3, grid=None) -> np.ndarray:
    """Coordinate ascent pada bobot multiplikatif per kelas (log-grid)."""
    grid = np.exp(np.linspace(-1.2, 1.2, 41)) if grid is None else grid
    w = np.ones(proba.shape[1])
    best = macro_f1(y, proba, w)
    for _ in range(n_iter):
        for k in range(1, proba.shape[1]):  # kelas 0 jadi referensi
            cand = w.copy()
            scores = []
            for g in grid:
                cand[k] = w[k] * g
                scores.append(macro_f1(y, proba, cand))
            j = int(np.argmax(scores))
            if scores[j] > best:
                best, w[k] = scores[j], w[k] * grid[j]
        grid = np.exp(np.linspace(-0.3, 0.3, 21))
    return w / w.sum() * len(w)


def add_te(X_parts: list[pd.DataFrame], codes: dict, y: np.ndarray, tr_idx, other_idx: list,
           seed: int = 0, keep_tables: bool = False):
    """Tambahkan kolom TE ke X_train (OOF) dan ke X lain (TE penuh dari tr_idx)."""
    tables = {}
    Xtr, others = X_parts[0], X_parts[1:]
    for name, (c, nl, m) in codes.items():
        (etr, ntr), applied, tab = encode_fold(c, y, tr_idx, other_idx, nl, K, m, seed=seed)
        for j, cls in enumerate(CLASSES):
            Xtr[f"te_{name}_{cls}"] = etr[:, j]
        Xtr[f"te_{name}_n"] = ntr
        for Xo, (eo, no) in zip(others, applied):
            for j, cls in enumerate(CLASSES):
                Xo[f"te_{name}_{cls}"] = eo[:, j]
            Xo[f"te_{name}_n"] = no
        if keep_tables:
            tables[name] = tab
    return tables


# ------------------------------------------------------------------ matriks & model
HC_CATS = ["hub_id", "kota_asal", "kota_tujuan", "package_id", "promo_id"]  # kategori → kode integer


def prepare_matrix(df: pl.DataFrame):
    """X (pandas) untuk seluruh baris train+test, label (−1 untuk test), indeks train & test."""
    X = to_pandas_X(df)
    for c in HC_CATS:
        X[c] = X[c].cat.codes.astype(np.float32)
    is_test = df["is_test"].to_numpy()
    y_all = np.full(df.height, -1)
    y_all[~is_test] = (df.filter(~pl.col("is_test"))[TARGET]
                       .replace_strict({c: i for i, c in enumerate(CLASSES)}).to_numpy())
    return X, y_all, np.where(~is_test)[0], np.where(is_test)[0]


def save_te_tables(df: pl.DataFrame, codes: dict, tables: dict, path_prefix: str):
    """Simpan tabel TE (nilai kunci → 7 proba + n) supaya inference tidak perlu label train."""
    for name, (cols, _) in TE_KEYS.items():
        tab, n = tables[name]
        keys = (df.select(cols).with_columns(pl.Series("_code", codes[name][0]))
                .unique("_code").sort("_code"))
        out = keys.with_columns(
            *[pl.Series(f"te_{name}_{c}", tab[keys["_code"].to_numpy(), j]) for j, c in enumerate(CLASSES)],
            pl.Series(f"te_{name}_n", n[keys["_code"].to_numpy()]),
        ).drop("_code")
        out.write_parquet(path_prefix.replace(".parquet", f"_{name}.parquet"))


def apply_te_tables(df: pl.DataFrame, X: pd.DataFrame, path_prefix: str) -> pd.DataFrame:
    """Inference: gabungkan tabel TE tersimpan ke baris df (urutan dipertahankan)."""
    X = X.copy()
    for name, (cols, _) in TE_KEYS.items():
        tab = pl.read_parquet(path_prefix.replace(".parquet", f"_{name}.parquet"))
        j = df.select(cols).join(tab, on=cols, how="left", maintain_order="left")
        assert j.height == df.height
        for c in tab.columns:
            if c not in cols:
                X[c] = j[c].to_numpy().astype(np.float32)
    return X


LGB_PARAMS = dict(objective="multiclass", num_class=K, learning_rate=0.03, num_leaves=63, min_data_in_leaf=400,
                  feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255,
                  cat_smooth=20, max_cat_to_onehot=8, verbose=-1, num_threads=10)
XGB_PARAMS = dict(objective="multi:softprob", num_class=K, eta=0.05, max_depth=8, min_child_weight=20,
                  subsample=0.8, colsample_bytree=0.4, reg_lambda=10.0, tree_method="hist", max_bin=255,
                  eval_metric="mlogloss", nthread=10, max_cat_to_onehot=8)
CAT_PARAMS = dict(loss_function="MultiClass", learning_rate=0.08, depth=8, l2_leaf_reg=5, iterations=4000,
                  od_type="Iter", od_wait=150, border_count=128, verbose=500)
if os.environ.get("USE_GPU") == "1":  # CatBoost & XGBoost di GPU (NVIDIA + CUDA)
    CAT_PARAMS.update(task_type="GPU", devices="0")
    XGB_PARAMS.update(device="cuda")
else:
    CAT_PARAMS.update(thread_count=os.cpu_count())
if os.environ.get("QUICK") == "1":  # smoke test: cepat, hanya untuk cek instalasi
    CAT_PARAMS.update(iterations=200)


def fit_predict(model: str, Xtr, ytr, Xva, yva, Xte, seed: int, path: str):
    if model == "lgb":
        import lightgbm as lgb
        dtr = lgb.Dataset(Xtr, ytr)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        m = lgb.train({**LGB_PARAMS, "seed": seed}, dtr, 20000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)])
        m.save_model(path + ".txt", num_iteration=m.best_iteration)
        it = m.best_iteration
        return m, m.predict(Xva, num_iteration=it), m.predict(Xte, num_iteration=it), it
    if model == "xgb":
        import xgboost as xgb
        dtr = xgb.DMatrix(Xtr, ytr, enable_categorical=True)
        dva = xgb.DMatrix(Xva, yva, enable_categorical=True)
        m = xgb.train({**XGB_PARAMS, "seed": seed}, dtr, 20000, evals=[(dva, "va")], early_stopping_rounds=200,
                      verbose_eval=500)
        best = m.best_iteration + 1
        m = m[:best]  # simpan hanya pohon sampai best iteration
        m.save_model(path + ".json")
        return m, m.predict(dva), m.predict(xgb.DMatrix(Xte, enable_categorical=True)), best
    if model == "cat":
        from catboost import CatBoostClassifier, Pool
        cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        conv = lambda d: d.assign(**{c: d[c].astype(str) for c in cats})
        ptr, pva = Pool(conv(Xtr), ytr, cat_features=cats), Pool(conv(Xva), yva, cat_features=cats)
        m = CatBoostClassifier(**CAT_PARAMS, random_seed=seed)
        m.fit(ptr, eval_set=pva, use_best_model=True)
        m.save_model(path + ".cbm")
        return m, m.predict_proba(pva), m.predict_proba(Pool(conv(Xte), cat_features=cats)), m.get_best_iteration()
    raise ValueError(model)


def load_predict(model: str, path: str, X: pd.DataFrame) -> np.ndarray:
    """Inference tanpa retrain: muat file model lalu prediksi proba."""
    if model == "lgb":
        import lightgbm as lgb
        return lgb.Booster(model_file=path + ".txt").predict(X)
    if model == "xgb":
        import xgboost as xgb
        b = xgb.Booster(); b.load_model(path + ".json")
        return b.predict(xgb.DMatrix(X, enable_categorical=True))
    if model == "cat":
        from catboost import CatBoostClassifier, Pool
        m = CatBoostClassifier(); m.load_model(path + ".cbm")
        cats = [c for c in X.columns if str(X[c].dtype) == "category"]
        return m.predict_proba(Pool(X.assign(**{c: X[c].astype(str) for c in cats}), cat_features=cats))
    raise ValueError(model)


def run_cv(model: str, seed: int, df_keys: pl.DataFrame, X: pd.DataFrame, y_all: np.ndarray, tr_all: np.ndarray,
           te_all: np.ndarray, codes: dict, folds: np.ndarray, art: str = "artifacts", quick: bool = False):
    """Training 5-fold untuk satu model. Simpan model per fold, tabel TE, OOF & proba test."""
    import gc, json, time
    os.makedirs(f"{art}/models", exist_ok=True)
    os.makedirs(f"{art}/te", exist_ok=True)
    oof = np.zeros((len(tr_all), K), dtype=np.float32)
    test_p = np.zeros((len(te_all), K), dtype=np.float32)
    log = []
    for k in range(1 if quick else int(folds.max()) + 1):
        t0 = time.time()
        tr_idx, va_idx = tr_all[folds != k], tr_all[folds == k]
        if quick:  # smoke test: 100rb baris saja
            tr_idx = np.random.default_rng(0).choice(tr_idx, 100_000, replace=False)
        Xtr, Xva, Xte = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy(), X.iloc[te_all].copy()
        tables = add_te([Xtr, Xva, Xte], codes, y_all, tr_idx, [va_idx, te_all], seed=k, keep_tables=True)
        if not quick and not os.path.exists(f"{art}/te/te_f{k}_customer_id.parquet"):
            save_te_tables(df_keys, codes, tables, f"{art}/te/te_f{k}.parquet")
        if k == 0 and not quick:
            json.dump(list(Xtr.columns), open(f"{art}/features.json", "w"))
        ytr, yva = y_all[tr_idx], y_all[va_idx]
        m, p_va, p_te, it = fit_predict(model, Xtr, ytr, Xva, yva, Xte, seed + k, f"{art}/models/{model}_s{seed}_f{k}")
        oof[folds == k], test_p = p_va, test_p + p_te / (int(folds.max()) + 1)
        r = dict(fold=k, iters=int(it), f1=macro_f1(yva, p_va), mins=round((time.time() - t0) / 60, 1))
        print(json.dumps(r), flush=True)
        log.append(r)
        del Xtr, Xva, Xte, m
        gc.collect()
    if quick:
        print("QUICK smoke test selesai — instalasi OK. Jalankan ulang tanpa QUICK=1 untuk training penuh.")
        return None
    y_tr = y_all[tr_all]
    summary = dict(model=model, seed=seed, f1_argmax=macro_f1(y_tr, oof), logloss=float(log_loss(y_tr, oof)),
                   fold_f1=[r["f1"] for r in log], fold_iters=[r["iters"] for r in log])
    print(json.dumps(summary), flush=True)
    np.save(f"{art}/oof_{model}_s{seed}.npy", oof)
    np.save(f"{art}/test_{model}_s{seed}.npy", test_p)
    json.dump(summary, open(f"{art}/summary_{model}_s{seed}.json", "w"), indent=1)
    return summary


def blend(ps: list, w) -> np.ndarray:
    """Rata-rata geometrik berbobot dari beberapa matriks proba."""
    lp = sum(wi * np.log(np.clip(p, 1e-7, 1)) for wi, p in zip(w, ps))
    e = np.exp(lp - lp.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_blend_weights(ps: list, y: np.ndarray) -> np.ndarray:
    """Bobot blend (softmax) yang meminimalkan logloss OOF."""
    from scipy.optimize import minimize
    if len(ps) == 1:
        return np.ones(1)
    obj = lambda a: log_loss(y, blend(ps, np.exp(a) / np.exp(a).sum()))
    a = minimize(obj, np.zeros(len(ps)), method="Nelder-Mead", options=dict(maxiter=200)).x
    return np.exp(a) / np.exp(a).sum()
