"""Eksperimen cepat 1 fold CatBoost GPU (fold 0, fold & TE sama dengan training penuh).

Usage: python src/exp_cat.py <tag> [--te SET,...] [--ids] [--lr 0.08] [--depth 8] [--l2 5] [--border 128]
  --te   : kumpulan kunci TE tambahan (lihat EXTRA_TE), dipisah koma
  --ids  : customer_id & courier_id mentah sebagai kategori native CatBoost (ordered target statistics)
Hasil: satu baris JSON di cache/exp_log.jsonl (logloss & Macro F1 fold 0) + proba val di cache/expc_<tag>_p.npy
"""
import json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import features as F
from common import (CAT_PARAMS, K, TE_KEYS, add_te, macro_f1, make_folds, optimize_class_weights, prepare_matrix,
                    te_codes)

EXTRA_TE = {  # nama set: {nama kunci: (kolom, smoothing m)}
    "month": {"hub_month": (["hub_id", "ship_month"], 30.0), "cour_month": (["courier_id", "ship_month"], 10.0),
              "dest_month": (["kota_tujuan", "ship_month"], 30.0)},
    "goods": {"cour_goods": (["courier_id", "goods_category"], 10.0),
              "cust_goods": (["customer_id", "goods_category"], 2.0),
              "hub_dest": (["hub_id", "dest_region"], 30.0)},
    "route": {"cour_route": (["courier_id", "route_pair"], 10.0), "hub_route": (["hub_id", "kota_tujuan"], 30.0),
              "cust_hub": (["customer_id", "hub_id"], 2.0)},
}

arg = lambda name, default: type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
tag = sys.argv[1]
te_sets = arg("--te", "").split(",") if "--te" in sys.argv else []
keys = dict(TE_KEYS)
for s in te_sets:
    keys.update(EXTRA_TE[s])

df = F.build(os.environ.get("DATA_DIR", "data"), "cache")
X, y_all, tr_all, te_all = prepare_matrix(df)
if "--ids" in sys.argv:
    for c in ["customer_id", "courier_id"]:
        X[c] = df[c].to_pandas().astype("category")
codes = te_codes(df, keys)
rcodes = (te_codes(df, {"customer_id": (["customer_id"], 1.0), "cust_hub": (["customer_id", "hub_id"], 1.0)})
          if "--resid" in sys.argv else None)
del df
folds = make_folds(y_all[tr_all])
tr_idx, va_idx = tr_all[folds != 0], tr_all[folds == 0]
Xtr, Xva = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
del X
add_te([Xtr, Xva], codes, y_all, tr_idx, [va_idx], seed=0)
ytr, yva = y_all[tr_idx], y_all[va_idx]
if "--resid" in sys.argv:  # residual TE customer (efek acak, mixed-effects) dari artifacts/stageA_oof.npy
    from te import encode_fold_resid
    R_all = np.zeros((len(y_all), K), dtype=np.float32)
    R_all[tr_all] = np.eye(K, dtype=np.float32)[y_all[tr_all]] - np.load("artifacts/stageA_oof.npy")
    rkeys = {"customer_id": 1.0, "cust_hub": 1.0} if "--resid2" in sys.argv else {"customer_id": 1.0}
    for name, m_ in rkeys.items():
        c, nl, _ = rcodes[name]
        (etr, ntr), [(eva, nva)], _ = encode_fold_resid(c, R_all, tr_idx, [va_idx], nl, m_ * arg("--rm", 1.0), seed=0)
        for j in range(K):
            Xtr[f"rte_{name}_{j}"], Xva[f"rte_{name}_{j}"] = etr[:, j], eva[:, j]
        Xtr[f"rte_{name}_n"], Xva[f"rte_{name}_n"] = ntr, nva

from catboost import CatBoostClassifier, Pool

params = dict(CAT_PARAMS, task_type="GPU", devices="0", learning_rate=arg("--lr", 0.08), depth=arg("--depth", 8),
              l2_leaf_reg=arg("--l2", 5.0), border_count=arg("--border", 128), iterations=arg("--iters", 8000),
              verbose=1000)
params.pop("thread_count", None)
cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
conv = lambda d: d.assign(**{c: d[c].astype(str) for c in cats})
t = time.time()
m = CatBoostClassifier(**params, random_seed=0)
m.fit(Pool(conv(Xtr), ytr, cat_features=cats), eval_set=Pool(conv(Xva), yva, cat_features=cats), use_best_model=True)
p = m.predict_proba(Pool(conv(Xva), cat_features=cats))
from sklearn.metrics import log_loss

# bobot kelas: dicari di separuh val, dievaluasi di separuh lain (dua arah) → tidak optimistis
h = np.arange(len(yva)) % 2 == 0
f1w = np.mean([macro_f1(yva[~a], p[~a], optimize_class_weights(p[a], yva[a])) for a in (h, ~h)])
res = dict(tag=tag, te=te_sets, ids="--ids" in sys.argv, lr=params["learning_rate"], depth=params["depth"],
           l2=params["l2_leaf_reg"], n_feat=Xtr.shape[1], iters=m.get_best_iteration(),
           logloss=round(float(log_loss(yva, p)), 5), f1_argmax=round(macro_f1(yva, p), 5), f1_w=round(float(f1w), 5),
           mins=round((time.time() - t) / 60, 1))
print(json.dumps(res), flush=True)
os.makedirs("cache", exist_ok=True)
open("cache/exp_log.jsonl", "a").write(json.dumps(res) + "\n")
np.save(f"cache/expc_{tag}_p.npy", p)
