"""Eksperimen cepat 1 fold LightGBM. Usage: python exp_lgb.py <tag> [--no-te] [--drop prefix,...]"""
import sys, time, json
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "src")
import features as F
from common import *

tag = sys.argv[1]
use_te = "--no-te" not in sys.argv
drop_pref = []
if "--drop" in sys.argv:
    drop_pref = sys.argv[sys.argv.index("--drop") + 1].split(",")
lr = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else 0.1
wpow = float(sys.argv[sys.argv.index("--wpow") + 1]) if "--wpow" in sys.argv else 0.0

df = F.build("data", "cache")
tr_mask = ~df["is_test"].to_numpy()
y_all = np.full(df.height, -1)
y_all[tr_mask] = df.filter(~pl.col("is_test"))[TARGET].replace_strict({c: i for i, c in enumerate(CLASSES)}).to_numpy()
X = to_pandas_X(df)
X = X[[c for c in X.columns if not any(c.startswith(p) for p in drop_pref)]]
if "--hc-num" in sys.argv:  # kategori kardinalitas tinggi -> kode integer
    for c in ["hub_id", "kota_asal", "kota_tujuan", "package_id", "promo_id"]:
        if c in X: X[c] = X[c].cat.codes.astype(np.float32)
tr_all = np.where(tr_mask)[0]
folds = make_folds(y_all[tr_all])
tr_idx, va_idx = tr_all[folds != 0], tr_all[folds == 0]
Xtr, Xva = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
if use_te:
    codes = te_codes(df)
    add_te([Xtr, Xva], codes, y_all, tr_idx, [va_idx])
ytr, yva = y_all[tr_idx], y_all[va_idx]
cw = (np.bincount(ytr) / len(ytr)) ** (-wpow)
REG = "--reg" in sys.argv
params = dict(objective="multiclass", num_class=K, learning_rate=lr, num_leaves=63 if REG else 127,
              min_data_in_leaf=400 if REG else 100, feature_fraction=0.3 if REG else 0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0 if REG else 1.0, max_bin=255,
              cat_smooth=20, max_cat_to_onehot=8, verbose=-1, num_threads=10, seed=0)
t = time.time()
dtr = lgb.Dataset(Xtr, ytr, weight=cw[ytr])
dva = lgb.Dataset(Xva, yva, reference=dtr)
m = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
p = m.predict(Xva, num_iteration=m.best_iteration)
w = optimize_class_weights(p, yva)
res = dict(tag=tag, n_feat=Xtr.shape[1], iters=m.best_iteration, logloss=float(m.best_score["valid_0"]["multi_logloss"]),
           f1_argmax=macro_f1(yva, p), f1_w=macro_f1(yva, p, w), mins=(time.time() - t) / 60)
print(json.dumps(res))
imp = sorted(zip(m.feature_importance("gain"), Xtr.columns), reverse=True)
tot = sum(i for i, _ in imp)
print("TOP:", [(c, round(i / tot * 100, 2)) for i, c in imp[:40]])
import pandas as pd; pd.DataFrame(imp, columns=["gain", "feat"]).to_csv(f"cache/imp_{tag}.csv", index=False)
print("ZERO:", [c for i, c in imp if i == 0])
open("cache/exp_log.jsonl", "a").write(json.dumps(res) + "\n")
np.save(f"cache/exp_{tag}_p.npy", p)
