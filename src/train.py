"""Training 5-fold penuh. Usage: python src/train.py {lgb|xgb|cat} [seed]

Output (di artifacts/):
  models/{model}_s{seed}_f{k}.{txt|json|cbm}
  te/te_f{k}.parquet         : tabel target encoding per fold (untuk inference)
  oof_{model}_s{seed}.npy    : proba OOF train (n_train × 7)
  test_{model}_s{seed}.npy   : proba test rata-rata 5 fold
"""
import json, os, sys, time
import numpy as np, pandas as pd, polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import features as F
from common import *

MODEL = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ART = "artifacts"
os.makedirs(f"{ART}/models", exist_ok=True)
os.makedirs(f"{ART}/te", exist_ok=True)

DATA_DIR = os.environ.get("DATA_DIR", "data")
df = F.build(DATA_DIR, "cache")
X, y_all, tr_all, te_all = prepare_matrix(df)
codes = te_codes(df)
key_cols = sorted({c for cols, _ in TE_KEYS.values() for c in cols})
df = df.select(key_cols)  # hemat RAM: sisakan kolom kunci TE saja
import gc; gc.collect()
folds = make_folds(y_all[tr_all])
np.save(f"{ART}/folds.npy", folds)

oof = np.zeros((len(tr_all), K), dtype=np.float32)
test_p = np.zeros((len(te_all), K), dtype=np.float32)
log = []
QUICK = os.environ.get("QUICK") == "1"  # smoke test: 1 fold, 100rb baris
for k in range(1 if QUICK else 5):
    t0 = time.time()
    tr_idx, va_idx = tr_all[folds != k], tr_all[folds == k]
    if QUICK:
        tr_idx = np.random.default_rng(0).choice(tr_idx, 100_000, replace=False)
    Xtr, Xva, Xte = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy(), X.iloc[te_all].copy()
    tables = add_te([Xtr, Xva, Xte], codes, y_all, tr_idx, [va_idx, te_all], seed=k, keep_tables=True)
    if SEED == 0 and MODEL == "lgb" and not QUICK:
        save_te_tables(df, codes, tables, f"{ART}/te/te_f{k}.parquet")
    ytr, yva = y_all[tr_idx], y_all[va_idx]
    model, p_va, p_te, it = fit_predict(MODEL, Xtr, ytr, Xva, yva, Xte, SEED + k,
                                        f"{ART}/models/{MODEL}_s{SEED}_f{k}")
    oof[folds == k], test_p = p_va, test_p + p_te / 5
    del Xtr, Xva, Xte, model; gc.collect()
    r = dict(fold=k, iters=it, f1=macro_f1(yva, p_va), mins=round((time.time() - t0) / 60, 1))
    print(json.dumps(r), flush=True)
    log.append(r)

if QUICK:
    print("QUICK smoke test selesai — instalasi OK. Jalankan ulang tanpa QUICK=1 untuk training penuh.")
    sys.exit(0)
y_tr = y_all[tr_all]
w = optimize_class_weights(oof, y_tr)
summary = dict(model=MODEL, seed=SEED, f1_argmax=macro_f1(y_tr, oof), f1_w=macro_f1(y_tr, oof, w),
               fold_f1=[r["f1"] for r in log], weights=w.tolist())
print(json.dumps(summary), flush=True)
np.save(f"{ART}/oof_{MODEL}_s{SEED}.npy", oof)
np.save(f"{ART}/test_{MODEL}_s{SEED}.npy", test_p)
json.dump(summary, open(f"{ART}/summary_{MODEL}_s{SEED}.json", "w"), indent=1)
