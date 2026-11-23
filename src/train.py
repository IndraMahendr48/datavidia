"""Training 5-fold penuh. Usage: python src/train.py {lgb|xgb|cat} [seed]

Env: USE_GPU=1 (CatBoost/XGBoost di GPU), QUICK=1 (smoke test), DATA_DIR (default: data)

Output (di artifacts/):
  models/{model}_s{seed}_f{k}.{txt|json|cbm}
  te/te_f{k}_*.parquet       : tabel target encoding per fold (untuk inference)
  features.json              : urutan kolom fitur
  oof_{model}_s{seed}.npy    : proba OOF train (n_train × 7)
  test_{model}_s{seed}.npy   : proba test rata-rata 5 fold
"""
import gc, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import features as F
from common import TE_KEYS, make_folds, prepare_matrix, run_cv, te_codes

MODEL = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ART = "artifacts"
os.makedirs(ART, exist_ok=True)

df = F.build(os.environ.get("DATA_DIR", "data"), "cache")
X, y_all, tr_all, te_all = prepare_matrix(df)
codes = te_codes(df)
df = df.select(sorted({c for cols, _ in TE_KEYS.values() for c in cols}))  # hemat RAM
gc.collect()
folds = make_folds(y_all[tr_all])
np.save(f"{ART}/folds.npy", folds)
run_cv(MODEL, SEED, df, X, y_all, tr_all, te_all, codes, folds, ART, quick=os.environ.get("QUICK") == "1")
