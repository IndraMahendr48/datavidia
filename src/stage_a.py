"""Tahap A residual TE: OOF proba dari model TANPA informasi label customer (TE customer dibuang).

Residual R = one-hot(y) − proba_A dipakai sebagai efek acak customer (mixed-effects) di model utama (RESID=1).
Usage: USE_GPU=1 python src/stage_a.py   →  artifacts/stageA_oof.npy (n_train × 7)
"""
import json, os, sys

import numpy as np
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.dirname(__file__))
import features as F
from common import macro_f1, make_folds, prepare_matrix, stage_a_oof

df = F.build(os.environ.get("DATA_DIR", "data"), "cache")
X, y_all, tr_all, te_all = prepare_matrix(df)
folds = make_folds(y_all[tr_all])
oof = stage_a_oof(X, y_all, tr_all, folds, df)
np.save("artifacts/stageA_oof.npy", oof)
print(json.dumps(dict(stage="A", f1=macro_f1(y_all[tr_all], oof), logloss=float(log_loss(y_all[tr_all], oof)))))
