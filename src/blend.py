"""Ensemble + aturan keputusan Macro F1 + submission. Usage: python src/blend.py cat_s0 xgb_s0 [lgb_s0 ...]

Aturan keputusan: argmax(P · W). W dimulai dari bobot kelas diagonal lalu disempurnakan menjadi matriks penuh
(lihat common.optimize_decision_matrix). Keduanya divalidasi nested (dicari di 4 fold, diuji di fold ke-5).
"""
import json, os, sys
import numpy as np, polars as pl
from joblib import Parallel, delayed
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.dirname(__file__))
from common import (CLASSES, blend, f1_matrix, fit_blend_weights, macro_f1, optimize_class_weights,
                    optimize_decision_matrix)

ART = "artifacts"
names = sys.argv[1:]
folds = np.load(f"{ART}/folds.npy")
sh = pl.read_csv("data/train/shipments.csv", columns=["status_penanganan"])
y = sh["status_penanganan"].replace_strict({c: i for i, c in enumerate(CLASSES)}).to_numpy()
oofs = [np.load(f"{ART}/oof_{n}.npy") for n in names]
tests = [np.load(f"{ART}/test_{n}.npy") for n in names]
for n, o in zip(names, oofs):
    print(f"{n:10s} logloss={log_loss(y, o):.4f} f1={macro_f1(y, o):.4f}")

bw = fit_blend_weights(oofs, y)
oof_b, test_b = blend(oofs, bw), blend(tests, bw)
print("blend weights", dict(zip(names, bw.round(3))), f"logloss={log_loss(y, oof_b):.4f} f1={macro_f1(y, oof_b):.4f}")


def nested_fold(k):  # cari di 4 fold, uji di fold k
    a, b = folds != k, folds == k
    w = optimize_class_weights(oof_b[a], y[a])
    W = optimize_decision_matrix(oof_b[a], y[a], w)
    return macro_f1(y[b], oof_b[b]), macro_f1(y[b], oof_b[b], w), f1_matrix(y[b], oof_b[b], W)


nested = np.array(Parallel(n_jobs=5)(delayed(nested_fold)(k) for k in range(5)))
print("nested per-fold  argmax:", nested[:, 0].round(4), " diag:", nested[:, 1].round(4), " matriks:", nested[:, 2].round(4))
print(f"NESTED mean  argmax={nested[:,0].mean():.4f}  diag={nested[:,1].mean():.4f}±{nested[:,1].std():.4f}  "
      f"matriks={nested[:,2].mean():.4f}±{nested[:,2].std():.4f}")

cw = optimize_class_weights(oof_b, y)
W = optimize_decision_matrix(oof_b, y, cw)
print("class weights", dict(zip(CLASSES, cw.round(3))), f"OOF f1 diag={macro_f1(y, oof_b, cw):.4f} matriks={f1_matrix(y, oof_b, W):.4f}")

pred = (test_b @ W).argmax(1)
ids = pl.read_csv("data/test/shipments.csv", columns=["shipment_id"])["shipment_id"]
sub = pl.DataFrame({"shipment_id": ids, "status_penanganan": [CLASSES[i] for i in pred]})
ss = pl.read_csv("data/sample_submission.csv")
assert sub.height == ss.height == 362245 and (sub["shipment_id"] == ss["shipment_id"]).all()
tag = "+".join(names)
os.makedirs("submissions", exist_ok=True)
sub.write_csv(f"submissions/sub_{tag}_mat.csv")
print(f"→ submissions/sub_{tag}_mat.csv")
print(sub["status_penanganan"].value_counts(normalize=True).sort("proportion", descending=True))
json.dump(dict(models=names, blend_weights=bw.tolist(), class_weights=cw.tolist(), decision_matrix=W.tolist(),
               oof_f1=f1_matrix(y, oof_b, W), nested_f1_diag=nested[:, 1].mean(), nested_f1=nested[:, 2].mean(),
               nested_std=nested[:, 2].std()),
          open(f"{ART}/ensemble_{tag}.json", "w"), indent=1)
