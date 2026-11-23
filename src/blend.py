"""Ensemble + optimasi bobot kelas + submission. Usage: python src/blend.py lgb_s0 xgb_s0 cat_s0"""
import json, os, sys
import numpy as np, polars as pl
from scipy.optimize import minimize
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.dirname(__file__))
from common import CLASSES, K, macro_f1, optimize_class_weights

ART = "artifacts"
names = sys.argv[1:]
folds = np.load(f"{ART}/folds.npy")
sh = pl.read_csv("data/train/shipments.csv", columns=["status_penanganan"])
y = sh["status_penanganan"].replace_strict({c: i for i, c in enumerate(CLASSES)}).to_numpy()
oofs = [np.load(f"{ART}/oof_{n.replace('_s', '_s', 1)}.npy") for n in names]
tests = [np.load(f"{ART}/test_{n}.npy") for n in names]
for n, o in zip(names, oofs):
    print(f"{n:10s} logloss={log_loss(y, o):.4f} f1={macro_f1(y, o):.4f}")

# bobot blend: minimisasi logloss (geometric mean of probs, softmax weights)
def blend(ps, w):
    lp = sum(wi * np.log(np.clip(p, 1e-7, 1)) for wi, p in zip(w, ps))
    e = np.exp(lp - lp.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

if len(names) > 1:
    obj = lambda a: log_loss(y, blend(oofs, np.exp(a) / np.exp(a).sum()))
    a = minimize(obj, np.zeros(len(names)), method="Nelder-Mead", options=dict(maxiter=200)).x
    bw = np.exp(a) / np.exp(a).sum()
else:
    bw = np.ones(1)
oof_b, test_b = blend(oofs, bw), blend(tests, bw)
print("blend weights", dict(zip(names, bw.round(3))), f"logloss={log_loss(y, oof_b):.4f} f1={macro_f1(y, oof_b):.4f}")

# validasi nested bobot kelas: fit di 4 fold, evaluasi di fold sisanya
nested = []
for k in range(5):
    w = optimize_class_weights(oof_b[folds != k], y[folds != k])
    nested.append((macro_f1(y[folds == k], oof_b[folds == k]), macro_f1(y[folds == k], oof_b[folds == k], w)))
nested = np.array(nested)
print("nested per-fold argmax:", nested[:, 0].round(4), "weighted:", nested[:, 1].round(4))
print(f"nested mean argmax={nested[:,0].mean():.4f}±{nested[:,0].std():.4f}  weighted={nested[:,1].mean():.4f}±{nested[:,1].std():.4f}")

cw = optimize_class_weights(oof_b, y)
print("class weights", dict(zip(CLASSES, cw.round(3))), f"OOF f1={macro_f1(y, oof_b, cw):.4f}")

pred = (test_b * cw).argmax(1)
ids = pl.read_csv("data/test/shipments.csv", columns=["shipment_id"])["shipment_id"]
sub = pl.DataFrame({"shipment_id": ids, "status_penanganan": [CLASSES[i] for i in pred]})
ss = pl.read_csv("data/sample_submission.csv")
assert sub.height == ss.height == 362245 and (sub["shipment_id"] == ss["shipment_id"]).all()
tag = "+".join(names)
os.makedirs("submissions", exist_ok=True)
sub.write_csv(f"submissions/sub_{tag}.csv")
print(sub["status_penanganan"].value_counts(normalize=True).sort("proportion", descending=True))
json.dump(dict(models=names, blend_weights=bw.tolist(), class_weights=cw.tolist(),
               oof_f1=macro_f1(y, oof_b, cw), nested_f1=nested[:, 1].mean(), nested_std=nested[:, 1].std()),
          open(f"{ART}/ensemble_{tag}.json", "w"), indent=1)
