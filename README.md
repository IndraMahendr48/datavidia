# Datavidia 11.0 — Task 1: Klasifikasi Status Penanganan Pengiriman

Klasifikasi multikelas (7 kelas) `status_penanganan`, metrik **Macro F1**.
Ringkasan kompetisi & EDA: [Datavidia11_Task1_Overview.md](Datavidia11_Task1_Overview.md).

## Struktur
```
data.py                     download data kompetisi → data/
src/features.py             join 9 tabel + agregasi 1:N + fitur turunan (tanpa target)
src/te.py                   target encoding multikelas OOF-safe
src/common.py               matriks fitur, folds, metrik, parameter & fit/predict model
src/train.py                training 5-fold: python src/train.py {lgb|xgb|cat} [seed]
src/blend.py                ensemble + optimasi bobot kelas (nested-CV) + submission
src/exp_lgb.py              eksperimen cepat 1 fold
datavidia11_task1_skeleton.ipynb   kerangka notebook final
```

## Quickstart
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export KAGGLE_KEY="KGAT_..."        # token Kaggle sendiri, jangan di-commit
python data.py
python src/train.py lgb 0           # CPU
USE_GPU=1 python src/train.py cat 0 # GPU
python src/blend.py lgb_s0 cat_s0 xgb_s0
```

Training GPU untuk anggota tim: lihat **[SETUP_GPU_TEMAN.md](SETUP_GPU_TEMAN.md)**.

## Validasi
- StratifiedKFold 5-fold (seed 42), fold yang sama untuk semua model.
- Target encoding dihitung **di dalam** fold (inner OOF) → tidak bocor.
- Bobot kelas untuk Macro F1 divalidasi nested (fit 4 fold, evaluasi fold ke-5).
