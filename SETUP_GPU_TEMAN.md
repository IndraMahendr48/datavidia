# Panduan Setup & Training GPU (CatBoost + XGBoost)

Halo! Tugasmu: jalankan training **CatBoost** dan **XGBoost** di GPU, lalu kirim hasilnya ke Indra.
Ikuti langkahnya berurutan, sebagian besar tinggal copy-paste.

**Kebutuhan:** GPU NVIDIA (VRAM ≥ 6 GB) + driver terbaru, RAM ≥ 16 GB (tutup aplikasi berat saat training),
disk kosong ±10 GB, Python **3.12**.

---

## 1. Clone repo & buat environment

```bash
git clone https://github.com/IndraMahendr48/datavidia.git
cd datavidia
python -m venv .venv
```

Aktifkan environment:

- **Linux / macOS:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

Lalu install library:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Cek GPU terdeteksi:

```bash
nvidia-smi
python -c "from catboost.utils import get_gpu_device_count as g; print('GPU CatBoost:', g())"
```

Angka GPU harus **≥ 1**. Kalau 0, update driver NVIDIA dulu.

## 2. Download data

Pakai token Kaggle **milikmu sendiri** (Kaggle → Settings → API → *Create New Token*). Jangan pernah di-commit.

- **Linux / macOS:**
  ```bash
  export KAGGLE_KEY="KGAT_tokenkamu"
  python data.py
  ```
- **Windows (PowerShell):**
  ```powershell
  $env:KAGGLE_KEY="KGAT_tokenkamu"
  python data.py
  ```

Hasilnya harus ada folder `data/train/`, `data/test/`, dan file `data/sample_submission.csv`.
(Kamu juga harus sudah join kompetisinya di Kaggle.)

## 3. Smoke test (±2–5 menit)

Untuk memastikan semua jalan sebelum training panjang:

- **Linux / macOS:** `USE_GPU=1 QUICK=1 python src/train.py cat 0`
- **Windows (PowerShell):**
  ```powershell
  $env:USE_GPU="1"; $env:QUICK="1"; python src/train.py cat 0
  ```

Harus muncul `QUICK smoke test selesai — instalasi OK`. Run pertama sekalian membangun cache fitur (`cache/features_v2.parquet`).

> **Windows:** setelah smoke test, hapus flag QUICK dulu: `Remove-Item Env:QUICK`

## 4. Training penuh

Jalankan **satu per satu, jangan bersamaan** (RAM bisa habis):

- **Linux / macOS:**
  ```bash
  USE_GPU=1 python src/train.py cat 0 2>&1 | tee train_cat_s0.log
  USE_GPU=1 python src/train.py xgb 0 2>&1 | tee train_xgb_s0.log
  ```
- **Windows (PowerShell):**
  ```powershell
  $env:USE_GPU="1"
  python src/train.py cat 0 2>&1 | Tee-Object train_cat_s0.log
  python src/train.py xgb 0 2>&1 | Tee-Object train_xgb_s0.log
  ```

Setiap model terdiri dari 5 fold. Tiap fold mencetak satu baris seperti
`{"fold": 0, "iters": ..., "f1": 0.61..., "mins": ...}`, dan di akhir muncul ringkasan `{"model": "cat", ... "f1_w": ...}`.
Kalau ada error, kirim screenshot / isi file log ke Indra.

## 5. Kirim hasil ke Indra ✅

Setelah **CatBoost selesai** (XGBoost menyusul kalau sudah selesai juga), zip file berikut:

```
artifacts/oof_cat_s0.npy
artifacts/test_cat_s0.npy
artifacts/summary_cat_s0.json
artifacts/models/cat_s0_f0.cbm ... cat_s0_f4.cbm
train_cat_s0.log
# kalau XGBoost sudah selesai juga:
artifacts/oof_xgb_s0.npy
artifacts/test_xgb_s0.npy
artifacts/summary_xgb_s0.json
artifacts/models/xgb_s0_f0.json ... xgb_s0_f4.json
train_xgb_s0.log
```

Contoh perintah zip:

- **Linux / macOS:** `zip -r hasil_gpu.zip artifacts/*cat* artifacts/*xgb* artifacts/models train_*.log`
- **Windows (PowerShell):** `Compress-Archive -Path artifacts\*cat*,artifacts\*xgb*,artifacts\models,train_*.log -DestinationPath hasil_gpu.zip`

Upload `hasil_gpu.zip` ke **Google Drive** lalu kirim link-nya ke Indra.
Jangan push file ini ke GitHub (ukurannya besar dan sudah di-`.gitignore`).

---

### Troubleshooting
| Masalah | Solusi |
|---|---|
| `Killed` / `MemoryError` | Tutup aplikasi lain; pastikan cuma 1 training yang jalan |
| CatBoost `CUDA out of memory` | Edit `src/common.py`, di `CAT_PARAMS` ubah `border_count=128` → `64` |
| GPU CatBoost = 0 | Update driver NVIDIA; cek `nvidia-smi` jalan |
| `FileNotFoundError: data/...` | Ulangi langkah 2; pastikan ada `data/train/shipments.csv` |
| Ingin cek progress | Lihat isi `train_cat_s0.log` |
