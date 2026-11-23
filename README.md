# Datavidia 11.0 — Task 1: Klasifikasi Status Penanganan Pengiriman

Klasifikasi multikelas (7 kelas) `status_penanganan`, metrik **Macro F1**.
Konteks kompetisi & EDA awal: [Datavidia11_Task1_Overview.md](Datavidia11_Task1_Overview.md).

> **TL;DR untuk anggota tim**
> 1. Setup sekali (§2), lalu download data (§3).
> 2. Mau **bikin submission / lihat notebook** → ambil `artifacts/` dari Drive, jalankan notebook (§4A).
> 3. Mau **training model** → §4B (GPU: [SETUP_GPU_TEMAN.md](SETUP_GPU_TEMAN.md)).
> 4. Mau **coba ide fitur / parameter** → §4C, dan **wajib** ikuti aturan di §6.

---

## 1. Status & hasil saat ini

| Komponen | Status | Catatan |
|---|---|---|
| Pipeline fitur (`features_v2`) | ✅ **dibekukan** | 560 fitur; perubahan hanya lewat aturan §6.2 |
| LightGBM 5-fold | ⏳ di PC teman (CPU) | fold 0 (lr 0,03): Macro F1 argmax 0,6167 · eksperimen 1 fold + bobot kelas ±0,623 |
| CatBoost 5-fold | ⏳ di PC teman (GPU) | lihat [SETUP_GPU_TEMAN.md](SETUP_GPU_TEMAN.md) |
| XGBoost 5-fold | ⏳ di PC teman (GPU) | idem |
| Ensemble + bobot kelas | script siap | `src/blend.py` |
| Notebook final | draft, belum dieksekusi penuh | `datavidia11_task1_solution.ipynb` (di-generate, lihat §5); diisi setelah artefak lengkap |
| Tes end-to-end (data kecil) | ✅ lolos | training + inference + reproducibility untuk LGB, Cat, XGB |

Referensi LB (23 Nov): top ±0,602 · top-20 ≥ 0,594.

## 2. Setup (sekali saja)

Butuh **Python 3.12**, RAM ≥ 16 GB, disk kosong ±10 GB.

```bash
git clone https://github.com/IndraMahendr48/datavidia.git
cd datavidia
python3.12 -m venv .venv            # Windows: py -3.12 -m venv .venv
source .venv/bin/activate           # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

macOS: LightGBM butuh OpenMP → `brew install libomp`.

## 3. Download data

Pakai **token Kaggle milik sendiri** (Kaggle → Settings → API → *Create New Token*) dan pastikan sudah join kompetisi.

```bash
export KAGGLE_KEY="KGAT_tokenkamu"  # Windows PowerShell: $env:KAGGLE_KEY="KGAT_tokenkamu"
python data.py
```

Hasil: `data/train/*.csv`, `data/test/*.csv`, `data/sample_submission.csv`.
Run pertama apa pun akan membangun cache fitur `cache/features_v2.parquet` (±10–30 detik).

## 4. Cara pakai

### 4A. Bikin submission dari model yang sudah ada (paling sering)

1. Download folder **`artifacts/`** dari Google Drive tim, taruh di root repo:
   ```
   artifacts/
     folds.npy  features.json  ensemble.json
     models/   lgb_s0_f0..4.txt   cat_s0_f0..4.cbm   xgb_s0_f0..4.json
     te/       te_f0..4_*.parquet
     oof_*.npy  test_*.npy  summary_*.json
   ```
2. Jalankan notebook (default `MODE = "inference"`, tidak retrain):
   ```bash
   jupyter notebook datavidia11_task1_solution.ipynb      # Run All
   # atau tanpa UI:
   jupyter nbconvert --to notebook --execute --inplace datavidia11_task1_solution.ipynb
   ```
3. Hasil: **`submission.csv`** di root + cek otomatis format (362.245 baris, urutan sama dengan sample) dan
   cek reproducibility (proba = saat training).

Alternatif cepat tanpa notebook: `python src/blend.py lgb_s0 cat_s0 xgb_s0` → `submissions/sub_lgb_s0+cat_s0+xgb_s0.csv`.

### 4B. Training model 5-fold

```bash
python src/train.py lgb 0                 # LightGBM (CPU, ±1,5–2 jam)
USE_GPU=1 python src/train.py cat 0       # CatBoost (GPU NVIDIA)
USE_GPU=1 python src/train.py xgb 0       # XGBoost  (GPU NVIDIA)
```

* Argumen kedua = **seed** model (fold tetap sama). Seed lain (mis. `1`) → bisa di-ensemble sebagai `lgb_s1`.
* `QUICK=1` → smoke test 1 fold kecil (±2 menit) untuk cek instalasi.
* **Jalankan satu training per mesin**: dua sekaligus = RAM habis (`Killed`).
* Setelah selesai, **upload ke Drive** (§6.3), jangan ke GitHub.

### 4C. Eksperimen cepat (1 fold, 3–12 menit)

```bash
python src/exp_lgb.py <nama_eksperimen> --hc-num --reg --lr 0.05
```

Setting `--hc-num --reg --lr 0.05` = konfigurasi final saat ini (pembanding). Hasil otomatis tercatat di
`cache/exp_log.jsonl` (logloss, F1 argmax, F1 berbobot) dan importance di `cache/imp_<nama>.csv`.
Opsi lain: `--no-te` (tanpa target encoding), `--drop prefix1,prefix2` (buang grup fitur), `--wpow 0.5` (class weight).

## 5. Struktur repo

```
data.py                          download data → data/ (token dari env KAGGLE_KEY)
src/features.py                  join 9 tabel + agregasi 1:N + fitur turunan (TANPA target)
src/te.py                        target encoding multikelas OOF
src/common.py                    matriks fitur, folds, metrik, parameter model, fit/predict, run_cv, blend
src/train.py                     training 5-fold (dipakai semua anggota)
src/blend.py                     ensemble + bobot kelas (nested-CV) + submission
src/exp_lgb.py                   eksperimen 1 fold
tools/make_notebook.py           GENERATOR notebook final (kode diambil langsung dari src/)
datavidia11_task1_solution.ipynb notebook final (hasil generate, jangan diedit manual)
datavidia11_task1_skeleton.ipynb kerangka awal (referensi)
SETUP_GPU_TEMAN.md               panduan training GPU
```

**Notebook di-generate**, bukan diedit manual: ubah kode di `src/` atau teks di `tools/make_notebook.py`, lalu
`python tools/make_notebook.py`. Dengan begitu kode notebook selalu identik dengan kode yang melatih model.

## 6. Aturan tim

### 6.1 Aturan kompetisi (wajib, dari panitia)
* Maks **3 submission/hari** per tim; pilih **2 final**. Public LB = 30% test, private = 70%.
* Prediksi **harus dari model**: dilarang label manual, hardcoding, atau memakai ground truth.
* Wajib menyerahkan **notebook** (proses data → fitur → model → training → prediksi) + **file model**;
  notebook harus bisa **inference tanpa retrain**.
* Model eksternal (TabPFN dll.) hanya boleh bila bobotnya publik dan dijalankan sendiri; cantumkan sumber/versi.

### 6.2 Aturan eksperimen (anti-overfit)
1. **Percayai CV lokal, bukan public LB.** Public LB hanya 30% test; perbedaan ±0,003 di LB adalah noise.
2. Ide baru diuji dulu dengan `exp_lgb.py` (§4C) dan dibandingkan dengan konfigurasi final.
   Diterima hanya jika **logloss turun jelas** (≳ 0,002) **dan** F1 tidak turun. Kenaikan kecil = noise.
3. Semua yang memakai **label** (target encoding, stacking, bobot) **wajib dihitung di dalam fold**.
   Statistik grup/frekuensi boleh di train+test karena tidak memakai label.
4. **Fitur dibekukan** di `features_v2`. Mengubah fitur = semua model (LGB, Cat, XGB) harus **di-train ulang**.
   Jika ada perubahan yang diterima: naikkan versi `FEAT_FILE` di `src/features.py` (mis. `features_v3.parquet`),
   umumkan di grup, lalu semua orang `git pull` dan training ulang.
5. Jangan ubah `make_folds` (seed 42) supaya OOF semua model tetap bisa di-blend.

### 6.3 Aturan Git & file
* **Jangan commit**: data, `artifacts/`, `cache/`, `.venv/`, `*.npy/*.parquet/*.cbm`, dan **token Kaggle**
  (sudah di-`.gitignore`; tetap cek `git status` sebelum commit).
* Selalu `git pull` sebelum mulai kerja. Untuk perubahan kode, buat branch → push → minta review di grup:
  ```bash
  git checkout -b fitur/nama-ide
  git add src/... && git commit -m "Tambah fitur X (logloss 0,787 → 0,784)"
  git push -u origin fitur/nama-ide
  ```
* **Artefak model** dibagikan lewat Google Drive tim dengan struktur yang sama persis dengan folder `artifacts/`
  (zip per model, mis. `hasil_cat_s0.zip`), plus file log training-nya.

### 6.4 Aturan submission
* Sebelum submit, tulis di **log submission** (tabel di bawah, update lewat PR atau di grup): file, model, CV.
* Prioritas submit: (1) ensemble CV terbaik, (2) model tunggal terbaik sebagai **hedge**, (3) slot cadangan.
* **2 final pick:** ensemble dengan CV nested terbaik + 1 hedge yang lebih sederhana. Bukan yang public LB-nya tertinggi.

| Tanggal | File | Model | CV nested Macro F1 | Public LB | Oleh |
|---|---|---|---:|---:|---|
| | | | | | |

## 7. Troubleshooting

| Masalah | Solusi |
|---|---|
| `Killed` / exit 137 / `MemoryError` | RAM habis: tutup aplikasi lain, jalankan satu training saja |
| `FileNotFoundError: data/...` | Ulangi §3; pastikan ada `data/train/shipments.csv` |
| `Set dulu environment variable KAGGLE_KEY` | Set token (§3) di terminal yang sama sebelum `python data.py` |
| LightGBM error `libomp` (macOS) | `brew install libomp` |
| CatBoost GPU = 0 / CUDA error | Update driver NVIDIA, cek `nvidia-smi`; lihat [SETUP_GPU_TEMAN.md](SETUP_GPU_TEMAN.md) |
| Notebook: "Model LightGBM belum tersedia" | Folder `artifacts/` belum ada → download dari Drive (§4A) |
| `fold berbeda dengan saat training!` | `artifacts/folds.npy` dari versi lain; pakai artefak yang sesuai |
| Fitur berubah tapi hasil tidak berubah | Cache lama terpakai: hapus `cache/features_v*.parquet` atau naikkan `FEAT_FILE` |
