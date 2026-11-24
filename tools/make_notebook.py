"""Generator notebook final.

Kode fungsi di notebook diambil LANGSUNG dari src/*.py (via ast), sehingga notebook selalu identik
dengan kode yang dipakai untuk melatih model. Jalankan: python tools/make_notebook.py
"""
import ast
import os
import textwrap

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "datavidia11_task1_solution.ipynb")


def grab(file: str, *names: str) -> str:
    """Ambil source definisi top-level (fungsi/kelas/assign/if) berdasarkan nama, urutan sesuai argumen."""
    src = open(os.path.join(ROOT, "src", file)).read()
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        seg = ast.get_source_segment(src, node)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[node.name] = seg
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    found[t.id] = seg
        elif isinstance(node, ast.If) and "USE_GPU" in seg:
            found["GPU_SWITCH"] = seg
        elif isinstance(node, ast.If) and "QUICK" in seg:
            found["QUICK_SWITCH"] = seg
    missing = [n for n in names if n not in found]
    assert not missing, f"tidak ketemu di {file}: {missing}"
    return "\n\n\n".join(found[n] for n in names)


cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))
code_raw = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

# =====================================================================================
md("""
# Datavidia 11.0 — Task 1: Klasifikasi `status_penanganan`

**Metrik:** Macro F1 (7 kelas) · **Validasi:** Stratified 5-Fold OOF · **Model:** CatBoost + XGBoost (ensemble, GPU); LightGBM opsional

Notebook ini memuat seluruh pipeline: pemrosesan data, penggabungan 9 tabel, pembuatan fitur, pengembangan model,
training, ensemble, optimasi Macro F1, dan **inferensi tanpa retrain** (memuat file model tersimpan).

| Mode | Perilaku |
|---|---|
| `MODE = "inference"` *(default)* | Bangun fitur → muat model & tabel target-encoding dari `artifacts/` → `submission.csv` |
| `MODE = "train"` | Latih ulang semua model 5-fold dari nol (±1,5–3 jam/model di CPU; CatBoost/XGBoost bisa GPU) |

### Ringkasan pendekatan
1. **Fitur level paket** dari 9 tabel relasional: join 5 tabel master + agregasi 3 tabel log (scan, rute, invoice).
2. **Temuan kunci EDA:** label sangat *menempel* pada entitas: dua paket dari customer yang sama berlabel sama 44% kali
   (acak: 23%), kurir 35%. Maka sinyal terbesar ada di **efek laten customer / kurir / hub**, yang ditangkap lewat
   (a) statistik grup tanpa target dan (b) target encoding multikelas OOF yang dihitung **di dalam** fold.
3. **GBDT** multikelas, ensemble rata-rata geometrik proba, lalu **bobot per kelas** untuk memaksimalkan Macro F1
   (divalidasi nested supaya tidak overfit).

## Daftar Isi
0. Setup & Konfigurasi · 1. Load Data · 2. EDA Ringkas · 3. Join Tabel Lookup · 4. Agregasi Tabel 1:N ·
5. Feature Engineering Lanjutan · 6. Target Encoding OOF · 7. Validasi & Baseline · 8. Seleksi Fitur ·
9. Model & Training · 10. Ensemble · 11. Optimasi Macro F1 · 12. Analisis Error · 13. Simpan Artefak ·
14. Inferensi Tanpa Retrain · 15. Ringkasan & Dependensi
""")

# ===================================================================================== 0
md("""
## 0. Setup & Konfigurasi
Semua yang bisa diubah ada di sel ini. `DATA_DIR` dideteksi otomatis (lokal `data/` atau Kaggle `/kaggle/input/...`).
""")
code("""
import gc, json, os, random, time, warnings, platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import ks_2samp
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
pl.Config.set_tbl_rows(25); pl.Config.set_tbl_width_chars(160); pl.Config.set_fmt_str_lengths(40)

# ------------------------------------------------------------------ konfigurasi
MODE = "inference"                                   # "train" | "inference"
# (model, seed, residual TE?, override parameter) — kandidat ensemble; yang artefaknya tidak ada dilewati
MODEL_RUNS = [  # ensemble final (semua dengan residual TE customer, Section 6.1)
    ("cat", 11, True, {"learning_rate": 0.02, "iterations": 15000}),
    ("cat", 12, True, {"learning_rate": 0.04, "iterations": 10000}),
    ("xgb", 11, True, {}),
]
# Varian tanpa residual TE (hedge / pembanding), juga tersimpan di artifacts/:
#   ("cat", 1, False, {"learning_rate": 0.02, "iterations": 15000}), ("cat", 2, False, {"learning_rate": 0.04,
#   "iterations": 10000}), ("cat", 0, False, {}), ("xgb", 0, False, {})
SEED = 42                                            # seed pembagian fold
ART = "artifacts"                                    # model, tabel TE, OOF, proba test
CACHE = "cache"                                      # cache tabel fitur (parquet)
SUBMISSION_PATH = "submission.csv"


def find_data_dir() -> str:
    cands = [os.environ.get("DATA_DIR", ""), "data", "/kaggle/input/task-1-datavidia-11-0",
             "/kaggle/input/task-1-datavidia-11-0/Task-1-Datavidia-11.0"]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "train", "shipments.csv")):
            return c
    raise FileNotFoundError("Data tidak ditemukan. Set env DATA_DIR atau jalankan data.py")


def set_seed(s: int):
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"] = str(s)


DATA_DIR = find_data_dir()
set_seed(SEED)
os.makedirs(ART, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
print(f"MODE={MODE} | DATA_DIR={DATA_DIR} | Python {platform.python_version()}")
print({p: version(p) for p in ["numpy", "pandas", "polars", "scikit-learn", "lightgbm", "xgboost", "catboost"]})
""")
code_raw(grab("features.py", "CLASSES", "TARGET", "EVENTS", "DEVICES", "MODAS", "INV_TYPES")
         + "\nK = len(CLASSES)\nCLS_IDX = {c: i for i, c in enumerate(CLASSES)}")

# ===================================================================================== 1
md("""
## 1. Load Data & Optimasi Memori
Memakai **polars** (kolumnar, multi-thread): `scan_events` train ±6,8 juta baris terbaca dalam hitungan detik dan jauh
lebih hemat RAM daripada pandas. Strategi memori: **agregasi dulu di polars, baru join**, lalu matriks fitur
akhir dikonversi ke `float32`.
""")
code_raw(grab("features.py", "read"))
code("""
TABLE_NAMES = ["shipments", "scan_events", "route_segments", "invoices",
               "customers", "couriers", "hubs", "service_packages", "promotions"]
t0 = time.time()
tables = {sp: {nm: read(DATA_DIR, sp, nm) for nm in TABLE_NAMES} for sp in ["train", "test"]}
print(f"load {time.time() - t0:.1f}s")
pl.DataFrame([dict(table=nm, train_rows=tables["train"][nm].height, test_rows=tables["test"][nm].height,
                   cols=tables["train"][nm].width, train_MB=round(tables["train"][nm].estimated_size("mb"), 1))
              for nm in TABLE_NAMES])
""")
md("**Sanity check:** kunci unik, semua paket punya baris di tabel log, tidak ada baris yatim, tabel master identik train vs test.")
code("""
for sp in ["train", "test"]:
    sh = tables[sp]["shipments"]
    assert sh["shipment_id"].n_unique() == sh.height, "shipment_id tidak unik"
    ids = set(sh["shipment_id"])
    for nm in ["scan_events", "route_segments", "invoices"]:
        child = set(tables[sp][nm]["shipment_id"].unique())
        assert child == ids, f"{sp}/{nm}: coverage tidak penuh / ada baris yatim"
for nm in ["customers", "couriers", "hubs", "service_packages", "promotions"]:
    assert tables["train"][nm].equals(tables["test"][nm]), f"{nm} berbeda train vs test"
cov = {c: tables["test"]["shipments"][c].is_in(tables["train"]["shipments"][c].implode()).mean()
       for c in ["customer_id", "courier_id", "hub_id", "kota_asal", "kota_tujuan"]}
print("✓ semua sanity check lolos")
print("Proporsi ID test yang muncul di train:", {k: round(v, 3) for k, v in cov.items()})
""")

# ===================================================================================== 2
md("""
## 2. EDA Ringkas
Fokus pada hal yang **mengubah keputusan pemodelan**: distribusi target, kualitas data, pergeseran train↔test,
dan, yang terpenting, di mana sinyalnya berada.
""")
code("""
y_str = tables["train"]["shipments"][TARGET]
dist = y_str.value_counts().with_columns((pl.col("count") / y_str.len() * 100).round(2).alias("pct")) \\
            .sort("count", descending=True)
fig, ax = plt.subplots(figsize=(9, 3.6))
sns.barplot(data=dist.to_pandas(), y=TARGET, x="pct", ax=ax, color="#4C72B0")
for i, v in enumerate(dist["pct"]): ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
ax.set(title="Distribusi target (train)", xlabel="% baris", ylabel="")
plt.tight_layout(); plt.show()
print(f"Rasio mayoritas : minoritas = {dist['count'][0] / dist['count'][-1]:.1f} : 1 → perlu optimasi bobot kelas untuk Macro F1")
""")
md("### 2.1 Missing value & pergeseran train ↔ test")
code("""
rows = []
for nm in TABLE_NAMES:
    t = tables["train"][nm]
    for c, n in zip(t.columns, t.null_count().row(0)):
        if n: rows.append(dict(table=nm, col=c, missing_pct=round(n / t.height * 100, 2)))
display(pl.DataFrame(rows))

sh_tr = tables["train"]["shipments"]
print("Missing volume_l per kelas (%):",
      sh_tr.group_by(TARGET).agg((pl.col("volume_l").is_null().mean() * 100).round(2)).sort(TARGET).rows())

num_cols = [c for c, t in sh_tr.schema.items() if t.is_numeric()]
rng = np.random.default_rng(0)
ks = sorted(((ks_2samp(rng.choice(sh_tr[c].drop_nulls().to_numpy(), 100_000),
                       rng.choice(tables["test"]["shipments"][c].drop_nulls().to_numpy(), 100_000)).statistic, c)
             for c in num_cols), reverse=True)
print("KS statistic train vs test (terbesar):", [(c, round(s, 3)) for s, c in ks[:5]])
""")
md("""
Missing sedikit dan **tidak informatif** (proporsinya rata antar kelas), dan GBDT menanganinya secara native.
KS hampir semua kolom < 0,01. Pengecualiannya **`ship_month`** (KS ±0,03), yang diperiksa lebih dalam di bawah.
""")
code("""
mon = (sh_tr.group_by("ship_month").agg(pl.len().alias("n_train"),
                                         (pl.col(TARGET) == "normal").mean().alias("p_normal"),
                                         (pl.col(TARGET) == "tertunda_operasional").mean().alias("p_tertunda"))
       .join(tables["test"]["shipments"].group_by("ship_month").agg(pl.len().alias("n_test")), on="ship_month")
       .with_columns((pl.col("n_test") / pl.col("n_test").sum()) / (pl.col("n_train") / pl.col("n_train").sum()))
       .rename({"n_test": "rasio_test_train"}).drop("n_train").sort("ship_month"))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.4))
ax[0].bar(mon["ship_month"], mon["rasio_test_train"], color="#4C72B0"); ax[0].axhline(1, color="k", lw=.8)
ax[0].set(title="Proporsi bulan: test / train", xlabel="ship_month", ylim=(0.8, 1.15))
ax[1].plot(mon["ship_month"], mon["p_tertunda"], "o-", label="tertunda_operasional")
ax[1].plot(mon["ship_month"], mon["p_normal"], "o-", label="normal")
ax[1].set(title="Proporsi kelas per bulan (train)", xlabel="ship_month"); ax[1].legend()
plt.tight_layout(); plt.show()
mon.with_columns(pl.exclude("ship_month").round(3))
""")
md("""
**Temuan:** split train/test **tidak sepenuhnya acak** terhadap bulan. Test kelebihan bulan Okt–Des (rasio 1,08–1,12)
dan kekurangan Mei–Agt (0,88–0,95). Bulan **3, 10, 11, 12 adalah musim sibuk**: `tertunda_operasional` naik dari
±11% menjadi ±21%. Artinya prior kelas di test sedikit bergeser. Model tetap memakai `ship_month` sebagai fitur;
dampak pergeseran ini terhadap bobot kelas diuji di Section 11 (hasilnya: dapat diabaikan).
Untuk semua kolom lain, Stratified K-Fold tetap skema validasi yang tepat (lihat *adversarial validation* di Section 7).
""")
md("### 2.2 Temuan kunci: label menempel pada entitas")
code("""
def pair_agreement(df: pl.DataFrame, key: str) -> float:
    \"\"\"P(dua paket acak dengan `key` yang sama punya label sama).\"\"\"
    g = df.group_by(key, TARGET).len()
    n = df.group_by(key).len()
    return float((g["len"] * (g["len"] - 1)).sum() / (n["len"] * (n["len"] - 1)).sum())

p = sh_tr[TARGET].value_counts()["count"].to_numpy() / sh_tr.height
base = float((p ** 2).sum())
agree = {k: pair_agreement(sh_tr, k) for k in ["customer_id", "courier_id", "hub_id", "package_id", "kota_tujuan"]}
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(list(agree), list(agree.values()), color="#55A868")
ax.axvline(base, color="crimson", ls="--", label=f"acak = {base:.3f}")
for i, v in enumerate(agree.values()): ax.text(v + 0.003, i, f"{v:.3f}", va="center")
ax.set(title="Peluang 2 paket dengan entitas sama berlabel sama", xlabel="pair agreement"); ax.legend()
plt.tight_layout(); plt.show()
print("Paket per customer (train):", sh_tr.group_by("customer_id").len()["len"].describe().filter(
      pl.col("statistic").is_in(["mean", "50%", "max"])).rows())
""")
md("""
**Implikasi:** customer (44% vs 23% acak) dan kurir (35%) punya *kecenderungan laten* yang kuat.
Karena 88,6% customer test juga ada di train, riwayat label mereka (via **target encoding OOF**) dan profil perilaku
mereka (via **statistik grup**, misalnya rata-rata gap scan semua paket milik seorang kurir) menjadi fitur paling penting.
Satu customer rata-rata hanya punya ±2,4 paket, jadi smoothing TE customer harus **kecil**.
""")
md("### 2.3 Sinyal dari tabel log per kelas")
code("""
sc = tables["train"]["scan_events"]
scan_cls = (sc.group_by("shipment_id").agg(
                (pl.col("event_type") == "upaya_antar").sum().alias("n_upaya_antar"),
                pl.col("gap_hours").mean().alias("gap_mean"), pl.col("scan_quality").mean().alias("quality_mean"),
                pl.col("is_manual").mean().alias("manual_rate"))
            .join(sh_tr.select("shipment_id", TARGET), on="shipment_id")
            .group_by(TARGET).agg(pl.exclude("shipment_id").mean().round(3)))
inv_cls = (tables["train"]["invoices"].group_by("shipment_id")
           .agg((pl.col("invoice_type") == "klaim").sum().alias("n_inv_klaim"))
           .join(sh_tr.select("shipment_id", TARGET), on="shipment_id")
           .group_by(TARGET).agg(pl.col("n_inv_klaim").mean().round(3)))
rt_cls = (tables["train"]["route_segments"].group_by("shipment_id")
          .agg(pl.len().alias("n_kaki"), pl.col("durasi_jam").mean().alias("durasi_mean"))
          .join(sh_tr.select("shipment_id", TARGET), on="shipment_id")
          .group_by(TARGET).agg(pl.exclude("shipment_id").mean().round(3)))
summary_cls = scan_cls.join(inv_cls, on=TARGET).join(rt_cls, on=TARGET) \\
    .with_columns(pl.col(TARGET).replace_strict(CLS_IDX).alias("_o")).sort("_o").drop("_o")
summary_cls
""")
md("""
Pola sesuai intuisi domain: `upaya_antar` banyak → konfirmasi alamat / gagal antar; `gap` & durasi rute besar →
tertunda; kualitas scan rendah + scan manual → rusak; invoice `klaim` → retur & klaim hilang.
Perbedaannya **nyata tapi tipis** di level paket, sehingga diperlukan fitur level entitas.
""")
md("### 2.4 Kolom yang dicurigai noise")
code("""
def eta_squared(df: pl.DataFrame, col: str) -> float:
    d = df.select(pl.col(col).cast(pl.Float64), TARGET).drop_nulls()
    mu = d[col].mean()
    g = d.group_by(TARGET).agg(pl.col(col).mean().alias("m"), pl.len().alias("n"))
    return float(((g["m"] - mu) ** 2 * g["n"]).sum() / ((d[col] - mu) ** 2).sum())

eta = sorted(((eta_squared(sh_tr, c), c) for c in num_cols), reverse=True)
print("η² terbesar :", [(c, round(v, 4)) for v, c in eta[:6]])
print("η² ≈ 0      :", [c for v, c in eta if v < 1e-4])
""")
md("""
Kolom dengan η² ≈ 0 (termasuk 6 kolom berdistribusi N(50, σ) seperti `skor_prioritas_kirim`) **tidak langsung dibuang**:
bisa jadi berguna lewat interaksi. Keputusannya diambil di Section 8 berdasarkan importance model.
""")

# ===================================================================================== 3
md("""
## 3. Penggabungan Tabel Lookup (1:1)
`shipments` di-*left join* dengan 5 tabel master, dengan prefix kolom per tabel (`cust_`, `cour_`, `hub_`, `pkg_`, `promo_`).
Jumlah baris di-*assert* tidak berubah. Sekaligus dibuat fitur **relasi antar tabel** (kurir bertugas di hub yang
sama? region customer = tujuan? promo aktif di bulan kirim?), **rasio fisik** paket, dan **jam kirim vs jam operasional hub**.
""")
code_raw(grab("features.py", "prefix", "join_lookups"))
code("""
lookups = {k: tables["train"][k] for k in ["customers", "couriers", "hubs", "service_packages", "promotions"]}
_demo = join_lookups(tables["train"]["shipments"], lookups)
print(_demo.shape)
_demo.select("shipment_id", "rel_cour_same_hub", "rel_cust_region_eq_dest", "rel_promo_active",
             "r_value_per_kg", "t_outside_hub_hours", "route_pair").head(5)
""")

# ===================================================================================== 4
md("""
## 4. Agregasi Tabel 1:N
Setiap tabel log diringkas menjadi **1 baris per `shipment_id`**:

| Tabel | Fitur |
|---|---|
| `scan_events` | jumlah & proporsi per `event_type`; `gap_hours` sum/mean/max/min/std (total & per event); `scan_quality` mean/min/max/std & jumlah scan berkualitas rendah; scan manual; per `device_type`; **urutan**: event terakhir, gap & kualitas scan pertama/terakhir, posisi `upaya_antar` pertama, jumlah perpindahan event |
| `route_segments` | jumlah kaki; jarak, durasi, biaya (sum/max/mean); kecepatan; biaya/km; jumlah & durasi per moda; moda pertama/terakhir; hub unik |
| `invoices` | jumlah & nominal per tipe (`pengiriman`/`langganan`/**`klaim`**); diskon, pajak; `days_to_pay`; penalti; status bayar |

Urutan scan = urutan baris di file (sudah diverifikasi: baris tiap paket bersebelahan dan selalu diawali `pickup`).
""")
code_raw(grab("features.py", "agg_scan", "agg_route", "agg_invoice"))
code("""
_ids = tables["train"]["shipments"]["shipment_id"].head(20_000).implode()
for fn, nm in [(agg_scan, "scan_events"), (agg_route, "route_segments"), (agg_invoice, "invoices")]:
    a = fn(tables["train"][nm].filter(pl.col("shipment_id").is_in(_ids)))
    print(f"{fn.__name__:12s} → {a.shape[1] - 1:3d} fitur | contoh kolom: {a.columns[1:7]}")
""")

# ===================================================================================== 5
md("""
## 5. Feature Engineering Lanjutan
Fungsi `build()` menyatukan semuanya untuk train **dan** test dengan kode yang sama, lalu menambahkan:

* **Fitur lintas tabel:** durasi rute aktual vs `estimasi_hari` paket, total gap scan vs estimasi, selisih jarak
  segmen vs `distance_km`, nominal klaim / nilai barang, biaya rute / nilai barang.
* **Frequency encoding** ID (customer, kurir, hub, kota, rute).
* **Statistik grup tanpa target** untuk customer (`gc_`), kurir (`gk_`), hub (`gh_`), rute (`gr_`), paket layanan (`gp_`):
  rata-rata 26 fitur perilaku di level entitas + deviasi paket dari rata-rata entitasnya. Ini **profil laten**
  entitas (misalnya kurir yang scan-nya sering telat / berkualitas rendah). Dihitung di train+test tanpa label,
  jadi **tidak ada leakage**.

ID kardinalitas tinggi (`customer_id`, `courier_id`, ...) **tidak** dipakai mentah sebagai fitur.
""")
code_raw(grab("features.py", "CAT_COLS", "ID_COLS", "FEAT_FILE", "build"))
code("""
t0 = time.time()
df = build(DATA_DIR, CACHE, tables)
n_tr = tables["train"]["shipments"].height
assert df.height == n_tr + tables["test"]["shipments"].height
assert (df["shipment_id"][:n_tr] == tables["train"]["shipments"]["shipment_id"]).all()       # urutan terjaga
assert (df["shipment_id"][n_tr:] == tables["test"]["shipments"]["shipment_id"]).all()
print(f"tabel fitur {df.shape} dibangun dalam {time.time() - t0:.0f}s, {df.estimated_size('gb'):.2f} GB")
groups = pd.Series([c.split("_")[0] if c.split("_")[0] in {"sc", "rt", "iv", "gc", "gk", "gh", "gr", "gp", "fq",
                    "cust", "cour", "hub", "pkg", "promo", "rel", "r", "t", "x"} else "shipments" for c in df.columns])
print(groups.value_counts().to_dict())
del tables, _demo, sc; gc.collect();
""")

# ===================================================================================== 6
md("""
## 6. Target Encoding Multikelas OOF-safe ⚠️
Untuk setiap kunci, encoding = proporsi 7 kelas tersmoothing ke prior: $\\hat p_k = \\frac{n_k + m\\,\\pi_k}{n + m}$, plus jumlah sampel $n$.

**Skema anti-leakage (dua tingkat):**
1. Di **dalam** setiap fold model (outer fold *k*), TE untuk baris training dihitung dengan **inner 5-fold OOF**:
   sebuah baris tidak pernah melihat labelnya sendiri.
2. TE untuk baris validasi dan test dihitung dari **seluruh** bagian training fold *k* (tidak menyentuh label validasi).
3. Tabel TE tiap fold disimpan (`artifacts/te/`) → inference test memakai tabel yang sama persis dengan saat training.

Smoothing `m` disesuaikan dengan jumlah data per level: customer hanya ±2,4 paket → `m=2`; kurir ±52 paket → `m=10`;
kunci lain `m=30`.
""")
code_raw(grab("te.py", "fit_te", "oof_te", "encode_fold"))
code_raw(grab("common.py", "TE_KEYS", "te_codes", "add_te", "save_te_tables", "apply_te_tables"))
md("**Demo kenapa OOF wajib:** TE customer yang dihitung *in-sample* (baris melihat labelnya sendiri) terlihat sangat bagus, padahal itu ilusi.")
code("""
_y = df.filter(~pl.col("is_test"))[TARGET].replace_strict(CLS_IDX).to_numpy()
_c = df.filter(~pl.col("is_test")).select(pl.col("customer_id").rank("dense").cast(pl.Int64) - 1).to_series().to_numpy()
_prior = np.bincount(_y, minlength=K) / len(_y)
_insample, _ = fit_te(_c, _y, _c.max() + 1, K, 2.0, _prior)
_oof, _ = oof_te(_c, _y, _c.max() + 1, K, 2.0)
_w = 1 / _prior  # koreksi prior supaya argmax tidak selalu 'normal'
print(f"Macro F1 argmax TE customer IN-SAMPLE (bocor): {f1_score(_y, (_insample[_c] * _w).argmax(1), average='macro'):.4f}")
print(f"Macro F1 argmax TE customer OOF (jujur)     : {f1_score(_y, (_oof * _w).argmax(1), average='macro'):.4f}")
del _c, _insample, _oof; gc.collect();
""")
md("""
### 6.1 Residual target encoding customer (efek acak ala *mixed-effects*) ⭐
TE customer biasa = proporsi label paket lain milik customer. Masalahnya, label itu **tercampur efek fitur paket**:
paket lain berlabel `klaim_hilang` mungkin karena barangnya mahal, bukan karena customer-nya "rawan klaim".
Dengan rata-rata hanya ±2 paket/customer di train, campuran ini membuat estimasi kecenderungan customer bising.

Solusi (prinsip *mixed-effects* / GPBoost: tree-boosting + efek acak per grup, Sigrist 2022):
1. **Tahap A** — CatBoost 5-fold **tanpa** informasi label customer → proba OOF $\\hat p^A$.
2. **Residual** $R_i = \\text{onehot}(y_i) - \\hat p^A_i$ = bagian label yang **tidak** dijelaskan fitur paket.
3. **Residual TE** customer = rata-rata $R$ paket lain milik customer, di-*shrink* ke 0: $\\frac{\\sum R}{n + m}$ (m=1),
   dihitung dengan skema OOF dua tingkat yang sama seperti TE biasa; tabelnya disimpan per fold (`te/rte_f*`).

Anti-leakage: baris validasi/test hanya memakai residual baris train-fold; model tahap A tidak pernah melihat
label customer, sehingga kebocoran yang tersisa hanya ke arah **konservatif** (mengecilkan sinyal).
Hasil eksperimen fold 0 (CatBoost lr 0,08): logloss **0,7847 → 0,7817**, Macro F1 (+bobot) **0,6225 → 0,6259**
— peningkatan terbesar di seluruh eksperimen.
""")
code_raw(grab("te.py", "fit_resid", "encode_fold_resid"))
code_raw(grab("common.py", "RESID_KEYS", "stage_a_oof", "resid_matrix", "add_resid"))

# ===================================================================================== 7
md("""
## 7. Skema Validasi & Baseline
* `StratifiedKFold(5, shuffle=True, seed=42)`: fold yang **sama** untuk semua model (OOF bisa di-blend).
* Matriks fitur: kategori kardinalitas rendah → kategori native; kategori kardinalitas tinggi (`hub_id`, kota,
  paket, promo) → kode integer + TE (eksperimen menunjukkan kategori native ratusan level **overfit**).
""")
code_raw(grab("common.py", "to_pandas_X", "HC_CATS", "prepare_matrix", "make_folds", "macro_f1"))
code("""
X, y_all, tr_all, te_all = prepare_matrix(df)
codes = te_codes(df)
folds = make_folds(y_all[tr_all], 5, SEED)
if os.path.exists(f"{ART}/folds.npy"):
    assert (np.load(f"{ART}/folds.npy") == folds).all(), "fold berbeda dengan saat training!"
y_tr = y_all[tr_all]
print(f"X: {X.shape} | train {len(tr_all):,} | test {len(te_all):,} | fold sizes {np.bincount(folds)}")
maj = np.zeros((len(y_tr), K)); maj[:, 0] = 1
print(f"Baseline tebak mayoritas ('normal'): Macro F1 = {macro_f1(y_tr, maj):.4f}")
""")
md("**Adversarial validation:** bisakah model membedakan baris train vs test? AUC ≈ 0,5 → distribusi sama → CV lokal bisa dipercaya.")
code("""
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(tr_all, 150_000, replace=False), rng.choice(te_all, 150_000, replace=False)])
is_t = np.r_[np.zeros(150_000), np.ones(150_000)]
Xa = X.iloc[idx].drop(columns=[c for c in X.columns if c.startswith("fq_")])  # frekuensi dihitung gabungan
aucs = []
for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(Xa, is_t):
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, verbose=-1, num_threads=os.cpu_count()),
                  lgb.Dataset(Xa.iloc[a], is_t[a]), 200)
    aucs.append(roc_auc_score(is_t[b], m.predict(Xa.iloc[b])))
print(f"Adversarial AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
del Xa, m; gc.collect();
""")
md("""
### Log eksperimen (LightGBM, 1 fold holdout 20%, `lr=0.1`, dari `src/exp_lgb.py`)
Setiap perubahan diterima hanya jika **logloss** (metrik yang lebih stabil daripada F1) turun jelas.

| # | Perubahan | Fitur | Logloss | Macro F1 (argmax) | Macro F1 (+bobot kelas) | Keputusan |
|---|---|---:|---:|---:|---:|---|
| 0 | Baseline overview (300rb sampel, tanpa TE) | 160 | – | 0,4747 | 0,4782 | – |
| 1 | Data penuh + agregasi lengkap + TE OOF (m=30) | 356 | 0,8209 | 0,5979 | 0,6078 | ✅ |
| 2 | Smoothing TE per kunci (customer m=2, kurir m=10) | 356 | 0,8203 | 0,5987 | 0,6072 | ✅ (netral, lebih masuk akal) |
| 3 | Kategori kardinalitas tinggi → kode integer (bukan kategori native) | 356 | **0,8040** | 0,6066 | 0,6158 | ✅ besar |
| 4 | + statistik grup customer/kurir/hub/rute/paket (26 fitur × 5 kunci) | 560 | **0,7947** | 0,6112 | 0,6192 | ✅ |
| 5 | Regularisasi (leaves 63, min_leaf 400, colsample 0,3, λ=10, lr 0,05) | 560 | **0,7871** | 0,6150 | 0,6232 | ✅ final |

*Catatan: F1 "+bobot kelas" di tabel ini dioptimasi pada fold yang sama (optimistis); angka jujur ada di Section 11 (nested).*

### Log eksperimen lanjutan (CatBoost GPU, fold 0, `src/exp_cat.py`)
Bobot kelas dicari di separuh baris validasi dan dievaluasi di separuh lainnya (dua arah) → tidak optimistis.

| # | Perubahan (vs baseline CatBoost lr 0,08) | Logloss | Macro F1 argmax | Macro F1 +bobot | Keputusan |
|---|---|---:|---:|---:|---|
| C0 | Baseline CatBoost depth 8, lr 0,08 | 0,7847 | 0,6158 | 0,6225 | – |
| C1 | + TE interaksi bulan (hub×bulan, kurir×bulan, kota×bulan) | 0,7850 | 0,6159 | 0,6227 | ❌ netral |
| C2 | + TE interaksi kategori barang (kurir×kategori, customer×kategori, hub×region) | 0,7850 | 0,6157 | 0,6224 | ❌ netral |
| C3 | + TE interaksi rute (kurir×rute, hub×kota tujuan, customer×hub) | 0,7846 | 0,6155 | 0,6230 | ❌ marginal |
| C4 | `customer_id`, `courier_id` sebagai kategori native CatBoost | 0,7860 | 0,6140 | 0,6232 | ❌ logloss memburuk |
| C5 | depth 10, lr 0,05 | 0,7852 | 0,6152 | 0,6220 | ❌ |
| C6 | lr 0,04 | 0,7834 | 0,6163 | 0,6236 | ✅ |
| C7 | lr 0,02 | **0,7825** | 0,6168 | 0,6238 | ✅ |
| C8 | **+ residual TE customer (Section 6.1)** | **0,7817** | **0,6173** | **0,6259** | ✅ **terbesar** |
| C9 | + residual TE customer×hub | 0,7818 | 0,6163 | 0,6245 | ❌ |

Data juga diperiksa untuk pola tersembunyi (urutan baris, duplikasi pesanan, hub di `scan_events`, customer di
`invoices`, sebaran kolom noise per kelas): **tidak ada kebocoran** — semua sinyal berasal dari pemodelan.
""")

# ===================================================================================== 8
md("""
## 8. Seleksi Fitur
Importance dirata-rata dari 5 model fold (LightGBM *gain* bila ada, jika tidak CatBoost *PredictionValuesChange*).
Kolom yang dicurigai noise dicek posisinya.
""")
code("""
lgb_files = [f"{ART}/models/lgb_s0_f{k}.txt" for k in range(5)]
CAT_IMP = "cat_s11" if os.path.exists(f"{ART}/models/cat_s11_f0.cbm") else "cat_s0"  # model residual bila ada
cat_files = [f"{ART}/models/{CAT_IMP}_f{k}.cbm" for k in range(5)]


def with_te(df_rows: pl.DataFrame, X_rows: pd.DataFrame, k: int, feats: list) -> pd.DataFrame:
    # tempel tabel TE fold k (+ residual TE bila model memakainya) lalu urutkan kolom sesuai model
    Xk = apply_te_tables(df_rows, X_rows, f"{ART}/te/te_f{k}.parquet")
    if any(f.startswith("rte_") for f in feats):
        Xk = apply_te_tables(df_rows, Xk, f"{ART}/te/rte_f{k}.parquet", RESID_KEYS)
    return Xk[feats]
imp = None
if all(os.path.exists(f) for f in lgb_files):
    boosters = [lgb.Booster(model_file=f) for f in lgb_files]
    imp = pd.DataFrame({"feat": boosters[0].feature_name(),
                        "gain": np.mean([b.feature_importance("gain") for b in boosters], 0)})
    imp_src = "LightGBM (gain)"
elif all(os.path.exists(f) for f in cat_files):
    from catboost import CatBoostClassifier
    cms = [CatBoostClassifier().load_model(f) for f in cat_files]
    imp = pd.DataFrame({"feat": cms[0].feature_names_,
                        "gain": np.mean([m.get_feature_importance() for m in cms], 0)})
    imp_src = f"CatBoost {CAT_IMP} (PredictionValuesChange)"
if imp is not None:
    print("Sumber importance:", imp_src)
    imp["pct"] = imp.gain / imp.gain.sum() * 100
    imp = imp.sort_values("pct", ascending=False).reset_index(drop=True)
    pre = imp.feat.str.split("_").str[0]
    imp["grup"] = np.where(pre.isin(["te", "gc", "gk", "gh", "gr", "gp", "sc", "rt", "iv", "cust", "cour", "hub",
                                     "pkg", "promo", "fq", "x", "r", "rel", "t"]), pre, "shipments")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    imp.head(25).iloc[::-1].plot.barh(x="feat", y="pct", ax=ax[0], legend=False, color="#4C72B0")
    ax[0].set(title="Top-25 fitur (gain %)", xlabel="% gain", ylabel="")
    imp.groupby("grup").pct.sum().sort_values().plot.barh(ax=ax[1], color="#DD8452")
    ax[1].set(title="Total gain per grup fitur", xlabel="% gain", ylabel="")
    plt.tight_layout(); plt.show()
    noise = ["skor_prioritas_kirim", "indeks_rute_sekunder", "faktor_muatan_internal", "varian_cuaca_rute",
             "indeks_waktu_proses", "skor_kualitas_cek"]
    print("Rank kolom noise (dari", len(imp), "fitur):",
          {c: int(imp.index[imp.feat == c][0]) + 1 for c in noise if c in set(imp.feat)})
    print(f"Fitur dengan gain < 0,01%: {(imp.pct < 0.01).sum()}")
else:
    print("Model belum tersedia — jalankan MODE='train' terlebih dulu.")
""")
md("""
**Keputusan:** fitur tidak dipangkas. Kolom noise dan fitur ber-gain sangat kecil sudah jarang dipakai model
(dengan `feature_fraction=0.3` dan `min_data_in_leaf=400`, split pada noise hampir tidak memberi gain). Membuangnya
tidak memberi perubahan yang lebih besar dari variasi antar-fold, dan menambah risiko *selection bias* terhadap CV.
Grup fitur paling bernilai: **target encoding** dan **statistik grup kurir/customer**, sesuai temuan EDA 2.2.
""")

# ===================================================================================== 9
md("""
## 9. Model & Training
Tiga GBDT dengan parameter yang dipilih dari eksperimen Section 7 (regularisasi kuat + early stopping 200 ronde
di fold validasi). Tanpa `class_weight` saat training: proba tetap terkalibrasi (baik untuk blending), lalu
Macro F1 dioptimasi lewat bobot kelas pasca-training (Section 11).

| Model | Catatan |
|---|---|
| LightGBM | `lr=0.03`, 63 leaves, `min_data_in_leaf=400`, `feature_fraction=0.3`, λ₂=10 |
| XGBoost | `hist`, depth 8, `colsample=0.4`, λ=10 (GPU opsional) |
| CatBoost | depth 8, `lr=0.08`, kategori native (*ordered target statistics*), GPU opsional |
""")
code_raw(grab("common.py", "LGB_PARAMS", "XGB_PARAMS", "CAT_PARAMS", "GPU_SWITCH", "fit_predict", "load_predict", "run_cv"))
code("""
if MODE == "train":
    key_cols = {c for cols, _ in TE_KEYS.values() for c in cols} | {c for cols, _ in RESID_KEYS.values() for c in cols}
    df_keys = df.select(sorted(key_cols))
    rcodes = te_codes(df, RESID_KEYS)
    np.save(f"{ART}/folds.npy", folds)
    if any(r for _, _, r, _ in MODEL_RUNS):  # tahap A residual TE (Section 6.1)
        np.save(f"{ART}/stageA_oof.npy", stage_a_oof(X, y_all, tr_all, folds, df))
    for model, seed, resid, override in MODEL_RUNS:
        print(f"===== {model} seed {seed} | residual TE={resid} | {override} =====")
        P = {"cat": CAT_PARAMS, "xgb": XGB_PARAMS, "lgb": LGB_PARAMS}[model]
        backup = dict(P); P.update(override)
        R_all = resid_matrix(y_all, tr_all, np.load(f"{ART}/stageA_oof.npy")) if resid else None
        run_cv(model, seed, df_keys, X, y_all, tr_all, te_all, codes, folds, ART,
               rcodes=rcodes if resid else None, R_all=R_all)
        P.clear(); P.update(backup)
else:
    print("MODE='inference' → training dilewati; memuat hasil CV tersimpan.")

runs = [(m, s) for m, s, *_ in MODEL_RUNS if os.path.exists(f"{ART}/oof_{m}_s{s}.npy")]
oofs = {f"{m}_s{s}": np.load(f"{ART}/oof_{m}_s{s}.npy") for m, s in runs}
cv_rows = []
for name, o in oofs.items():
    per_fold = [macro_f1(y_tr[folds == k], o[folds == k]) for k in range(5)]
    cv_rows.append(dict(model=name, logloss=round(log_loss(y_tr, o), 4), macro_f1=round(macro_f1(y_tr, o), 4),
                        f1_fold_std=round(np.std(per_fold), 4), **{f"f{k}": round(v, 4) for k, v in enumerate(per_fold)}))
pd.DataFrame(cv_rows)
""")

# ===================================================================================== 10
md("""
## 10. Ensemble
Rata-rata **geometrik** proba (rata-rata log-proba lalu softmax) dengan bobot model dicari via minimisasi logloss
OOF: hanya *n_model − 1* parameter bebas, jadi risiko overfit minimal.
""")
code_raw(grab("common.py", "blend", "fit_blend_weights"))
code("""
names = list(oofs)
bw = fit_blend_weights([oofs[n] for n in names], y_tr)
oof_b = blend([oofs[n] for n in names], bw)
print("Bobot blend:", dict(zip(names, bw.round(3))))
print(f"Ensemble OOF: logloss={log_loss(y_tr, oof_b):.4f} | Macro F1 (argmax)={macro_f1(y_tr, oof_b):.4f}")
""")

# ===================================================================================== 11
md("""
## 11. Optimasi Macro F1 (bobot per kelas)
Argmax proba memaksimalkan *akurasi*, bukan Macro F1. Kita kalikan proba dengan bobot $w_k$ lalu argmax;
$w$ dicari dengan *coordinate ascent* pada grid log.

**Anti-overfit:** bobot dievaluasi secara **nested**: dicari di 4 fold OOF, diuji di fold ke-5 yang tidak ikut dipakai
mencarinya. Bobot final dicari di seluruh OOF (7 parameter dari 1,24 juta baris: risiko overfit sangat kecil).
""")
code_raw(grab("common.py", "optimize_class_weights"))
code("""
nested = []
for k in range(5):
    w_k = optimize_class_weights(oof_b[folds != k], y_tr[folds != k])
    nested.append(dict(fold=k, f1_argmax=macro_f1(y_tr[folds == k], oof_b[folds == k]),
                       f1_weighted=macro_f1(y_tr[folds == k], oof_b[folds == k], w_k), **dict(zip(CLASSES, w_k.round(2)))))
nested = pd.DataFrame(nested)
display(nested.round(4))
print(f"NESTED Macro F1: argmax {nested.f1_argmax.mean():.4f} ± {nested.f1_argmax.std():.4f} → "
      f"berbobot {nested.f1_weighted.mean():.4f} ± {nested.f1_weighted.std():.4f}")
CLASS_W = optimize_class_weights(oof_b, y_tr)
print("Bobot kelas final:", dict(zip(CLASSES, CLASS_W.round(3))))
print(f"OOF Macro F1 dengan bobot final: {macro_f1(y_tr, oof_b, CLASS_W):.4f}")
""")
md("""
**Cek pergeseran bulan (Section 2.1).** Apakah bobot kelas perlu disetel dengan baris OOF yang diberi *importance
weight* P_test(bulan)/P_train(bulan)? Bobot ini hanya memakai fitur test, bukan label. Diuji secara nested dan
dievaluasi pada distribusi bulan ala test.
""")
code_raw(grab("common.py", "month_shift_weights"))
code("""
SW = month_shift_weights(df["ship_month"].to_numpy()[tr_all], df["ship_month"].to_numpy()[te_all])
rows = []
for k in range(5):
    a, b = folds != k, folds == k
    w_plain = optimize_class_weights(oof_b[a], y_tr[a])
    w_month = optimize_class_weights(oof_b[a], y_tr[a], sw=SW[a])
    rows.append(dict(fold=k, bobot_biasa=macro_f1(y_tr[b], oof_b[b], w_plain, SW[b]),
                     bobot_bulan=macro_f1(y_tr[b], oof_b[b], w_month, SW[b])))
shift = pd.DataFrame(rows)
display(shift.round(4))
print(f"Selisih rata-rata (bulan − biasa): {(shift.bobot_bulan - shift.bobot_biasa).mean():+.4f} "
      f"→ jauh di bawah std antar fold; dipakai bobot biasa (lebih sederhana)")
""")
md("""
### 11.1 Matriks keputusan penuh
Bobot diagonal hanya menggeser skor tiap kelas secara independen. Untuk metrik *fractional-linear* seperti Macro F1,
klasifikasi optimal Bayes berbentuk $\\arg\\max_j (P\\,W)_j$ dengan $W$ matriks **penuh**
([Narasimhan et al., ICML 2015](https://proceedings.mlr.press/v37/narasimhanb15.html)). Elemen $W_{ij}$ (i≠j)
mengizinkan proba kelas *i* ikut mendorong/menahan kelas *j* (mis. kelas yang sering tertukar).
$W$ dimulai dari `diag(CLASS_W)` lalu disempurnakan dengan *coordinate ascent*. Divalidasi **nested** seperti di atas.
""")
code_raw(grab("common.py", "f1_matrix", "optimize_decision_matrix"))
code("""
from joblib import Parallel, delayed

def _nested_matrix(k):
    a, b = folds != k, folds == k
    w_k = optimize_class_weights(oof_b[a], y_tr[a])
    W_k = optimize_decision_matrix(oof_b[a], y_tr[a], w_k)
    return dict(fold=k, diagonal=macro_f1(y_tr[b], oof_b[b], w_k), matriks=f1_matrix(y_tr[b], oof_b[b], W_k))

nested_m = pd.DataFrame(Parallel(n_jobs=5)(delayed(_nested_matrix)(k) for k in range(5)))
nested_m["selisih"] = nested_m.matriks - nested_m.diagonal
display(nested_m.round(4))
print(f"NESTED Macro F1: diagonal {nested_m.diagonal.mean():.4f} → matriks {nested_m.matriks.mean():.4f} "
      f"({nested_m.selisih.mean():+.4f} ± {nested_m.selisih.std():.4f}, naik di {(nested_m.selisih > 0).sum()}/5 fold)")
DECISION_W = optimize_decision_matrix(oof_b, y_tr, CLASS_W)
print(f"OOF Macro F1 dengan matriks final: {f1_matrix(y_tr, oof_b, DECISION_W):.4f}")
display(pd.DataFrame(DECISION_W, index=CLASSES, columns=CLASSES).round(3))
""")
code("""
pred_oof = (oof_b @ DECISION_W).argmax(1)
cmp = pd.DataFrame({"asli": np.bincount(y_tr, minlength=K) / len(y_tr),
                    "prediksi argmax": np.bincount(oof_b.argmax(1), minlength=K) / len(y_tr),
                    "prediksi matriks keputusan": np.bincount(pred_oof, minlength=K) / len(y_tr)}, index=CLASSES)
ax = (cmp * 100).plot.bar(figsize=(10, 3.8), rot=20, color=["#999999", "#4C72B0", "#55A868"])
ax.set(title="Distribusi kelas: asli vs prediksi OOF", ylabel="%"); plt.tight_layout(); plt.show()
""")
md("Bobot kelas menggeser prediksi dari `normal` ke kelas minoritas, mendekati distribusi asli. Inilah yang diinginkan Macro F1.")

# ===================================================================================== 12
md("## 12. Analisis Error")
code("""
print(classification_report(y_tr, pred_oof, target_names=CLASSES, digits=4))
cm = confusion_matrix(y_tr, pred_oof, normalize="true")
fig, ax = plt.subplots(figsize=(8.5, 6.5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=False)
ax.set(title="Confusion matrix OOF (normalisasi per baris = recall)", xlabel="prediksi", ylabel="asli")
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()
""")
code("""
# Kontribusi fitur (TreeSHAP bawaan LightGBM / CatBoost) untuk kelas-kelas tersulit, sampel baris validasi fold 0
contrib = None
if all(os.path.exists(f) for f in lgb_files):
    va0 = tr_all[folds == 0][:4000]
    Xs = apply_te_tables(df[va0], X.iloc[va0], f"{ART}/te/te_f0.parquet")[boosters[0].feature_name()]
    contrib = boosters[0].predict(Xs, pred_contrib=True).reshape(len(Xs), K, -1)[:, :, :-1]
elif all(os.path.exists(f) for f in cat_files):
    from catboost import Pool
    va0 = tr_all[folds == 0][:2000]
    Xs = with_te(df[va0], X.iloc[va0], 0, cms[0].feature_names_)
    cats = [c for c in Xs.columns if str(Xs[c].dtype) == "category"]
    pool = Pool(Xs.assign(**{c: Xs[c].astype(str) for c in cats}), cat_features=cats)
    contrib = cms[0].get_feature_importance(pool, type="ShapValues")[:, :, :-1]  # (n, K, fitur)
if contrib is not None:
    focus = ["gagal_antar_awal", "retur_pengirim", "rusak_perjalanan", "klaim_hilang"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, c in zip(axes, focus):
        s = pd.Series(np.abs(contrib[:, CLS_IDX[c]]).mean(0), index=Xs.columns).nlargest(10)[::-1]
        s.plot.barh(ax=ax, color="#C44E52"); ax.set_title(c); ax.set_xlabel("mean |kontribusi|")
    plt.tight_layout(); plt.show()
""")
md("""
**Catatan:** TE pada baris validasi fold 0 di sel ini memakai tabel TE fold 0 (dari bagian train fold 0),
sama seperti saat validasi, jadi tidak ada kebocoran.
""")

# ===================================================================================== 13
md("""
## 13. Simpan Model & Artefak
| File | Isi |
|---|---|
| `artifacts/models/lgb_s0_f{k}.txt`, `cat_s0_f{k}.cbm`, `xgb_s0_f{k}.json` | model per fold (5 × 3) |
| `artifacts/te/te_f{k}_{kunci}.parquet` | tabel target encoding per fold |
| `artifacts/folds.npy` | pembagian fold |
| `artifacts/oof_*.npy`, `test_*.npy` | proba OOF & test per model (untuk blending & cek reproducibility) |
| `artifacts/ensemble.json` | daftar model, bobot blend, bobot kelas, matriks keputusan, skor CV |
""")
code("""
ensemble_cfg = dict(models=names, blend_weights=bw.tolist(), class_weights=CLASS_W.tolist(),
                    decision_matrix=DECISION_W.tolist(), classes=CLASSES,
                    oof_logloss=float(log_loss(y_tr, oof_b)), oof_macro_f1=float(f1_matrix(y_tr, oof_b, DECISION_W)),
                    nested_macro_f1_diag=float(nested.f1_weighted.mean()),
                    nested_macro_f1=float(nested_m.matriks.mean()), nested_std=float(nested_m.matriks.std()))
json.dump(ensemble_cfg, open(f"{ART}/ensemble.json", "w"), indent=1)  # deterministik dari OOF tersimpan
ensemble_cfg = json.load(open(f"{ART}/ensemble.json"))                 # inference di bawah hanya membaca file ini
sizes = {}
for root, _, files in os.walk(ART):
    for f in files:
        sizes[os.path.relpath(os.path.join(root, f), ART)] = os.path.getsize(os.path.join(root, f)) / 1e6
print(f"{len(sizes)} file artefak, total {sum(sizes.values()):.0f} MB")
print(json.dumps({k: v for k, v in ensemble_cfg.items() if k != "classes"}, indent=1))
""")

# ===================================================================================== 14
md("""
## 14. Inferensi Tanpa Retrain → `submission.csv`
Bagian ini **hanya memuat file**: model per fold + tabel TE per fold + konfigurasi ensemble. Pipeline fitur test
adalah fungsi `build()` yang sama dengan train. Alur per model: untuk tiap fold *k*, tempel TE fold *k* → muat model
fold *k* → proba, lalu rata-rata 5 fold → blend geometrik → × matriks keputusan W → argmax.
""")
code("""
def predict_test(df: pl.DataFrame, X: pd.DataFrame, te_all: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    df_te, X_te = df[te_all], X.iloc[te_all]
    probs = {}
    for name in cfg["models"]:
        model, seed = name.rsplit("_s", 1)
        ff = f"{ART}/features_{name}.json"  # urutan fitur per model (model residual punya kolom rte_*)
        features = json.load(open(ff if os.path.exists(ff) else f"{ART}/features.json"))
        p = np.zeros((len(te_all), K))
        for k in range(5):
            Xk = with_te(df_te, X_te, k, features)
            p += load_predict(model, f"{ART}/models/{model}_s{seed}_f{k}", Xk) / 5
        probs[name] = p
        print(f"✓ {name}: 5 fold diprediksi")
    pb = blend([probs[n] for n in cfg["models"]], cfg["blend_weights"])
    return (pb @ np.array(cfg["decision_matrix"], dtype=np.float32)).argmax(1), probs


t0 = time.time()
pred_test, test_probs = predict_test(df, X, te_all, ensemble_cfg)
sub = pd.DataFrame({"shipment_id": df["shipment_id"].to_numpy()[te_all],
                    "status_penanganan": np.array(CLASSES)[pred_test]})
sub.to_csv(SUBMISSION_PATH, index=False)
print(f"inference {time.time() - t0:.0f}s → {SUBMISSION_PATH}")
""")
code("""
# Validasi format submission
ss = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
assert len(sub) == len(ss) == 362_245, "jumlah baris salah"
assert (sub.shipment_id.values == ss.shipment_id.values).all(), "urutan shipment_id beda dari sample_submission"
assert sub.status_penanganan.isin(CLASSES).all() and sub.shipment_id.is_unique
print("✓ format submission valid")

# Reproducibility: proba hasil inference == proba test yang disimpan saat training
for name, p in test_probs.items():
    ref = f"{ART}/test_{name}.npy"
    if os.path.exists(ref):
        print(f"{name}: max |diff| vs saat training = {np.abs(p - np.load(ref)).max():.2e}")
display(sub.status_penanganan.value_counts(normalize=True).round(4).to_frame("proporsi"))
sub.head()
""")

# ===================================================================================== 15
md("""
## 15. Ringkasan & Dependensi

### Hasil
Tabel CV per model (Section 9), ensemble (Section 10), dan **Macro F1 nested** (Section 11) adalah estimasi skor
private yang paling jujur. Seluruh keputusan diambil dari CV lokal, **bukan** public LB (hanya 30% test).

### 2 final pick
1. **Ensemble residual**: CatBoost lr 0,02 + CatBoost lr 0,04 + XGBoost, semuanya dengan residual TE customer,
   + matriks keputusan (CV terbaik, nested Macro F1 ±0,6294).
2. **Hedge tanpa residual TE**: CatBoost (3 varian) + XGBoost + matriks keputusan (nested ±0,6283), untuk berjaga
   seandainya residual TE berperilaku berbeda di data test.

| Tahap | Nested Macro F1 |
|---|---:|
| CatBoost lr 0,08 + bobot kelas diagonal | 0,6256 |
| + XGBoost (ensemble) | 0,6267 |
| + matriks keputusan penuh + CatBoost lr kecil | 0,6283 |
| **+ residual TE customer (final)** | **0,6294** |

LightGBM (`src/train.py lgb 0`) bisa ditambahkan ke `MODEL_RUNS`; notebook otomatis memakainya jika artefaknya ada.

### Anti-overfit checklist
- [x] Semua yang memakai target (TE, bobot blend, bobot kelas) dihitung di dalam fold / divalidasi nested
- [x] Statistik grup & frequency encoding tidak memakai label
- [x] Early stopping pada fold validasi; regularisasi kuat (min_data_in_leaf 400, λ=10, colsample 0,3)
- [x] Keputusan berdasarkan CV mean ± std; perubahan < std antar fold dianggap tidak ada
- [x] Adversarial validation memastikan train/test sebanding
- [x] Tidak ada tuning ke public LB

### Cara menjalankan
```bash
pip install -r requirements.txt
export KAGGLE_KEY=...; python data.py          # data → data/
# opsi A: notebook ini dengan MODE="inference" (butuh folder artifacts/)
# opsi B: latih ulang lewat script
python src/train.py lgb 0
USE_GPU=1 python src/train.py cat 0
USE_GPU=1 python src/train.py xgb 0
```

### Dependensi
Python 3.12 · numpy 2.5.5 · pandas 3.1.1 · polars 2.1.0 · pyarrow 26.0.1 · scipy 1.18.1 · scikit-learn 1.9.1 ·
lightgbm 4.7.0 · xgboost 3.4.1 · catboost 1.2.10 · matplotlib 3.11.2 · seaborn 0.13.2.
Tidak memakai model pre-trained / LLM eksternal. Semua model dilatih sendiri dari data kompetisi.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print(f"notebook ditulis: {OUT} ({len(cells)} sel)")
