# Datavidia 11.0 — Task 1: Klasifikasi Status Penanganan Pengiriman

> Ringkasan kompetisi + EDA + rekomendasi model.
> Semua angka di sini dihitung langsung dari data kompetisi (`eda_full.py`, `quick_lgb.py` di `~/kaggle/task1-dv11/`).

---

## 1. Deskripsi Tugas

**Cerita:** Rian (QC di ekspedisi antarpulau) punya masalah: status penanganan tiap paket diisi **manual** oleh petugas → sering telat, lupa, asal pilih. Dia mau model yang membaca *jejak* paket (scan, rute, tagihan, dll.) lalu menandai status penanganannya otomatis.

**Tugas:** **Klasifikasi multikelas (7 kelas).** Untuk setiap `shipment_id` di `test/shipments.csv`, prediksi satu label `status_penanganan`.

| Kelas | Arti |
|---|---|
| `normal` | Selesai tanpa insiden |
| `konfirmasi_alamat` | Butuh konfirmasi / perbaikan alamat |
| `tertunda_operasional` | Tertunda karena hub numpuk / transit telat |
| `gagal_antar_awal` | Gagal diantar di percobaan pertama |
| `retur_pengirim` | Dikembalikan ke pengirim |
| `rusak_perjalanan` | Rusak selama perjalanan |
| `klaim_hilang` | Hilang → klaim |

**Metrik: Macro F1**

$$Macro\ F1 = \frac{1}{7}\sum_{k=1}^{7} F1_k$$

Artinya **ke-7 kelas bobotnya sama**. Kelas langka (`klaim_hilang` 3%) sama pentingnya dengan `normal` (38%). Jadi model yang cuma jago di `normal` bakal jeblok.

**Format submission:**
```
shipment_id,status_penanganan
SHP-5A64RENJ,normal
...
```
362.245 baris.

---

## 2. Aturan Penting (wajib dibaca tim)

- Tim **3 orang**, satu kampus.
- **Maks 3 submission/hari**, pilih **2 final**. Skor private = yang tertinggi dari 2 pilihan itu.
- Public LB = **30%** test (acak), private = **70%**. Jadi **percayai CV lokal, jangan overfit ke public LB**.
- **Prediksi wajib dari model.** Dilarang label manual, hardcoding, pakai ground truth.
- **LLM / AutoML / GenAI diperbolehkan dipakai untuk menghasilkan prediksi.** serta TabPFN dkk boleh asal bobot publik & dijalankan sendiri (cantumkan sumber/versi).
- Tim harus **mengembangkan & menjalankan sendiri** training + inference.
- Wajib serahkan **notebook** (pemrosesan, join tabel, fitur, model, training, prediksi) + **file model** (`.pkl/.joblib/.txt/.cbm/...`). Notebook harus bisa **inference tanpa retrain**.


---

## 3. Struktur Data (9 tabel relasional)

```
                  customers (customer_id)
                        │
service_packages ── shipments ── couriers (courier_id) ── hubs
  (package_id)     (shipment_id)        │
promotions ────────┘   │  │  │          hub_id ──────────┘
  (promo_id)           │  │  └── invoices        (1:N, 1–2 per paket)
                       │  └───── route_segments  (1:N, 1–3 kaki)
                       └──────── scan_events     (1:N, 2–18 scan)
```

| Tabel | Train | Test | Key | Isi |
|---|---|---|---|---|
| `shipments` | 1.243.617 × 38 | 362.245 × 37 | `shipment_id` | Tabel utama + **target** |
| `scan_events` | 6.837.111 | 1.993.132 | `shipment_id` | Log scan: `event_type`, `gap_hours`, `device_type`, `scan_quality`, `is_manual` |
| `route_segments` | 2.200.148 | 641.446 | `shipment_id` | Kaki rute: `moda`, `jarak_segmen_km`, `durasi_jam`, `biaya_segmen` |
| `invoices` | 1.467.722 | 427.458 | `shipment_id` | Tagihan: `invoice_type` (pengiriman/langganan/**klaim**), `payment_status`, `days_to_pay`, `is_penalty` |
| `customers` | 573.841 | sama | `customer_id` | `address_quality`, `complaint_index`, `credit_score`, `segment`, dll. |
| `couriers` | 23.806 | sama | `courier_id` | `courier_score`, `experience_years`, `vehicle_type`, dll. |
| `hubs` | 654 | sama | `hub_id` | `automation_level`, `capacity_index`, `density_index`, jam buka/tutup |
| `service_packages` | 62 | sama | `package_id` | `speed_class`, `estimasi_hari`, `fragile_score` |
| `promotions` | 34 | sama | `promo_id` | Diskon & periode promo (`TANPA_PROMO` = tidak ada) |

Rata-rata per paket: **5,5 scan**, **1,77 kaki rute**, **1,18 invoice**. Semua paket punya minimal 1 baris di ketiga tabel 1:N.

---

## 4. Distribusi Target (train)

| Kelas | Jumlah | % |
|---|---:|---:|
| normal | 473.531 | 38,08 |
| konfirmasi_alamat | 272.911 | 21,94 |
| tertunda_operasional | 185.496 | 14,92 |
| gagal_antar_awal | 124.341 | 10,00 |
| retur_pengirim | 87.740 | 7,06 |
| rusak_perjalanan | 61.942 | 4,98 |
| klaim_hilang | 37.656 | 3,03 |

**Imbalanced** (rasio 12,6 : 1 antara terbanyak dan tersedikit). Untuk Macro F1, kita perlu `class_weight` / reweighting **dan** tuning threshold per kelas.

---

## 5. Kualitas Data

**Missing value (sedikit):**
- `shipments.volume_l`: 2,98% (test 3,04%). Proporsinya **rata di semua kelas**, jadi missing-nya acak dan nggak informatif.
- `invoices.discount`: 42,4% (kemungkinan artinya "tidak ada diskon", jadi bisa diisi 0 + flag)
- `route_segments.durasi_jam`: 2,0%
- `customers.credit_score`: 1,99%, `couriers.courier_score`: 1,08%

**Train vs test shift:** sangat kecil. KS statistic semua kolom numerik < 0,01, kecuali `ship_month` (0,029). Train dan test kelihatannya split acak, jadi **Stratified K-Fold** sudah cocok sebagai CV.

**Overlap ID test ↔ train:**
- `courier_id`, `hub_id`, `kota_asal/tujuan`: **100%** muncul di train
- `customer_id`: **88,6%** muncul di train (1 customer rata-rata ~2,4 paket). Ini membuka peluang fitur **riwayat customer** (target encoding OOF).

**Kolom mencurigakan (kemungkinan noise):** deskripsi lomba bilang *"tidak semua kolom benar-benar berguna"*. Enam kolom berikut distribusinya mirip N(50, σ), dan sinyal univariatnya ≈ 0 (η² ≤ 0,00001):
`skor_prioritas_kirim`, `indeks_rute_sekunder`, `faktor_muatan_internal`, `varian_cuaca_rute`, `indeks_waktu_proses`, `skor_kualitas_cek`.
Selain itu ada juga `weight_kg`, `volume_l`, `berat_volumetrik`, `ship_hour`, `ship_dow`, `qty_items`, `gift_wrap`, `coupon_count`, `fragility_flag`, `insurance_opt`, `is_promo` yang sinyal univariatnya juga ≈ 0. Cek pakai **null importance / permutation importance** sebelum dibuang, karena bisa jadi mereka berguna lewat interaksi.

---

## 6. Sinyal per Kelas (temuan EDA)

### 6.1 Dari `shipments`
Kekuatan sinyal univariat (η² numerik / Cramér's V kategorikal):

| Fitur | Sinyal | Catatan |
|---|---|---|
| `hub_id` | **V = 0,229** | Paling kuat. Hub tertentu sering bermasalah |
| `jumlah_catatan` | η² = 0,064 | konfirmasi_alamat rata-rata 0,77 vs normal 0,36 |
| `declared_value` | η² = 0,059 | **klaim_hilang rata-rata Rp411rb** vs normal Rp116rb (barang mahal → hilang) |
| `distance_km` | η² = 0,035 | **rusak_perjalanan 1.174 km** vs normal 842 km |
| `package_id` | V = 0,028 | |
| `flag_alamat_baru` | η² = 0,013 | konfirmasi_alamat 19,8% vs normal 10% |
| `ship_month` | η² = 0,006 | tertunda_operasional rata-rata bulan 7,4 (musim sibuk?) |
| `is_cod` | kecil | retur_pengirim 8,9% vs ~6,3% kelas lain |

### 6.2 Dari `scan_events` (rata-rata per paket)

| Kelas | `upaya_antar` | `selesai` | `gap_hours` | `scan_quality` | `is_manual` |
|---|---:|---:|---:|---:|---:|
| normal | 1,31 | 0,71 | 7,6 | **3,79** | 0,15 |
| konfirmasi_alamat | **1,89** | 0,58 | 8,2 | 3,65 | 0,18 |
| tertunda_operasional | 1,39 | 0,69 | **11,4** | 3,58 | 0,19 |
| gagal_antar_awal | **1,58** | 0,65 | 8,1 | 3,40 | 0,22 |
| retur_pengirim | 1,51 | 0,66 | 8,2 | 3,59 | 0,19 |
| rusak_perjalanan | 1,34 | 0,70 | 8,0 | **3,09** | **0,28** |
| klaim_hilang | 1,35 | 0,70 | 8,1 | 3,63 | 0,18 |

Polanya masuk akal:
- Banyak `upaya_antar` → masalah alamat / gagal antar.
- `gap_hours` besar → tertunda.
- `scan_quality` rendah + banyak scan manual → rusak.
- `device_type` **tidak** membedakan kelas (distribusinya identik).

### 6.3 Dari `route_segments`
- `rusak_perjalanan` punya **lebih banyak kaki rute** (2,02 vs ~1,75) dan jarak lebih jauh, jadi makin banyak transit makin rawan rusak.
- `tertunda_operasional` punya **durasi paling lama** (19,1 jam/segmen vs normal 12,7) dan lebih sering pakai **udara/laut**.

### 6.4 Dari `invoices`
- **Invoice tipe `klaim`**: retur_pengirim 0,37/paket, klaim_hilang 0,32, kelas lain ~0,15–0,16. Kuat, tapi tidak deterministik (normal pun ada klaim 16%).
- `amount` invoice klaim_hilang lebih besar (Rp15,2rb vs ~Rp10,5rb).
- `days_to_pay`: normal 3,7 hari, tertunda 5,3 hari.
- `is_penalty` dan `payment_status=gagal` sangat jarang tapi lebih banyak di tertunda_operasional.

---

## 7. Baseline Cepat (patokan, bukan submission)

Setup: LightGBM multiclass, `class_weight='balanced'`. Fiturnya: shipments + join 5 tabel master + agregasi sederhana 3 tabel 1:N (count per kategori, mean/max/min/sum numerik), total 160 fitur. Data 300rb sampel, holdout 20% (60rb).

| | Macro F1 |
|---|---:|
| argmax | **0,4747** |
| + tuning bobot per kelas | **0,4782** |

F1 per kelas (holdout):

| Kelas | Precision | Recall | F1 |
|---|---:|---:|---:|
| normal | 0,753 | 0,734 | **0,744** |
| konfirmasi_alamat | 0,618 | 0,591 | 0,604 |
| tertunda_operasional | 0,609 | 0,565 | 0,586 |
| klaim_hilang | 0,375 | 0,416 | 0,394 |
| rusak_perjalanan | 0,329 | 0,394 | 0,359 |
| gagal_antar_awal | 0,325 | 0,350 | 0,337 |
| retur_pengirim | 0,304 | 0,344 | **0,323** |

**Fitur terpenting (gain):** `hub_id` (25,8%), `cour_hub_id`, `package_id`, `kota_asal`, `kota_tujuan`, `cust_address_quality`, `cust_complaint_index`, `declared_value`, `courier_score`, `scan_quality_mean`, `hubs_density_index`, ...

**Konteks LB (per 23 Nov):** top ~0,602, top-20 ≥ 0,594. Artinya baseline sederhana ini masih **jauh (~0,12)** di bawah top. Masih ada sinyal besar yang belum ketangkap. Dugaanku (belum dites) sinyalnya ada di:
1. **Urutan scan** (event terakhir, ada `selesai` atau tidak, posisi `upaya_antar`, gap per tahap), bukan cuma hitungan.
2. **Riwayat customer / kurir / hub** lewat target encoding OOF (88,6% customer test ada di train).
3. Interaksi rute (hub di segmen vs hub asal, hub di scan vs hub kurir).
4. Fitur rasio: durasi aktual vs `estimasi_hari`, `declared_value` / berat, jam kirim vs jam buka hub.

Kelas yang paling menyeret skor: **retur_pengirim, gagal_antar_awal, rusak_perjalanan, klaim_hilang**. Fitur baru sebaiknya fokus membedakan 4 kelas ini.

---

## 8. Rekomendasi Model

### Pilihan utama: Gradient Boosting (GBDT) — LightGBM + CatBoost (+ XGBoost)
Kenapa:
- Data **tabular relasional** dengan banyak kategori dan numerik campur. GBDT hampir selalu juara di jenis data begini.
- Cepat untuk 1,2 juta baris, bisa handle missing value, dan bisa kasih feature importance (berguna untuk buang kolom noise).
- **CatBoost** bagus untuk kategori high-cardinality (`hub_id`, `courier_id`, kota) karena pakai ordered target statistics, jadi lebih tahan overfit.
- **LightGBM** paling cepat untuk iterasi fitur.
- Hasil akhir: **ensemble rata-rata probabilitas** dari 2–3 model, biasanya +0,005–0,015.

### Opsional
- **TabPFN v2**: boleh (bobot publik, jalankan sendiri, cantumkan versi), tapi hanya kuat di dataset kecil, jadi kurang praktis untuk 1,2 juta baris. Paling-paling dipakai di subset atau stacking.
- **MLP / TabNet**: biasanya kalah dari GBDT di data begini, tapi bisa jadi tambahan diversity buat ensemble kalau ada waktu.
- **Sequence model (GRU/Transformer) di scan_events**: overkill. Lebih efisien bikin fitur urutan manual, lalu masukkan ke GBDT.

### Trik khusus Macro F1 (penting!)
1. `class_weight='balanced'` atau `sample_weight ∝ 1/freq^α` (coba α 0,5–1).
2. **Tuning bobot/threshold per kelas di probabilitas OOF**: kalikan proba dengan w_k lalu argmax, cari w_k pakai coordinate ascent. Gratis +0,003 sampai +0,03.
3. Lakukan tuning ini di **OOF seluruh train** (bukan public LB) supaya tidak overfit.

---

## 9. Pipeline yang Disarankan

1. **Load & join** 5 tabel master ke `shipments` (left join by ID).
2. **Agregasi 1:N per `shipment_id`:**
   - `scan_events`: count per `event_type`; gap_hours sum/mean/max/std; scan_quality min/mean; % manual; **event terakhir**; ada `selesai`?; jumlah `upaya_antar` sebelum `selesai`; jumlah hub unik.
   - `route_segments`: jumlah kaki; moda unik / flag laut-udara; sum durasi & jarak; kecepatan (jarak/durasi); max durasi; hub segmen ≠ hub asal.
   - `invoices`: count per `invoice_type` (terutama **klaim**); sum amount klaim; days_to_pay max; is_penalty; payment_status.
3. **Encoding ID:** target encoding **out-of-fold** (7 kolom proba per kelas) + frequency encoding untuk `customer_id`, `courier_id`, `hub_id`, kota. ⚠️ Jangan jadikan `customer_id` kategori langsung di LightGBM (300rb+ level, lambat dan overfit).
4. **Fitur rasio & interaksi** (lihat poin 7).
5. **Seleksi fitur:** null importance / permutation untuk membuang kolom noise.
6. **CV:** StratifiedKFold 5-fold, simpan OOF + test proba tiap model.
7. **Model:** LightGBM, CatBoost, XGBoost, lalu ensemble proba.
8. **Optimasi Macro F1:** bobot per kelas di OOF, lalu argmax ke test.
9. **Simpan model** (`.txt` / `.cbm` / `.json`) dan notebook inference tanpa retrain.
10. **2 final pick:** ensemble CV terbaik + 1 hedge (misal model tunggal terbaik atau versi bobot lain).

---

## 10. Catatan Teknis

- Data besar (`scan_events` train 470 MB). Pakai `dtype` hemat (category/float32), atau polars. Agregasi dulu, baru join.
- pandas 3.x: kolom string bukan `object` lagi, jadi cek tipe pakai `pd.api.types.is_numeric_dtype`.
- Kalau pakai Kaggle Notebook, CPU 4 core + RAM 30 GB cukup untuk LightGBM. CatBoost bisa pakai GPU Kaggle.
