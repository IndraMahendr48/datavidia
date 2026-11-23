"""Feature engineering pipeline — Datavidia 11.0 Task 1.

Semua fungsi di sini *deterministik* dan tidak memakai target, sehingga
pipeline yang sama dipakai untuk train maupun test (inference tanpa retrain).
Fitur berbasis target (target encoding) ada di `te.py` dan dihitung di dalam fold.
"""
from __future__ import annotations

import os
import polars as pl

CLASSES = [
    "normal", "konfirmasi_alamat", "tertunda_operasional", "gagal_antar_awal",
    "retur_pengirim", "rusak_perjalanan", "klaim_hilang",
]
TARGET = "status_penanganan"
EVENTS = ["pickup", "sortasi", "transit", "keluar_hub", "upaya_antar", "selesai"]
DEVICES = ["scanner", "pda", "gerbang_otomatis", "manual"]
MODAS = ["darat", "laut", "udara", "kereta"]
INV_TYPES = ["pengiriman", "langganan", "klaim"]


def read(data_dir: str, split: str, name: str) -> pl.DataFrame:
    return pl.read_csv(os.path.join(data_dir, split, f"{name}.csv"), infer_schema_length=20000)


# ---------------------------------------------------------------- lookups (1:1)
def prefix(df: pl.DataFrame, p: str, key: str) -> pl.DataFrame:
    return df.rename({c: f"{p}{c}" for c in df.columns if c != key})


def join_lookups(sh: pl.DataFrame, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    n = sh.height
    out = (
        sh.join(prefix(tables["customers"], "cust_", "customer_id"), on="customer_id", how="left", maintain_order="left")
        .join(prefix(tables["couriers"], "cour_", "courier_id"), on="courier_id", how="left", maintain_order="left")
        .join(prefix(tables["hubs"], "hub_", "hub_id"), on="hub_id", how="left", maintain_order="left")
        .join(prefix(tables["service_packages"], "pkg_", "package_id"), on="package_id", how="left", maintain_order="left")
        .join(prefix(tables["promotions"], "promo_", "promo_id"), on="promo_id", how="left", maintain_order="left")
    )
    assert out.height == n, "join lookup mengubah jumlah baris!"
    m = pl.col("ship_month")
    b, e = pl.col("promo_bulan_mulai"), pl.col("promo_bulan_akhir")
    out = out.with_columns(
        # relasi antar tabel
        (pl.col("cour_hub_id") == pl.col("hub_id")).cast(pl.Int8).alias("rel_cour_same_hub"),
        (pl.col("cust_region") == pl.col("dest_region")).cast(pl.Int8).alias("rel_cust_region_eq_dest"),
        (pl.col("hub_region") == pl.col("dest_region")).cast(pl.Int8).alias("rel_hub_region_eq_dest"),
        (pl.col("hub_region") == pl.col("cust_region")).cast(pl.Int8).alias("rel_hub_region_eq_cust"),
        (pl.col("kota_asal") == pl.col("kota_tujuan")).cast(pl.Int8).alias("rel_same_city"),
        pl.when(pl.col("promo_id") == "TANPA_PROMO").then(None)
        .when(b <= e).then(((m >= b) & (m <= e)).cast(pl.Int8))
        .otherwise(((m >= b) | (m <= e)).cast(pl.Int8)).alias("rel_promo_active"),
        # rasio / fisika paket
        (pl.col("declared_value") / pl.col("weight_kg").clip(0.01)).alias("r_value_per_kg"),
        (pl.col("berat_volumetrik") / pl.col("weight_kg").clip(0.01)).alias("r_volw_per_w"),
        (pl.col("volume_l") / pl.col("weight_kg").clip(0.01)).alias("r_vol_per_w"),
        (pl.max_horizontal("weight_kg", "berat_volumetrik") / pl.col("pkg_max_weight_kg")).alias("r_w_over_pkgmax"),
        (pl.max_horizontal("weight_kg", "berat_volumetrik") / pl.col("cour_kapasitas_muatan_kg")).alias("r_w_over_courcap"),
        (pl.col("distance_km") / pl.col("pkg_estimasi_hari").clip(1)).alias("r_km_per_estday"),
        (pl.col("declared_value") / pl.col("qty_items").clip(1)).alias("r_value_per_item"),
        # jam kirim vs jam operasional hub
        ((pl.col("ship_hour") < pl.col("hub_jam_buka")) | (pl.col("ship_hour") >= pl.col("hub_jam_tutup"))).cast(pl.Int8).alias("t_outside_hub_hours"),
        (pl.col("hub_jam_tutup") - pl.col("hub_jam_buka")).alias("t_hub_open_len"),
        (pl.col("hub_jam_tutup") - pl.col("ship_hour")).alias("t_hours_to_close"),
        (pl.col("hub_density_index") / pl.col("hub_capacity_index").clip(0.01)).alias("r_hub_load"),
        (pl.col("fragility_flag") * pl.col("pkg_fragile_score")).alias("x_fragile"),
        (pl.col("flag_alamat_baru") + pl.col("jumlah_catatan")).alias("x_addr_risk"),
        (pl.col("kota_asal") + "_" + pl.col("kota_tujuan")).alias("route_pair"),
    )
    return out


# ---------------------------------------------------------------- scan_events
def agg_scan(sc: pl.DataFrame) -> pl.DataFrame:
    sc = sc.with_columns(pl.int_range(pl.len()).over("shipment_id").alias("pos"))
    et, g, q = pl.col("event_type"), pl.col("gap_hours"), pl.col("scan_quality")
    aggs = [
        pl.len().alias("sc_n"),
        g.sum().alias("sc_gap_sum"), g.mean().alias("sc_gap_mean"), g.max().alias("sc_gap_max"),
        g.min().alias("sc_gap_min"), g.std().alias("sc_gap_std"),
        q.mean().alias("sc_q_mean"), q.min().alias("sc_q_min"), q.max().alias("sc_q_max"), q.std().alias("sc_q_std"),
        (q <= 2).sum().alias("sc_q_low_n"), (q == 1).sum().alias("sc_q1_n"),
        pl.col("is_manual").sum().alias("sc_manual_n"), pl.col("is_manual").mean().alias("sc_manual_rate"),
        et.last().alias("sc_last_event"),
        g.last().alias("sc_last_gap"), g.first().alias("sc_first_gap"),
        q.last().alias("sc_last_q"), q.first().alias("sc_first_q"),
        pl.col("pos").filter(et == "upaya_antar").min().alias("sc_first_upaya_pos"),
        et.n_unique().alias("sc_n_unique_event"),
        (et != et.shift(1)).sum().alias("sc_n_switch"),
    ]
    for e in EVENTS:
        m = et == e
        aggs += [
            m.sum().alias(f"sc_n_{e}"),
            g.filter(m).sum().alias(f"sc_gap_sum_{e}"),
            g.filter(m).max().alias(f"sc_gap_max_{e}"),
            q.filter(m).mean().alias(f"sc_q_mean_{e}"),
            pl.col("is_manual").filter(m).sum().alias(f"sc_manual_{e}"),
        ]
    for d in DEVICES:
        m = pl.col("device_type") == d
        aggs += [m.sum().alias(f"sc_dev_{d}"), q.filter(m).mean().alias(f"sc_q_dev_{d}")]
    out = sc.group_by("shipment_id").agg(aggs)
    out = out.with_columns(
        (pl.col("sc_n_selesai") > 0).cast(pl.Int8).alias("sc_has_selesai"),
        (pl.col("sc_n_upaya_antar") / pl.col("sc_n")).alias("sc_upaya_frac"),
        (pl.col("sc_gap_max") / pl.col("sc_gap_mean").clip(0.01)).alias("sc_gap_peak_ratio"),
        (pl.col("sc_n_upaya_antar") - pl.col("sc_n_selesai")).alias("sc_upaya_minus_selesai"),
        (pl.col("sc_n_sortasi") + pl.col("sc_n_transit") + pl.col("sc_n_keluar_hub")).alias("sc_n_mid"),
    )
    return out


# ---------------------------------------------------------------- route_segments
def agg_route(rs: pl.DataFrame) -> pl.DataFrame:
    rs = rs.sort(["shipment_id", "urutan"])
    d, t, c, j = pl.col("durasi_jam"), pl.col("biaya_segmen"), pl.col("moda"), pl.col("jarak_segmen_km")
    aggs = [
        pl.len().alias("rt_n"),
        j.sum().alias("rt_km_sum"), j.max().alias("rt_km_max"),
        d.sum().alias("rt_dur_sum"), d.max().alias("rt_dur_max"), d.mean().alias("rt_dur_mean"),
        d.null_count().alias("rt_dur_null"),
        t.sum().alias("rt_cost_sum"), t.max().alias("rt_cost_max"),
        (j / d.clip(0.01)).mean().alias("rt_speed_mean"), (j / d.clip(0.01)).min().alias("rt_speed_min"),
        (t / j.clip(0.1)).mean().alias("rt_cost_per_km"),
        pl.col("segmen_hub_id").n_unique().alias("rt_n_hub"),
        c.first().alias("rt_first_moda"), c.last().alias("rt_last_moda"),
        c.n_unique().alias("rt_n_moda"),
        pl.col("segmen_hub_id").first().alias("rt_first_hub"),
        pl.col("segmen_hub_id").last().alias("rt_last_hub"),
    ]
    for m in MODAS:
        f = c == m
        aggs += [f.sum().alias(f"rt_n_{m}"), d.filter(f).sum().alias(f"rt_dur_{m}"), j.filter(f).sum().alias(f"rt_km_{m}")]
    out = rs.group_by("shipment_id").agg(aggs)
    return out.with_columns((pl.col("rt_km_sum") / pl.col("rt_dur_sum").clip(0.01)).alias("rt_speed_total"))


# ---------------------------------------------------------------- invoices
def agg_invoice(inv: pl.DataFrame) -> pl.DataFrame:
    it, a = pl.col("invoice_type"), pl.col("amount")
    aggs = [
        pl.len().alias("iv_n"), a.sum().alias("iv_amount_sum"), a.max().alias("iv_amount_max"),
        pl.col("discount").sum().alias("iv_disc_sum"), pl.col("discount").null_count().alias("iv_disc_null"),
        pl.col("tax").sum().alias("iv_tax_sum"),
        pl.col("days_to_pay").mean().alias("iv_dtp_mean"), pl.col("days_to_pay").max().alias("iv_dtp_max"),
        pl.col("is_penalty").sum().alias("iv_penalty"),
        pl.col("issued_month").min().alias("iv_month_min"), pl.col("issued_month").max().alias("iv_month_max"),
        pl.col("payment_method").first().alias("iv_pay_method"),
        pl.col("payment_status").first().alias("iv_pay_status_first"),
    ]
    for ty in INV_TYPES:
        f = it == ty
        aggs += [
            f.sum().alias(f"iv_n_{ty}"), a.filter(f).sum().alias(f"iv_amt_{ty}"),
            pl.col("days_to_pay").filter(f).max().alias(f"iv_dtp_{ty}"),
            pl.col("discount").filter(f).sum().alias(f"iv_disc_{ty}"),
        ]
    for st in ["lunas", "tertunda", "gagal"]:
        aggs.append((pl.col("payment_status") == st).sum().alias(f"iv_st_{st}"))
    return inv.group_by("shipment_id").agg(aggs)


# ---------------------------------------------------------------- full build
CAT_COLS = [
    "hub_id", "package_id", "promo_id", "channel", "pickup_type", "goods_category", "dest_region",
    "kota_asal", "kota_tujuan", "cust_region", "cust_segment", "cust_preferred_payment", "cust_loyalty_tier",
    "cust_kota_tipe", "cust_metode_notifikasi", "cour_vehicle_type", "cour_contract_type", "cour_shift_code",
    "hub_region", "hub_hub_type", "pkg_speed_class", "pkg_handling_class", "promo_nama_promo", "promo_tipe_promo",
    "sc_last_event", "rt_first_moda", "rt_last_moda", "iv_pay_method", "iv_pay_status_first",
]
# ID kardinalitas tinggi → tidak dijadikan fitur langsung, hanya via frequency / target encoding
ID_COLS = ["shipment_id", "customer_id", "courier_id", "cour_hub_id", "route_pair", "rt_first_hub", "rt_last_hub"]


FEAT_FILE = "features_v2.parquet"


def build(data_dir: str, cache_dir: str | None = None, tables: dict | None = None) -> pl.DataFrame:
    """Bangun tabel fitur gabungan train+test (kolom `is_test`). Tanpa target leakage.

    `tables` (opsional): {split: {nama_tabel: DataFrame}} yang sudah di-load; kalau None dibaca dari CSV.
    """
    if cache_dir and os.path.exists(os.path.join(cache_dir, FEAT_FILE)):
        return pl.read_parquet(os.path.join(cache_dir, FEAT_FILE))
    get = (lambda sp, nm: tables[sp][nm]) if tables else (lambda sp, nm: read(data_dir, sp, nm))
    lookups = {k: get("train", k) for k in ["customers", "couriers", "hubs", "service_packages", "promotions"]}
    parts = []
    for split in ["train", "test"]:
        sh = get(split, "shipments")
        df = join_lookups(sh, lookups)
        for fn, name in [(agg_scan, "scan_events"), (agg_route, "route_segments"), (agg_invoice, "invoices")]:
            a = fn(get(split, name))
            assert a["shipment_id"].n_unique() == a.height
            df = df.join(a, on="shipment_id", how="left", maintain_order="left")
        assert df.height == sh.height
        df = df.with_columns(pl.lit(split == "test").alias("is_test"))
        if TARGET not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(TARGET))
        parts.append(df)
    df = pl.concat(parts, how="diagonal_relaxed")

    # fitur turunan lintas tabel
    df = df.with_columns(
        (pl.col("rt_dur_sum") / (pl.col("pkg_estimasi_hari") * 24)).alias("x_dur_vs_est"),
        (pl.col("sc_gap_sum") / (pl.col("pkg_estimasi_hari") * 24)).alias("x_gap_vs_est"),
        (pl.col("rt_km_sum") - pl.col("distance_km")).alias("x_km_diff"),
        (pl.col("iv_amt_klaim") / pl.col("declared_value").clip(1)).alias("x_claim_over_value"),
        (pl.col("rt_cost_sum") / pl.col("declared_value").clip(1)).alias("x_cost_over_value"),
        (pl.col("iv_month_min") - pl.col("ship_month")).alias("x_inv_month_lag"),
        (pl.col("rt_last_hub") != pl.col("hub_id")).cast(pl.Int8).alias("x_last_hub_diff"),
    )

    # frequency encoding (train+test, tanpa target → aman)
    for col in ["customer_id", "courier_id", "hub_id", "kota_asal", "kota_tujuan", "route_pair", "package_id",
                "rt_last_hub"]:
        df = df.with_columns(pl.len().over(col).alias(f"fq_{col}"))

    # statistik grup tanpa target: perilaku customer / kurir / hub rata-rata
    grp_feats = ["sc_n_upaya_antar", "sc_gap_mean", "sc_q_mean", "sc_manual_rate", "declared_value",
                 "jumlah_catatan", "flag_alamat_baru", "iv_n_klaim", "rt_dur_sum", "is_cod", "distance_km",
                 "sc_gap_max", "sc_q_min", "sc_q_low_n", "sc_n", "sc_n_selesai", "iv_dtp_mean", "iv_amt_klaim",
                 "rt_n", "rt_n_udara", "rt_n_laut", "rt_speed_total", "iv_st_tertunda", "x_dur_vs_est",
                 "weight_kg", "ship_month"]
    for key, p in [("customer_id", "gc"), ("courier_id", "gk"), ("hub_id", "gh"), ("route_pair", "gr"),
                   ("package_id", "gp")]:
        exprs = [pl.col("sc_gap_mean").std().over(key).alias(f"{p}_sc_gap_mean_std"),
                 pl.col("sc_q_mean").std().over(key).alias(f"{p}_sc_q_mean_std")]
        for f in grp_feats:
            mean = pl.col(f).mean().over(key)
            exprs += [mean.alias(f"{p}_{f}_mean"), (pl.col(f) - mean).alias(f"{p}_{f}_dev")]
        df = df.with_columns(exprs)

    # kategori
    df = df.with_columns([pl.col(c).cast(pl.String).fill_null("NA") for c in CAT_COLS])
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        df.write_parquet(os.path.join(cache_dir, FEAT_FILE))
    return df
