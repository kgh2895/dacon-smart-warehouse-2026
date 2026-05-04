"""
v8 — PB-style features + objective diversity + sample weighting.

Combines three improvements over the attack pipeline:
1. Rich PB-notebook-style feature engineering (~480 features)
2. Per-model objective/transform diversity (MAE raw, Huber log1p, MAE log1p)
3. Target-skew sample weighting (q90/q95/q99 bonus)

Strict-clean: test.csv used only for final inference.
"""

import argparse
import os
import pickle
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

import config as cfg
import run_experiments_v5 as v5
import run_experiments_strict as strict
import run_experiments_attack as attack

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V8_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_v8")
V8_SANITY_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_v8_sanity")


def log(msg):
    v5.log(msg)


# ── PB-style features ──────────────────────────────────────────────

def safe_divide(a, b):
    num = pd.Series(a, dtype="float64")
    den = pd.Series(b, dtype="float64").replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def add_onset_features(df, value_col, prefix, grp_key):
    """Track when an event (charging, queueing) first occurs in scenario."""
    if value_col not in df.columns:
        return []
    positive = df[value_col].fillna(0).gt(0).astype(bool)
    t = df["timestep"].where(positive)
    first = t.groupby(grp_key).transform(lambda s: s.ffill().cummin())
    prev = positive.groupby(grp_key).shift(1, fill_value=False).astype(bool)

    df[f"{prefix}_ever_started"] = first.notna().astype(np.int8)
    df[f"{prefix}_start_idx"] = first.fillna(-1).astype(np.int16)
    df[f"{prefix}_started_now"] = (positive & ~prev).astype(np.int8)
    df[f"{prefix}_started_early"] = (first <= 5).fillna(False).astype(np.int8)
    df[f"{prefix}_steps_since"] = np.where(
        first.notna(), (df["timestep"] - first).astype(float), -1.0
    ).astype(np.float32)
    return [
        f"{prefix}_ever_started", f"{prefix}_start_idx",
        f"{prefix}_started_now", f"{prefix}_started_early",
        f"{prefix}_steps_since",
    ]


SEQ_COLS_EXTRA = [
    "order_inflow_15m", "unique_sku_15m", "robot_active", "robot_idle",
    "robot_charging", "battery_mean", "battery_std", "low_battery_ratio",
    "charge_queue_length", "avg_charge_wait", "congestion_score",
    "max_zone_density", "blocked_path_15m", "near_collision_15m",
    "fault_count_15m", "avg_recovery_time", "task_reassign_15m",
    "replenishment_overlap", "pack_utilization", "loading_dock_util",
    "staging_area_util", "label_print_queue",
]

BASELINE_EXPAND_COLS = [
    "avg_items_per_order", "urgent_order_ratio", "heavy_item_ratio",
    "cold_chain_ratio", "sku_concentration", "bulk_order_ratio",
    "avg_trip_distance", "network_latency_ms", "air_quality_idx",
    "barcode_read_success_rate", "hvac_power_kw", "ambient_noise_db",
    "inventory_turnover_rate", "safety_score_monthly", "scanner_error_rate",
    "wms_response_time_ms", "backorder_ratio",
]


def add_pb_style_features(train_fe, test_fe, feature_cols):
    """Add PB-notebook-style features on top of existing pipeline."""
    train_out = train_fe.copy()
    test_out = test_fe.copy()
    new_cols = []

    for df in (train_out, test_out):
        grp_key = df[cfg.GROUP_COL]
        grp = df.groupby(cfg.GROUP_COL, sort=False)

        # --- 1. Time phase features ---
        ts = df["timestep"]
        df["time_frac"] = (ts / 24.0).astype(np.float32)
        df["time_remaining"] = (24 - ts).astype(np.int16)
        df["time_idx_sq"] = (df["time_frac"] ** 2).astype(np.float32)
        df["is_early_phase"] = (ts <= 5).astype(np.int8)
        df["is_mid_phase"] = ((ts >= 6) & (ts <= 15)).astype(np.int8)
        df["is_late_phase"] = (ts >= 16).astype(np.int8)

        # --- 2. Missing indicators ---
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                        and c not in {cfg.ID_COL, cfg.GROUP_COL, "layout_id", cfg.TARGET, "_is_train"}]
        missing_cols = [c for c in numeric_cols if df[c].isna().any()]
        for c in missing_cols:
            col_name = f"{c}__is_missing"
            df[col_name] = df[c].isna().astype(np.int8)
            if col_name not in new_cols:
                new_cols.append(col_name)
        df["n_missing_raw"] = df[missing_cols].isna().sum(axis=1).astype(np.int16) if missing_cols else 0
        df["missing_ratio_raw"] = (df["n_missing_raw"] / max(len(numeric_cols), 1)).astype(np.float32)

        # --- 3. Layout density features ---
        area = df.get("floor_area_sqm", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
        if "floor_area_sqm" in df.columns:
            if "ceiling_height_m" in df.columns:
                df["warehouse_volume"] = (df["floor_area_sqm"] * df["ceiling_height_m"]).astype(float)
            if "intersection_count" in df.columns:
                df["intersection_density"] = safe_divide(df["intersection_count"], area).astype(np.float32)
            if "pack_station_count" in df.columns:
                df["pack_station_density"] = safe_divide(df["pack_station_count"], area).astype(np.float32)
            if "charger_count" in df.columns:
                df["charger_density"] = safe_divide(df["charger_count"], area).astype(np.float32)
            if "robot_total" in df.columns:
                df["robot_density_layout"] = safe_divide(df["robot_total"], area).astype(np.float32)
        if {"intersection_count", "aisle_width_avg"}.issubset(df.columns):
            df["movement_friction"] = safe_divide(df["intersection_count"], df["aisle_width_avg"]).astype(np.float32)
        if {"layout_compactness", "zone_dispersion"}.issubset(df.columns):
            df["compactness_x_dispersion"] = (df["layout_compactness"] * df["zone_dispersion"]).astype(float)
        if {"one_way_ratio", "intersection_count", "aisle_width_avg"}.issubset(df.columns):
            df["one_way_friction"] = (df["one_way_ratio"] * safe_divide(df["intersection_count"], df["aisle_width_avg"])).astype(float)

        # --- 4. Robot state decomposition ---
        if {"robot_active", "robot_idle", "robot_charging"}.issubset(df.columns):
            rts = df["robot_active"] + df["robot_idle"] + df["robot_charging"]
            rts_safe = rts.replace(0, np.nan)
            df["robot_total_state"] = rts
            df["robot_total_gap"] = rts - df["robot_total"]
            df["robot_active_share"] = safe_divide(df["robot_active"], rts_safe).astype(np.float32)
            df["robot_idle_share"] = safe_divide(df["robot_idle"], rts_safe).astype(np.float32)
            df["robot_charging_share"] = safe_divide(df["robot_charging"], rts_safe).astype(np.float32)
            df["charging_to_active_ratio"] = safe_divide(df["robot_charging"], df["robot_active"]).astype(np.float32)
            df["idle_to_active_ratio"] = safe_divide(df["robot_idle"], df["robot_active"]).astype(np.float32)

        # --- 5. Ratio features ---
        rt = df["robot_total"].replace(0, np.nan) if "robot_total" in df.columns else None
        cc = df["charger_count"].replace(0, np.nan) if "charger_count" in df.columns else None
        ps = df["pack_station_count"].replace(0, np.nan) if "pack_station_count" in df.columns else None
        ra = df["robot_active"].replace(0, np.nan) if "robot_active" in df.columns else None

        ratio_specs = [
            ("inflow_per_robot", "order_inflow_15m", rt),
            ("inflow_per_pack_station", "order_inflow_15m", ps),
            ("unique_sku_per_robot", "unique_sku_15m", rt),
            ("unique_sku_per_pack_station", "unique_sku_15m", ps),
            ("charging_per_charger", "robot_charging", cc),
            ("inflow_per_charger", "order_inflow_15m", cc),
            ("congestion_per_active", "congestion_score", ra),
            ("density_per_active", "max_zone_density", ra),
            ("fault_per_active", "fault_count_15m", ra),
            ("collision_per_active", "near_collision_15m", ra),
            ("blocked_per_active", "blocked_path_15m", ra),
            ("robot_active_per_intersection", "robot_active", df.get("intersection_count", pd.Series(np.nan, index=df.index)).replace(0, np.nan)),
        ]
        if "aisle_width_avg" in df.columns:
            aw = df["aisle_width_avg"].replace(0, np.nan)
            ratio_specs.append(("congestion_per_width", "congestion_score", aw))
            ratio_specs.append(("zone_density_per_width", "max_zone_density", aw))
            ratio_specs.append(("inflow_per_aisle_width", "order_inflow_15m", aw))
        if "staff_on_floor" in df.columns:
            ratio_specs.append(("inflow_per_staff", "order_inflow_15m", df["staff_on_floor"].replace(0, np.nan)))
        if "label_print_queue" in df.columns and ps is not None:
            ratio_specs.append(("label_queue_per_pack", "label_print_queue", ps))

        for name, num_col, denom in ratio_specs:
            if num_col in df.columns and denom is not None:
                df[name] = safe_divide(df[num_col], denom).astype(np.float32)

        # --- 6. Pressure interaction features ---
        if {"robot_charging", "charge_queue_length", "charger_count"}.issubset(df.columns):
            df["charge_pressure_pb"] = safe_divide(
                df["robot_charging"] + df["charge_queue_length"], cc
            ).astype(np.float32)
        if {"order_inflow_15m", "avg_package_weight_kg"}.issubset(df.columns):
            df["demand_mass"] = (df["order_inflow_15m"] * df["avg_package_weight_kg"]).astype(float)
            if rt is not None:
                df["demand_mass_per_robot"] = safe_divide(df["demand_mass"], rt).astype(np.float32)
        if {"order_inflow_15m", "avg_trip_distance"}.issubset(df.columns):
            df["trip_load"] = (df["order_inflow_15m"] * df["avg_trip_distance"]).astype(float)
            if rt is not None:
                df["trip_load_per_robot"] = safe_divide(df["trip_load"], rt).astype(np.float32)
        if {"order_inflow_15m", "unique_sku_15m"}.issubset(df.columns):
            df["complexity_load"] = (df["order_inflow_15m"] * df["unique_sku_15m"]).astype(float)
            if ps is not None:
                df["complexity_load_per_pack"] = safe_divide(df["complexity_load"], ps).astype(np.float32)
        if {"congestion_score", "low_battery_ratio"}.issubset(df.columns):
            df["congestion_x_lowbat"] = (df["congestion_score"] * df["low_battery_ratio"]).astype(float)
        if {"low_battery_ratio", "robot_active"}.issubset(df.columns):
            df["battery_pressure_pb"] = (df["low_battery_ratio"] * df["robot_active"]).astype(float)
        if {"charge_queue_length", "avg_charge_wait"}.issubset(df.columns):
            df["queue_wait_pressure"] = (df["charge_queue_length"] * df["avg_charge_wait"]).astype(float)
        if {"loading_dock_util", "pack_utilization"}.issubset(df.columns):
            df["dock_pack_pressure"] = (df["loading_dock_util"] * df["pack_utilization"]).astype(float)
        if {"staging_area_util", "pack_utilization"}.issubset(df.columns):
            df["staging_pack_pressure"] = (df["staging_area_util"] * df["pack_utilization"]).astype(float)
        if {"avg_recovery_time", "fault_count_15m"}.issubset(df.columns):
            df["recovery_x_fault"] = (df["avg_recovery_time"] * df["fault_count_15m"]).astype(float)
        if {"near_collision_15m", "blocked_path_15m"}.issubset(df.columns):
            df["collision_x_blocked"] = (df["near_collision_15m"] * df["blocked_path_15m"]).astype(float)

        # --- 7. Threshold features ---
        if "battery_mean" in df.columns:
            df["battery_below_44"] = np.clip(44.0 - df["battery_mean"], 0, None).astype(np.float32)
        if "charge_pressure_pb" in df.columns:
            df["charge_pressure_above_1_36"] = np.clip(df["charge_pressure_pb"] - 1.36, 0, None).astype(np.float32)

        # --- 8. Squared features ---
        for col in ["pack_utilization", "loading_dock_util", "staging_area_util"]:
            if col in df.columns:
                df[f"{col}_sq"] = (df[col].astype(float) ** 2).astype(np.float32)

        # --- 9. Onset features ---
        onset_new = add_onset_features(df, "robot_charging", "charging_onset", grp_key)
        onset_new += add_onset_features(df, "charge_queue_length", "queue_onset", grp_key)

        # --- 10. Rolling deviation & max (for SEQ_COLS not already covered) ---
        for col in SEQ_COLS_EXTRA:
            if col not in df.columns:
                continue
            rollmax_name = f"{col}_rollmax3"
            dev_name = f"{col}_dev_rmean3"
            if rollmax_name in df.columns:
                continue  # already exists
            lag1 = grp[col].shift(1)
            lg = lag1.groupby(grp_key)
            rmean = lg.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
            rmax = lg.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
            df[rollmax_name] = rmax
            df[dev_name] = df[col] - rmean

        # --- 11. Baseline expanding features ---
        for col in BASELINE_EXPAND_COLS:
            if col not in df.columns:
                continue
            emean_name = f"{col}_expmean"
            delta_name = f"{col}_delta_expmean"
            if emean_name in df.columns:
                continue
            prev = grp[col].shift(1)
            emean = prev.groupby(grp_key).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
            df[emean_name] = emean
            df[delta_name] = df[col] - emean

        # --- 12. Layout one-hot ---
        if "layout_type" in df.columns:
            for lt in ["grid", "hub_spoke", "hybrid", "narrow"]:
                df[f"layout_type_{lt}"] = (df["layout_type"] == lt).astype(np.int8)
        elif "layout_type_enc" in df.columns:
            for lt_val, lt_name in [(0, "grid"), (1, "hub_spoke"), (2, "hybrid"), (3, "narrow")]:
                df[f"layout_type_{lt_name}"] = (df["layout_type_enc"] == lt_val).astype(np.int8)

        # --- 13. Scenario-level aggregation (mean/std/max/min across all 25 timesteps) ---
        sc_agg_candidates = [
            "order_inflow_15m", "congestion_score", "low_battery_ratio",
            "robot_charging", "charge_queue_length", "robot_active", "robot_idle",
            "battery_mean", "pack_utilization", "loading_dock_util",
            "staging_area_util", "fault_count_15m", "near_collision_15m",
            "blocked_path_15m", "avg_charge_wait", "avg_trip_distance",
            "charge_pressure_pb", "battery_pressure_pb", "demand_mass_per_robot",
            "congestion_x_lowbat", "queue_wait_pressure",
        ]
        sc_agg_cols = [c for c in sc_agg_candidates if c in df.columns]
        sc_grp = df.groupby(cfg.GROUP_COL, sort=False)[sc_agg_cols]
        sc_mean = sc_grp.transform("mean").add_suffix("__sc_mean")
        sc_std  = sc_grp.transform("std").fillna(0).add_suffix("__sc_std")
        sc_max  = sc_grp.transform("max").add_suffix("__sc_max")
        sc_min  = sc_grp.transform("min").add_suffix("__sc_min")
        for agg_df in (sc_mean, sc_std, sc_max, sc_min):
            for c in agg_df.columns:
                df[c] = agg_df[c].astype(np.float32)

    # Collect all new columns (union from both dfs)
    base_set = set(feature_cols)
    all_cols_train = set(train_out.columns)
    all_cols_test = set(test_out.columns)
    exclude = {cfg.ID_COL, cfg.GROUP_COL, "layout_id", cfg.TARGET, "_is_train", "layout_type"}
    new_feature_cols = sorted(
        (all_cols_train & all_cols_test) - base_set - exclude
    )
    final_cols = list(feature_cols) + [c for c in new_feature_cols if c not in base_set]

    log(f"  PB-style features: {len(feature_cols)} -> {len(final_cols)} (+{len(final_cols) - len(feature_cols)})")
    return train_out, test_out, final_cols


# ── Sample weighting ────────────────────────────────────────────────

def build_v8_sample_weight(y_raw, time_idx=None):
    """Train-only sample weight: boost high-delay and late-timestep samples."""
    w = np.ones(len(y_raw), dtype=np.float32)
    q90 = np.nanquantile(y_raw, 0.90)
    q95 = np.nanquantile(y_raw, 0.95)
    q99 = np.nanquantile(y_raw, 0.99)
    w += 0.15 * (y_raw >= q90).astype(np.float32)
    w += 0.30 * (y_raw >= q95).astype(np.float32)
    w += 0.60 * (y_raw >= q99).astype(np.float32)
    if time_idx is not None:
        t = np.asarray(time_idx, dtype=np.float32)
        t_max = max(t.max(), 1.0)
        w += 0.08 * (t / t_max)
    w /= w.mean()  # normalize to mean 1.0
    log(f"  sample weight: q90={q90:.1f} q95={q95:.1f} q99={q99:.1f} w_range=[{w.min():.3f}, {w.max():.3f}]")
    return w


# ── Per-model training ──────────────────────────────────────────────

MODEL_SPECS = [
    {
        "name": "lgb_mae_raw",
        "family": "lgb",
        "transform": "none",
        "params": {
            "objective": "mae", "metric": "mae",
            "n_estimators": 3000, "learning_rate": 0.03,
            "num_leaves": 96, "max_depth": -1,
            "min_child_samples": 80,
            "subsample": 0.9, "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1, "reg_lambda": 1.5,
            "verbosity": -1,
        },
    },
    {
        "name": "lgb_huber_log",
        "family": "lgb",
        "transform": "log1p",
        "params": {
            "objective": "huber", "alpha": 0.9,
            "metric": "mae",
            "n_estimators": 3000, "learning_rate": 0.03,
            "num_leaves": 128, "max_depth": -1,
            "min_child_samples": 60,
            "subsample": 0.9, "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05, "reg_lambda": 1.0,
            "verbosity": -1,
        },
    },
    {
        "name": "xgb_mae_raw",
        "family": "xgb",
        "transform": "none",
        "params": {
            "objective": "reg:absoluteerror", "eval_metric": "mae",
            "n_estimators": 3000, "learning_rate": 0.03,
            "max_depth": 8, "min_child_weight": 6,
            "subsample": 0.9, "colsample_bytree": 0.85,
            "reg_alpha": 0.05, "reg_lambda": 1.5,
            "tree_method": "hist", "verbosity": 0,
        },
    },
    {
        "name": "cat_mae_log",
        "family": "cat",
        "transform": "log1p",
        "params": {
            "loss_function": "MAE", "eval_metric": "MAE",
            "iterations": 3000, "learning_rate": 0.03,
            "depth": 8, "l2_leaf_reg": 5.0,
            "subsample": 0.9,
            "verbose": 100,
            "early_stopping_rounds": 100,
        },
    },
    {
        "name": "lgb_mae_log",
        "family": "lgb",
        "transform": "log1p",
        "params": {
            "objective": "mae", "metric": "mae",
            "n_estimators": 3000, "learning_rate": 0.03,
            "max_depth": 8, "num_leaves": 127,
            "min_child_samples": 50,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.5, "reg_lambda": 1.0,
            "verbosity": -1,
        },
    },
]


def train_single_model_cv(spec, X, y_raw, groups, X_test,
                          n_folds=5, early_stop=100, seed=42,
                          sample_weight=None, ckpt_prefix="v8"):
    """Train one model type with its own objective and transform."""
    name = spec["name"]
    family = spec["family"]
    transform = spec["transform"]
    params = spec["params"].copy()
    ckpt_name = f"{ckpt_prefix}_{name}_seed{seed}"

    cached = v5.load_ckpt(ckpt_name)
    if cached is not None:
        return cached

    # Set seed
    if family == "lgb":
        params["random_state"] = seed
    elif family == "xgb":
        params["random_state"] = seed
    elif family == "cat":
        params["random_seed"] = seed

    y_tr_all = v5.transform_target(y_raw, transform)
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
        fold_ckpt = f"{ckpt_name}_fold{fold}"
        fc = v5.load_ckpt(fold_ckpt)
        if fc is not None:
            oof[val_idx] = fc["oof"]
            test_pred += fc["test"] / n_folds
            log(f"  {name} seed{seed} fold{fold+1} MAE: {fc['mae']:.4f} (캐시)")
            continue

        ft0 = time.time()
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_tr_all[tr_idx], y_tr_all[val_idx]
        y_val_raw = y_raw[val_idx]
        sw = sample_weight[tr_idx] if sample_weight is not None else None

        if family == "lgb":
            m = v5.train_lgb(X_tr, y_tr, X_val, y_val, params, early_stop, sw)
        elif family == "xgb":
            m = v5.train_xgb(X_tr, y_tr, X_val, y_val, params, early_stop, sw)
        elif family == "cat":
            m = v5.train_cat(X_tr, y_tr, X_val, y_val, params, sw)
        else:
            raise ValueError(f"Unknown family: {family}")

        p_val = v5.inverse_transform(m.predict(X_val), transform)
        p_test = v5.inverse_transform(m.predict(X_test), transform)
        oof[val_idx] = p_val
        test_pred += p_test / n_folds
        fold_mae = mean_absolute_error(y_val_raw, p_val)
        log(f"  {name} seed{seed} fold{fold+1} MAE: {fold_mae:.4f} ({time.time()-ft0:.0f}s)")

        v5.save_ckpt(fold_ckpt, {"oof": p_val, "test": p_test, "mae": fold_mae})

    cv = mean_absolute_error(y_raw, oof)
    result = (cv, test_pred, oof)
    v5.save_ckpt(ckpt_name, result)
    log(f"  {name} seed{seed} CV: {cv:.4f}")
    return result


def train_model_multiseed(spec, X, y_raw, groups, X_test,
                          seeds=(42, 123, 2026), ckpt_prefix="v8", **kwargs):
    """Multi-seed wrapper for a single model spec."""
    all_oof, all_test = [], []
    for seed in seeds:
        cv, test, oof = train_single_model_cv(
            spec, X, y_raw, groups, X_test, seed=seed,
            ckpt_prefix=ckpt_prefix, **kwargs,
        )
        all_oof.append(oof)
        all_test.append(test)

    oof_avg = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)
    cv_avg = mean_absolute_error(y_raw, oof_avg)
    log(f"  {spec['name']} {len(seeds)}-seed avg CV: {cv_avg:.4f}")
    return cv_avg, test_avg, oof_avg


# ── Submission helpers ──────────────────────────────────────────────

def save_submission(test_fe, pred, name):
    sub = pd.DataFrame({
        cfg.ID_COL: test_fe[cfg.ID_COL].astype(str),
        cfg.TARGET: np.clip(pred, 0, None),
    })
    sample = pd.read_csv(cfg.SAMPLE_SUB_FILE)
    assert len(sub) == len(sample), f"row mismatch: {len(sub)} != {len(sample)}"
    assert sub[cfg.TARGET].isna().sum() == 0, "NaN"
    assert (sub[cfg.TARGET] >= 0).all(), "negative"
    path = os.path.join(cfg.SUBMISSION_DIR, f"{name}_submission.csv")
    sub.to_csv(path, index=False, float_format="%.10f", lineterminator="\n")
    log(f"  저장: {path} mean={sub[cfg.TARGET].mean():.3f} std={sub[cfg.TARGET].std():.3f} max={sub[cfg.TARGET].max():.3f}")
    return path


# ── Main orchestrator ───────────────────────────────────────────────

def run_v8(sanity=False, ckpt_prefix="v8"):
    log("=" * 60)
    log(f"v8 — PB features + objective diversity + sample weight  [prefix={ckpt_prefix}]")
    if sanity:
        log("!! SANITY CHECK 모드")
    log("=" * 60)

    # 1. Build strict base features
    train_fe, test_fe, feature_cols = strict.build_strict_data(sanity=sanity)

    # 2. Add pressure + lead/forward (same as attack3: expanded + core)
    log("  attack3-style features (expanded pressure + core lead)...")
    train_fe, test_fe, feature_cols = attack.add_lead_features_split(
        train_fe, test_fe, feature_cols,
        lead_mode="core", base_mode="expanded",
    )

    # 3. Add PB-style features
    log("  PB-style feature augmentation...")
    train_fe, test_fe, feature_cols = add_pb_style_features(
        train_fe, test_fe, feature_cols,
    )

    X = train_fe[feature_cols]
    X_test = strict.align_test_features(test_fe, feature_cols)
    y_raw = train_fe[cfg.TARGET].values
    groups = train_fe[cfg.GROUP_COL].values

    log(f"  Final features: {len(feature_cols)}, train={X.shape}, test={X_test.shape}")

    # 4. Sample weight
    time_idx = train_fe["timestep"].values if "timestep" in train_fe.columns else None
    sw = build_v8_sample_weight(y_raw, time_idx)

    # 5. Train diverse models
    seeds = [42] if sanity else [42, 123, 2026]
    n_folds = 2 if sanity else 5
    early_stop = 30 if sanity else 100

    specs = MODEL_SPECS
    if sanity:
        # Reduce iterations for sanity
        specs = []
        for s in MODEL_SPECS:
            s2 = {**s, "params": s["params"].copy()}
            if s2["family"] in ("lgb", "xgb"):
                s2["params"]["n_estimators"] = 200
            elif s2["family"] == "cat":
                s2["params"]["iterations"] = 200
            specs.append(s2)

    model_results = {}
    for spec in specs:
        log(f"\n  === {spec['name']} ({spec['family']}, {spec['transform']}) ===")
        cv, test, oof = train_model_multiseed(
            spec, X, y_raw, groups, X_test,
            seeds=seeds, n_folds=n_folds, early_stop=early_stop,
            sample_weight=sw, ckpt_prefix=ckpt_prefix,
        )
        model_results[spec["name"]] = {"cv": cv, "test": test, "oof": oof}

    # 6. Load existing checkpoints
    log("\n  === Existing checkpoints ===")
    existing = {}
    for ckpt_name, ckpt_dir, prefix in [
        ("attack3_gbdt", attack.ATTACK_CKPT_DIR, "attack_3_expanded_core_lead_gbdt"),
        ("attack4_gbdt", attack.ATTACK_CKPT_DIR, "attack_4_expanded_ops_lead_gbdt"),
        ("strict2_tabnet", strict.STRICT_CKPT_DIR, "strict_2_focused_tabnet"),
    ]:
        try:
            cvs, test_pred, oof_pred = attack.load_strict_seed_bundle(prefix, seeds=(42, 123, 2026))
            if len(oof_pred) != len(y_raw):
                log(f"  {ckpt_name}: size mismatch ({len(oof_pred)} vs {len(y_raw)}), skipping")
                continue
            cv = mean_absolute_error(y_raw, oof_pred)
            existing[ckpt_name] = {"cv": cv, "test": test_pred, "oof": oof_pred}
            log(f"  {ckpt_name}: CV={cv:.4f}")
        except FileNotFoundError:
            log(f"  {ckpt_name}: not found, skipping")

    # 7. Build blend candidates
    all_names = []
    all_oofs = []
    all_tests = []

    for name, res in model_results.items():
        all_names.append(name)
        all_oofs.append(res["oof"])
        all_tests.append(res["test"])

    for name, res in existing.items():
        all_names.append(name)
        all_oofs.append(res["oof"])
        all_tests.append(res["test"])

    # 8. N-way blend
    log("\n  === N-way blend optimization ===")
    weights, blend_cv = v5.optimize_blend_multi(all_oofs, y_raw, all_names)

    oof_blend = sum(w * o for w, o in zip(weights, all_oofs))
    test_blend = sum(w * t for w, t in zip(weights, all_tests))

    # 9. Save submission variants
    log("\n  === Submission variants ===")

    # Variant 1: full blend noclip
    save_submission(test_fe, test_blend, "v8_diverse_blend_noclip")

    # Variant 2: full blend clip130
    clip130 = np.percentile(oof_blend, 99) * 1.30
    save_submission(test_fe, np.clip(test_blend, 0, clip130), "v8_diverse_blend_clip130")
    cv_clip130 = mean_absolute_error(y_raw, np.clip(oof_blend, 0, clip130))
    log(f"  clip130={clip130:.2f}, CV clipped={cv_clip130:.4f}")

    # Variant 3: v8 new models only
    v8_names = list(model_results.keys())
    v8_oofs = [model_results[n]["oof"] for n in v8_names]
    v8_tests = [model_results[n]["test"] for n in v8_names]
    w_v8, cv_v8 = v5.optimize_blend_multi(v8_oofs, y_raw, v8_names)
    test_v8_only = sum(w * t for w, t in zip(w_v8, v8_tests))
    save_submission(test_fe, test_v8_only, "v8_new_models_only")

    # Variant 4: top3 by CV
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]["cv"])
    top3 = sorted_models[:3]
    top3_names = [n for n, _ in top3]
    top3_oofs = [model_results[n]["oof"] for n in top3_names]
    top3_tests = [model_results[n]["test"] for n in top3_names]
    w_top3, cv_top3 = v5.optimize_blend_multi(top3_oofs, y_raw, top3_names)
    test_top3 = sum(w * t for w, t in zip(w_top3, top3_tests))
    save_submission(test_fe, test_top3, "v8_top3_blend")

    # Variant 5: v8 blend + attack3 noclip submission average
    if "attack3_gbdt" in existing:
        hybrid = 0.5 * test_blend + 0.5 * existing["attack3_gbdt"]["test"]
        save_submission(test_fe, hybrid, "v8_attack_hybrid")

    # Variant 6: conservative clip110
    clip110 = np.percentile(oof_blend, 99) * 1.10
    save_submission(test_fe, np.clip(test_blend, 0, clip110), "v8_conservative_clip110")

    # Summary
    log("\n" + "=" * 60)
    log("v8 결과 요약")
    log("=" * 60)
    log(f"  Features: {len(feature_cols)}")
    for name, res in model_results.items():
        log(f"  {name:<20} CV={res['cv']:.4f}")
    for name, res in existing.items():
        log(f"  {name:<20} CV={res['cv']:.4f} (cached)")
    log(f"  N-way blend CV: {blend_cv:.4f}")
    log(f"  clip130 CV: {cv_clip130:.4f}")
    log(f"  v8-only blend CV: {cv_v8:.4f}")
    log(f"  top3 blend CV: {cv_top3:.4f}")
    log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="v8 diverse pipeline")
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--ckpt_prefix", default="v8", help="체크포인트 prefix (기본: v8)")
    args = parser.parse_args()

    ckpt_dir = V8_SANITY_CKPT_DIR if args.sanity else V8_CKPT_DIR
    v5.CKPT_DIR = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    t0 = time.time()
    run_v8(sanity=args.sanity, ckpt_prefix=args.ckpt_prefix)
    log(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
