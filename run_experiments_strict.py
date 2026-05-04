"""
Strict-clean experiments for code-verifiable submissions.

Rules enforced:
- test.csv is never used for model fitting, CV tuning, feature selection,
  sample weighting, scaler/imputer fitting, or clipping-threshold selection.
- test.csv is loaded only after train-side features and thresholds are fixed,
  then used for final inference.
- no pseudo-labeling, no train/test adversarial modeling, no Public-LB tuning.
"""

import argparse
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import config as cfg
from dataset import (
    add_expanding_features,
    add_interaction_features,
    add_lag_features,
    add_rolling_features,
    add_scenario_stats,
    add_timestep,
    merge_layout,
)
import run_experiments_v5 as v5


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRICT_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_strict")
STRICT_SANITY_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_strict_sanity")


def log(msg):
    v5.log(msg)


def load_train_only():
    train = pd.read_csv(cfg.TRAIN_FILE)
    layout = pd.read_csv(cfg.LAYOUT_FILE)
    return train, layout


def load_test_only():
    test = pd.read_csv(cfg.TEST_FILE)
    layout = pd.read_csv(cfg.LAYOUT_FILE)
    return test, layout


def build_features_one(df, layout, has_target):
    """Build features for one split only. No train/test concat."""
    out = df.copy()
    out = merge_layout(out, layout)
    out = add_timestep(out)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    out = add_expanding_features(out)
    out = add_interaction_features(out)
    out = add_scenario_stats(out)

    exclude = set(cfg.ID_COLS + ["_is_train"])
    if has_target:
        exclude.add(cfg.TARGET)
    feature_cols = [c for c in out.columns if c not in exclude]
    return out, feature_cols


def align_test_features(test_fe, feature_cols):
    missing = [c for c in feature_cols if c not in test_fe.columns]
    if missing:
        raise ValueError(f"test missing features: {missing[:10]}")
    return test_fe[feature_cols]


def take_sanity_subset(train_fe, max_groups=1200):
    groups = train_fe[cfg.GROUP_COL].drop_duplicates().iloc[:max_groups]
    return train_fe[train_fe[cfg.GROUP_COL].isin(groups)].reset_index(drop=True)


def add_highload_strict(train_fe, test_fe, feature_cols):
    """v5-4 style high-load features without adversarial check/selection."""
    train_out = train_fe.copy()
    test_out = test_fe.copy()
    thresholds = {
        "order_p75": np.nanpercentile(train_out["order_inflow_15m"], 75),
        "battery_p90": np.nanpercentile(train_out["low_battery_ratio"], 90),
        "congestion_p90": np.nanpercentile(train_out["congestion_score"], 90),
    }
    log(
        "  strict thresholds: "
        f"order_P75={thresholds['order_p75']:.2f} "
        f"battery_P90={thresholds['battery_p90']:.4f} "
        f"congestion_P90={thresholds['congestion_p90']:.2f}"
    )

    for df in (train_out, test_out):
        rt = df["robot_total"].replace(0, np.nan)
        cc = df["charger_count"].replace(0, np.nan)

        df["order_surge"] = (df["order_inflow_15m"] > thresholds["order_p75"]).astype(np.float32)
        df["battery_critical"] = (df["low_battery_ratio"] > thresholds["battery_p90"]).astype(np.float32)
        df["congestion_critical"] = (df["congestion_score"] > thresholds["congestion_p90"]).astype(np.float32)
        df["robot_capacity_used"] = (df["robot_active"] + df["robot_charging"]) / rt
        df["stress_index"] = df["order_inflow_15m"] * df["congestion_score"] * df["low_battery_ratio"]
        df["bottleneck_score"] = df["max_zone_density"] * (1 - df["robot_idle"] / rt)
        df["recovery_pressure"] = df["charge_queue_length"] * df["avg_charge_wait"] / (cc + 1)
        df["demand_supply_gap"] = df["order_inflow_15m"] - df["robot_active"]
        df["cascade_risk"] = df["fault_count_15m"] * df["congestion_score"] * df["blocked_path_15m"]
        df["pack_station_per_robot"] = df["pack_station_count"] / rt
        df["charger_per_robot"] = df["charger_count"] / rt

        grp = df.groupby(cfg.GROUP_COL)["stress_index"]
        df["stress_acceleration"] = df["stress_index"] - grp.shift(1)
        shifted = grp.shift(1)
        df["sustained_stress"] = shifted.groupby(df[cfg.GROUP_COL]).rolling(
            3, min_periods=1
        ).mean().reset_index(level=0, drop=True)
        sc_max = grp.transform("max").replace(0, np.nan)
        df["peak_stress_ratio"] = df["stress_index"] / sc_max

    new_cols = [
        "order_surge", "battery_critical", "congestion_critical", "robot_capacity_used",
        "stress_index", "bottleneck_score", "recovery_pressure", "demand_supply_gap",
        "cascade_risk", "pack_station_per_robot", "charger_per_robot",
        "stress_acceleration", "sustained_stress", "peak_stress_ratio",
    ]
    return train_out, test_out, feature_cols + new_cols


def add_pressure_strict(train_fe, test_fe, feature_cols, mode):
    """v6 pressure features, but built without any test-driven selection/weights."""
    train_out, test_out, cols = add_highload_strict(train_fe, test_fe, feature_cols)
    new_cols = []

    for df in (train_out, test_out):
        robot_total = df["robot_total"].replace(0, np.nan)
        robot_active = df["robot_active"].replace(0, np.nan)
        charger_count = df["charger_count"].replace(0, np.nan)
        pack_station = df["pack_station_count"].replace(0, np.nan)

        df["order_per_robot"] = df["order_inflow_15m"] / robot_total
        df["order_per_active_robot"] = df["order_inflow_15m"] / robot_active
        df["urgent_order_load"] = df["order_inflow_15m"] * df["urgent_order_ratio"]
        df["heavy_order_load"] = df["order_inflow_15m"] * df["heavy_item_ratio"]
        df["sku_complexity_load"] = df["unique_sku_15m"] * (1 + df["sku_concentration"])
        df["capacity_gap"] = df["order_inflow_15m"] - (df["robot_active"] + df["robot_idle"])
        df["idle_capacity_buffer"] = df["robot_idle"] / robot_total
        df["charger_queue_per_charger"] = df["charge_queue_length"] / charger_count
        df["charge_pressure"] = df["charge_queue_length"] * df["avg_charge_wait"] / (charger_count + 1)
        df["battery_deficit_pressure"] = (100 - df["battery_mean"]) * df["low_battery_ratio"]
        df["charging_capacity_pressure"] = df["robot_charging"] / charger_count
        df["pack_pressure"] = df["order_inflow_15m"] * df["pack_utilization"] / (pack_station + 1)
        df["dock_pressure"] = df["loading_dock_util"] * df["outbound_truck_wait_min"]
        df["traffic_pressure"] = df["congestion_score"] * (
            df["blocked_path_15m"] + df["near_collision_15m"] + df["intersection_wait_time_avg"]
        )
        df["conveyor_pack_pressure"] = df["pack_utilization"] / (df["conveyor_speed_mps"] + 0.1)
        df["staging_pressure"] = df["staging_area_util"] * df["order_inflow_15m"]

        if mode == "expanded":
            df["staffing_pressure"] = df["order_inflow_15m"] / (df["staff_on_floor"] + 1)
            df["forecast_miss_load"] = df["order_inflow_15m"] * (1 - df["daily_forecast_accuracy"])
            df["agv_failure_load"] = df["order_inflow_15m"] * (1 - df["agv_task_success_rate"])
            df["system_drag"] = (
                df["wms_response_time_ms"]
                * (df["network_latency_ms"] + 1)
                * (1 + df["scanner_error_rate"])
            )
            df["inventory_congestion"] = (
                df["storage_density_pct"] * df["vertical_utilization"] * df["aisle_traffic_score"]
            )
            df["robot_health_drag"] = (
                (df["fleet_age_months_avg"] / 60.0)
                * (100 - df["maintenance_schedule_score"])
                * (100 - df["robot_calibration_score"])
                / 100.0
            )
            df["pick_wave_pressure"] = (
                df["order_wave_count"] * df["pick_list_length_avg"] * (1 + df["bulk_order_ratio"])
            )
            df["express_pressure"] = (
                df["express_lane_util"] * df["urgent_order_ratio"] * df["order_inflow_15m"]
            )
            df["return_rework_pressure"] = (
                df["return_order_ratio"] * df["quality_check_rate"] * df["order_inflow_15m"]
            )
            df["cold_chain_load"] = df["cold_chain_ratio"] * df["order_inflow_15m"]

        for col in ["capacity_gap", "charge_pressure", "pack_pressure", "traffic_pressure"]:
            grp = df.groupby(cfg.GROUP_COL)[col]
            df[f"{col}_diff1"] = df[col] - grp.shift(1)
            shifted = grp.shift(1)
            df[f"{col}_rmean3"] = shifted.groupby(df[cfg.GROUP_COL]).rolling(
                3, min_periods=1
            ).mean().reset_index(level=0, drop=True)

    focused_cols = [
        "order_per_robot", "order_per_active_robot", "urgent_order_load", "heavy_order_load",
        "sku_complexity_load", "capacity_gap", "idle_capacity_buffer",
        "charger_queue_per_charger", "charge_pressure", "battery_deficit_pressure",
        "charging_capacity_pressure", "pack_pressure", "dock_pressure", "traffic_pressure",
        "conveyor_pack_pressure", "staging_pressure",
    ]
    expanded_cols = [
        "staffing_pressure", "forecast_miss_load", "agv_failure_load", "system_drag",
        "inventory_congestion", "robot_health_drag", "pick_wave_pressure", "express_pressure",
        "return_rework_pressure", "cold_chain_load",
    ]
    temporal_cols = []
    for c in ["capacity_gap", "charge_pressure", "pack_pressure", "traffic_pressure"]:
        temporal_cols.extend([f"{c}_diff1", f"{c}_rmean3"])

    new_cols = focused_cols + temporal_cols
    if mode == "expanded":
        new_cols += expanded_cols

    out_cols = list(cols)
    seen = set(out_cols)
    for col in new_cols:
        if col not in seen:
            out_cols.append(col)
            seen.add(col)
    return train_out, test_out, out_cols


def oof_clip(oof_pred, test_pred, factor=1.10):
    clip_val = np.percentile(oof_pred, 99) * factor
    oof_out = np.clip(oof_pred, 0, clip_val)
    test_out = np.clip(test_pred, 0, clip_val)
    return oof_out, test_out, clip_val


def save_submission(test_fe, preds, name):
    sub = pd.DataFrame({cfg.ID_COL: test_fe[cfg.ID_COL], cfg.TARGET: np.clip(preds, 0, None)})
    sample = pd.read_csv(cfg.SAMPLE_SUB_FILE)
    if len(sub) != len(sample):
        raise ValueError(f"submission rows mismatch: {len(sub)} != {len(sample)}")
    path = os.path.join(cfg.SUBMISSION_DIR, f"{name}_submission.csv")
    sub.to_csv(path, index=False)
    log(f"  저장: {path} mean={sub[cfg.TARGET].mean():.2f} std={sub[cfg.TARGET].std():.2f} max={sub[cfg.TARGET].max():.2f}")
    return path


def gbdt_params(sanity=False):
    lgb_p = cfg.LGB_PARAMS.copy()
    xgb_p = cfg.XGB_PARAMS.copy()
    cat_p = cfg.CAT_PARAMS.copy()
    if sanity:
        lgb_p["n_estimators"] = cfg.SANITY_N_ESTIMATORS
        xgb_p["n_estimators"] = cfg.SANITY_N_ESTIMATORS
        cat_p["iterations"] = cfg.SANITY_N_ESTIMATORS
    return lgb_p, xgb_p, cat_p


def build_strict_data(sanity=False):
    log("train 데이터 로딩 및 train-only feature build...")
    train_raw, layout = load_train_only()
    train_fe, feature_cols = build_features_one(train_raw, layout, has_target=True)
    if sanity:
        train_fe = take_sanity_subset(train_fe)
        log(f"  sanity train subset: {train_fe.shape}")

    log("test 데이터 로딩 및 fixed transform 적용...")
    test_raw, layout_test = load_test_only()
    test_fe, test_cols = build_features_one(test_raw, layout_test, has_target=False)
    missing = [c for c in feature_cols if c not in test_cols]
    if missing:
        raise ValueError(f"feature mismatch: {missing[:10]}")

    log(f"  strict base features: {len(feature_cols)} train={train_fe.shape} test={test_fe.shape}")
    return train_fe, test_fe, feature_cols


def run_experiment(name, train_fe, test_fe, feature_cols, mode, sanity=False, skip_tabnet=False):
    log("=" * 60)
    log(f"{name}: strict-clean mode={mode}")
    log("=" * 60)

    if mode == "base":
        train_x, test_x, cols = train_fe.copy(), test_fe.copy(), list(feature_cols)
    elif mode in ("focused", "expanded"):
        train_x, test_x, cols = add_pressure_strict(train_fe, test_fe, feature_cols, mode=mode)
    else:
        raise ValueError(mode)

    X = train_x[cols]
    X_test = align_test_features(test_x, cols)
    y_raw = train_x[cfg.TARGET].values
    groups = train_x[cfg.GROUP_COL].values

    seeds = [42] if sanity else [42, 123, 2026]
    n_folds = 2 if sanity else cfg.N_FOLDS
    early_stop = 30 if sanity else cfg.EARLY_STOPPING_ROUNDS
    max_epochs = 10 if sanity else 200
    lgb_p, xgb_p, cat_p = gbdt_params(sanity=sanity)

    t0 = time.time()
    log("  [1/3] strict GBDT 학습...")
    cv_gbdt, test_gbdt, oof_gbdt = v5.run_gbdt_multiseed(
        X, y_raw, groups, X_test,
        seeds=seeds,
        lgb_params=lgb_p,
        xgb_params=xgb_p,
        cat_params=cat_p,
        n_folds=n_folds,
        early_stop=early_stop,
        ckpt_prefix=f"{name}_gbdt",
    )

    models = [("gbdt", cv_gbdt, oof_gbdt, test_gbdt)]
    if not skip_tabnet:
        log("  [2/3] strict TabNet 학습 (scaler fit=train only)...")
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X.fillna(0)).astype(np.float32)
        Xt_arr = scaler.transform(X_test.fillna(0)).astype(np.float32)
        cv_tab, test_tab, oof_tab = v5.run_tabnet_multiseed(
            X_arr, y_raw, groups, Xt_arr,
            seeds=seeds,
            ckpt_prefix=f"{name}_tabnet",
            n_folds=n_folds,
            max_epochs=max_epochs,
        )
        models.append(("tabnet", cv_tab, oof_tab, test_tab))

    log("  [3/3] train OOF 기준 blend/clipping...")
    if len(models) == 1:
        model_name, cv, oof, test_pred = models[0]
        alpha = 1.0
    else:
        _, _, oof_a, test_a = models[0]
        _, _, oof_b, test_b = models[1]
        alpha, cv = v5.optimize_blend_2way(oof_a, oof_b, y_raw)
        oof = alpha * oof_a + (1 - alpha) * oof_b
        test_pred = alpha * test_a + (1 - alpha) * test_b

    oof_pp, test_pp, clip_val = oof_clip(oof, test_pred, factor=1.10)
    cv_pp = mean_absolute_error(y_raw, oof_pp)
    path = save_submission(test_fe, test_pp, name.replace("_", "-"))
    log(
        f"{name} 완료 -- CV raw={cv:.4f} CV clipped={cv_pp:.4f} "
        f"alpha={alpha:.3f} clip_from_oof={clip_val:.2f} path={path} "
        f"elapsed={(time.time() - t0) / 60:.1f}m"
    )
    return {"name": name, "cv": cv, "cv_clipped": cv_pp, "path": path}


def parse_exp(arg):
    all_exps = [
        ("strict_1_base", "base"),
        ("strict_2_focused", "focused"),
        ("strict_3_expanded", "expanded"),
    ]
    if not arg:
        return all_exps
    selected = set()
    for part in arg.split(","):
        if "-" in part:
            a, b = part.split("-")
            selected.update(range(int(a), int(b) + 1))
        else:
            selected.add(int(part))
    return [item for i, item in enumerate(all_exps, start=1) if i in selected]


def main():
    parser = argparse.ArgumentParser(description="strict-clean smart warehouse experiments")
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--skip-tabnet", action="store_true")
    args = parser.parse_args()

    v5.CKPT_DIR = STRICT_SANITY_CKPT_DIR if args.sanity else STRICT_CKPT_DIR
    os.makedirs(v5.CKPT_DIR, exist_ok=True)
    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    log("=" * 60)
    log("strict-clean 실험 시작")
    if args.sanity:
        log("!! SANITY CHECK 모드")
    log(f"체크포인트: {v5.CKPT_DIR}")
    log("=" * 60)

    train_fe, test_fe, feature_cols = build_strict_data(sanity=args.sanity)
    results = []
    for name, mode in parse_exp(args.exp):
        results.append(run_experiment(
            name, train_fe, test_fe, feature_cols,
            mode=mode,
            sanity=args.sanity,
            skip_tabnet=args.skip_tabnet,
        ))

    print("\n" + "=" * 60)
    print("strict-clean 결과 요약")
    print("=" * 60)
    for r in results:
        print(f"{r['name']:<20} CV={r['cv']:.4f} clipped={r['cv_clipped']:.4f} {r['path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
