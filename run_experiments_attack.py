"""
Strict-clean attack-track experiments.

This reimplements the v7 lead/forward idea without train/test concat and
without v5/v6/v7 cached submissions. Test rows are used only for final
feature transformation and inference.

Risk note:
- Lead/forward covariates use later timesteps from the same scenario.
- They do not use test targets or pseudo-labels, but they assume all scenario
  covariates are available at inference time.
"""

import argparse
import os
import pickle
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import config as cfg
import run_experiments_strict as strict
import run_experiments_v5 as v5


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTACK_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_attack")
ATTACK_SANITY_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_attack_sanity")
STRICT_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_strict")


CORE_LEAD_FEATURES = [
    "order_inflow_15m", "congestion_score", "low_battery_ratio", "battery_mean",
    "robot_idle", "robot_charging", "robot_active", "max_zone_density",
    "charge_queue_length", "avg_charge_wait", "pack_utilization",
    "loading_dock_util", "blocked_path_15m", "near_collision_15m",
]

OPS_LEAD_FEATURES = [
    "urgent_order_ratio", "heavy_item_ratio", "unique_sku_15m", "sku_concentration",
    "fault_count_15m", "avg_recovery_time", "staging_area_util",
    "intersection_wait_time_avg", "aisle_traffic_score", "outbound_truck_wait_min",
    "wms_response_time_ms", "network_latency_ms",
]


def log(msg):
    v5.log(msg)


def append_unique(base_cols, new_cols):
    seen = set(base_cols)
    out = list(base_cols)
    for col in new_cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def add_lead_features_split(train_fe, test_fe, feature_cols, lead_mode="core", base_mode="focused"):
    """Add same-scenario forward covariates per split. No train/test concat."""
    if base_mode in ("focused", "expanded"):
        train_out, test_out, cols = strict.add_pressure_strict(
            train_fe, test_fe, feature_cols, mode=base_mode
        )
    elif base_mode == "base":
        train_out, test_out, cols = train_fe.copy(), test_fe.copy(), list(feature_cols)
    else:
        raise ValueError(base_mode)

    lead_features = list(CORE_LEAD_FEATURES)
    if lead_mode == "ops":
        lead_features += OPS_LEAD_FEATURES
    elif lead_mode != "core":
        raise ValueError(lead_mode)

    new_cols = []
    for df in (train_out, test_out):
        grp = df.groupby(cfg.GROUP_COL, sort=False)
        for col in lead_features:
            if col not in df.columns:
                continue
            g = grp[col]
            lead1 = f"{col}_lead1"
            lead2 = f"{col}_lead2"
            delta = f"{col}_lead1_delta"
            fmean2 = f"{col}_fmean2"
            df[lead1] = g.shift(-1)
            df[lead2] = g.shift(-2)
            df[delta] = df[lead1] - df[col]
            df[fmean2] = (df[col] + df[lead1]) / 2.0
            new_cols.extend([lead1, lead2, delta, fmean2])

        if {"order_inflow_15m_lead1", "congestion_score_lead1", "low_battery_ratio_lead1"}.issubset(df.columns):
            df["future_stress_30m"] = (
                df["order_inflow_15m_lead1"].fillna(df["order_inflow_15m"])
                * df["congestion_score_lead1"].fillna(df["congestion_score"])
                * df["low_battery_ratio_lead1"].fillna(df["low_battery_ratio"])
            )
            if "stress_index" in df.columns:
                df["future_stress_delta"] = df["future_stress_30m"] - df["stress_index"]
                new_cols.append("future_stress_delta")
            new_cols.append("future_stress_30m")

        if {"charge_queue_length_lead1", "avg_charge_wait_lead1", "charger_count"}.issubset(df.columns):
            df["future_charge_pressure"] = (
                df["charge_queue_length_lead1"].fillna(df["charge_queue_length"])
                * df["avg_charge_wait_lead1"].fillna(df["avg_charge_wait"])
                / (df["charger_count"].replace(0, np.nan) + 1)
            )
            new_cols.append("future_charge_pressure")

    final_cols = append_unique(cols, new_cols)
    log(
        f"  attack lead features base={base_mode} lead={lead_mode}: "
        f"{len(feature_cols)} -> {len(final_cols)} (+{len(final_cols) - len(feature_cols)})"
    )
    return train_out, test_out, final_cols


def load_strict_seed_bundle(prefix, seeds=(42, 123, 2026)):
    oofs, tests, cvs = [], [], []
    for seed in seeds:
        path = os.path.join(STRICT_CKPT_DIR, f"{prefix}_seed{seed}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            cv, test_pred, oof_pred = pickle.load(f)
        cvs.append(cv)
        tests.append(test_pred)
        oofs.append(oof_pred)
    oof_avg = np.mean(oofs, axis=0)
    test_avg = np.mean(tests, axis=0)
    return cvs, test_avg, oof_avg


def save_submission(test_fe, pred, name):
    sub = pd.DataFrame({
        cfg.ID_COL: test_fe[cfg.ID_COL].astype(str),
        cfg.TARGET: np.clip(pred, 0, None),
    })
    sample = pd.read_csv(cfg.SAMPLE_SUB_FILE)
    if list(sub.columns) != list(sample.columns):
        raise ValueError(f"column mismatch: {list(sub.columns)} != {list(sample.columns)}")
    if len(sub) != len(sample):
        raise ValueError(f"row mismatch: {len(sub)} != {len(sample)}")
    if not sub[cfg.ID_COL].astype(str).equals(sample[cfg.ID_COL].astype(str)):
        raise ValueError("ID order mismatch")
    if sub[cfg.TARGET].isna().any() or (sub[cfg.TARGET] < 0).any():
        raise ValueError("bad predictions")
    path = os.path.join(cfg.SUBMISSION_DIR, f"{name}_submission.csv")
    sub.to_csv(path, index=False, float_format="%.10f", lineterminator="\n")
    log(
        f"  저장: {path} mean={sub[cfg.TARGET].mean():.3f} "
        f"std={sub[cfg.TARGET].std():.3f} max={sub[cfg.TARGET].max():.3f}"
    )
    return path


def save_variants(name, test_fe, y_raw, test_gbdt, oof_gbdt, test_tab, oof_tab):
    out = []

    gbdt_cv = mean_absolute_error(y_raw, oof_gbdt)
    clip_g = np.percentile(oof_gbdt, 99) * 1.30
    gbdt_clip_cv = mean_absolute_error(y_raw, np.clip(oof_gbdt, 0, clip_g))
    out.append({
        "name": f"{name}_gbdt_clip130",
        "cv": gbdt_clip_cv,
        "path": save_submission(test_fe, np.clip(test_gbdt, 0, clip_g), f"{name}-gbdt-clip130"),
    })
    log(f"  {name} GBDT raw CV={gbdt_cv:.4f} clip130 CV={gbdt_clip_cv:.4f} clip={clip_g:.2f}")

    if len(oof_tab) != len(y_raw):
        log(
            f"  TabNet OOF length mismatch ({len(oof_tab)} != {len(y_raw)}); "
            "blend variants skipped. This is expected in sanity mode."
        )
        return out

    alpha, blend_cv = v5.optimize_blend_2way(oof_gbdt, oof_tab, y_raw)
    oof_blend = alpha * oof_gbdt + (1 - alpha) * oof_tab
    test_blend = alpha * test_gbdt + (1 - alpha) * test_tab

    for factor in [1.10, 1.30, None]:
        if factor is None:
            label = "noclip"
            pred = np.clip(test_blend, 0, None)
            cv = mean_absolute_error(y_raw, np.clip(oof_blend, 0, None))
        else:
            label = f"clip{int(factor * 100):03d}"
            clip_val = np.percentile(oof_blend, 99) * factor
            pred = np.clip(test_blend, 0, clip_val)
            cv = mean_absolute_error(y_raw, np.clip(oof_blend, 0, clip_val))
        out.append({
            "name": f"{name}_blend_{label}",
            "cv": cv,
            "path": save_submission(test_fe, pred, f"{name}-blend-{label}"),
            "alpha": alpha,
        })
        log(f"  {name} blend {label}: alpha={alpha:.3f} rawCV={blend_cv:.4f} cv={cv:.4f}")

    return out


def run_experiment(name, train_fe, test_fe, feature_cols, lead_mode, base_mode, sanity=False):
    log("=" * 60)
    log(f"{name}: attack base={base_mode} lead={lead_mode}")
    log("=" * 60)

    train_x, test_x, cols = add_lead_features_split(
        train_fe, test_fe, feature_cols, lead_mode=lead_mode, base_mode=base_mode
    )
    X = train_x[cols]
    X_test = strict.align_test_features(test_x, cols)
    y_raw = train_x[cfg.TARGET].values
    groups = train_x[cfg.GROUP_COL].values

    seeds = [42] if sanity else [42, 123, 2026]
    n_folds = 2 if sanity else cfg.N_FOLDS
    early_stop = 30 if sanity else cfg.EARLY_STOPPING_ROUNDS
    lgb_p, xgb_p, cat_p = strict.gbdt_params(sanity=sanity)

    log("  [1/3] attack GBDT 학습...")
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

    log("  [2/3] strict2 TabNet clean 캐시 로드...")
    cvs_tab, test_tab, oof_tab = load_strict_seed_bundle(
        "strict_2_focused_tabnet", seeds=(42, 123, 2026)
    )
    log(f"  strict2 TabNet seed CV={['%.4f' % c for c in cvs_tab]}")

    log("  [3/3] OOF 기준 후보 저장...")
    variants = save_variants(name, test_fe, y_raw, test_gbdt, oof_gbdt, test_tab, oof_tab)
    return {"name": name, "cv_gbdt": cv_gbdt, "variants": variants}


def parse_exp(arg):
    exps = [
        ("attack_1_core_lead", "core", "focused"),
        ("attack_2_ops_lead", "ops", "focused"),
        ("attack_3_expanded_core_lead", "core", "expanded"),
        ("attack_4_expanded_ops_lead", "ops", "expanded"),
    ]
    if not arg:
        return exps[:2]
    selected = set()
    for part in arg.split(","):
        if "-" in part:
            a, b = part.split("-")
            selected.update(range(int(a), int(b) + 1))
        else:
            selected.add(int(part))
    return [item for i, item in enumerate(exps, start=1) if i in selected]


def main():
    parser = argparse.ArgumentParser(description="strict-clean attack-track experiments")
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--exp", type=str, default=None)
    args = parser.parse_args()

    v5.CKPT_DIR = ATTACK_SANITY_CKPT_DIR if args.sanity else ATTACK_CKPT_DIR
    os.makedirs(v5.CKPT_DIR, exist_ok=True)
    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    log("=" * 60)
    log("strict-clean attack-track 시작")
    if args.sanity:
        log("!! SANITY CHECK 모드")
    log(f"체크포인트: {v5.CKPT_DIR}")
    log("=" * 60)

    train_fe, test_fe, feature_cols = strict.build_strict_data(sanity=args.sanity)
    results = []
    t0 = time.time()
    for name, lead_mode, base_mode in parse_exp(args.exp):
        results.append(run_experiment(
            name, train_fe, test_fe, feature_cols,
            lead_mode=lead_mode,
            base_mode=base_mode,
            sanity=args.sanity,
        ))

    print("\n" + "=" * 60)
    print("attack-track 결과 요약")
    print("=" * 60)
    for res in results:
        print(f"{res['name']:<32} GBDT CV={res['cv_gbdt']:.4f}")
        for var in res["variants"]:
            extra = f" alpha={var['alpha']:.3f}" if "alpha" in var else ""
            print(f"  {var['name']:<36} CV={var['cv']:.4f}{extra} {var['path']}")
    print(f"총 소요: {(time.time() - t0) / 60:.1f}분")
    print("=" * 60)


if __name__ == "__main__":
    main()
