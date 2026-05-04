"""데이터 로드 + 피처 엔지니어링 파이프라인"""

import pandas as pd
import numpy as np
import config as cfg


def load_raw_data():
    train = pd.read_csv(cfg.TRAIN_FILE)
    test = pd.read_csv(cfg.TEST_FILE)
    layout = pd.read_csv(cfg.LAYOUT_FILE)
    return train, test, layout


def merge_layout(df: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
    layout = layout.copy()
    layout["layout_type_enc"] = layout["layout_type"].map(cfg.LAYOUT_TYPE_MAP)
    layout = layout.drop(columns=["layout_type"])
    df = df.merge(layout, on=cfg.LAYOUT_KEY, how="left")
    return df


def add_timestep(df: pd.DataFrame) -> pd.DataFrame:
    df["timestep"] = df.groupby(cfg.GROUP_COL).cumcount()
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for feat in cfg.LAG_FEATURES:
        grp = df.groupby(cfg.GROUP_COL)[feat]
        for lag in cfg.LAG_STEPS:
            df[f"{feat}_lag{lag}"] = grp.shift(lag)
        df[f"{feat}_diff1"] = df[feat] - df[f"{feat}_lag1"]
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    for feat in cfg.LAG_FEATURES:
        grp = df.groupby(cfg.GROUP_COL)[feat]
        for w in cfg.ROLLING_WINDOWS:
            shifted = grp.shift(1)
            rolling = shifted.groupby(df[cfg.GROUP_COL]).rolling(w, min_periods=1)
            df[f"{feat}_rmean{w}"] = rolling.mean().reset_index(level=0, drop=True)
            df[f"{feat}_rstd{w}"] = rolling.std().reset_index(level=0, drop=True)
    return df


def add_expanding_features(df: pd.DataFrame) -> pd.DataFrame:
    for feat in cfg.EXPANDING_FEATURES:
        shifted = df.groupby(cfg.GROUP_COL)[feat].shift(1)
        expanding = shifted.groupby(df[cfg.GROUP_COL]).expanding(min_periods=1)
        df[f"{feat}_exp_mean"] = expanding.mean().reset_index(level=0, drop=True)
        df[f"{feat}_exp_std"] = expanding.std().reset_index(level=0, drop=True)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["battery_x_congestion"] = df["low_battery_ratio"] * df["congestion_score"]
    df["inflow_x_utilization"] = df["order_inflow_15m"] * df["robot_utilization"]
    df["charging_x_demand"] = df["robot_charging"] * df["order_inflow_15m"]
    df["idle_x_inflow"] = df["robot_idle"] * df["order_inflow_15m"]
    df["battery_per_congestion"] = df["battery_mean"] / (df["congestion_score"] + 1)

    # layout 기반 비율
    rt = df["robot_total"].replace(0, np.nan)
    df["active_ratio"] = df["robot_active"] / rt
    df["charging_ratio"] = df["robot_charging"] / rt
    df["idle_ratio"] = df["robot_idle"] / rt

    cc = df["charger_count"].replace(0, np.nan)
    df["charger_utilization"] = df["robot_charging"] / cc
    df["orders_per_robot"] = df["order_inflow_15m"] / rt
    return df


def add_scenario_stats(df: pd.DataFrame) -> pd.DataFrame:
    for feat in cfg.SCENARIO_STAT_FEATURES:
        grp = df.groupby(cfg.GROUP_COL)[feat]
        sc_mean = grp.transform("mean")
        df[f"{feat}_sc_mean"] = sc_mean
        df[f"{feat}_sc_std"] = grp.transform("std")
        df[f"{feat}_sc_max"] = grp.transform("max")
        df[f"{feat}_dev_from_sc"] = df[feat] - sc_mean
    return df


def build_features(train: pd.DataFrame, test: pd.DataFrame, layout: pd.DataFrame):
    """메인 피처 빌드. (train_fe, test_fe, feature_cols) 반환."""
    train = train.copy()
    test = test.copy()

    train["_is_train"] = 1
    test["_is_train"] = 0
    if cfg.TARGET not in test.columns:
        test[cfg.TARGET] = np.nan

    df = pd.concat([train, test], ignore_index=True)

    df = merge_layout(df, layout)
    df = add_timestep(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_expanding_features(df)
    df = add_interaction_features(df)
    df = add_scenario_stats(df)

    # 피처 컬럼 결정: ID/메타/타깃/_is_train 제외
    exclude = set(cfg.ID_COLS + [cfg.TARGET, "_is_train"])
    feature_cols = [c for c in df.columns if c not in exclude]

    train_fe = df[df["_is_train"] == 1].reset_index(drop=True)
    test_fe = df[df["_is_train"] == 0].reset_index(drop=True)
    train_fe = train_fe.drop(columns=["_is_train"])
    test_fe = test_fe.drop(columns=["_is_train"])

    print(f"[dataset] 피처 수: {len(feature_cols)}")
    print(f"[dataset] train: {train_fe.shape}, test: {test_fe.shape}")
    return train_fe, test_fe, feature_cols


if __name__ == "__main__":
    train, test, layout = load_raw_data()
    train_fe, test_fe, feature_cols = build_features(train, test, layout)
    print(f"\n피처 목록 ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  {c}")
    print(f"\ntrain NaN 비율: {train_fe[feature_cols].isnull().mean().mean():.4f}")
    print(f"test  NaN 비율: {test_fe[feature_cols].isnull().mean().mean():.4f}")
