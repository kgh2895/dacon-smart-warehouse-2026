"""v1 설정 — 스마트 창고 출고 지연 예측"""

import os

# ── 경로 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submissions")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
LAYOUT_FILE = os.path.join(DATA_DIR, "layout_info.csv")
SAMPLE_SUB_FILE = os.path.join(DATA_DIR, "sample_submission.csv")

# ── 컬럼 ──────────────────────────────────────────────
TARGET = "avg_delay_minutes_next_30m"
ID_COL = "ID"
GROUP_COL = "scenario_id"
LAYOUT_KEY = "layout_id"
ID_COLS = [ID_COL, LAYOUT_KEY, GROUP_COL]

# ── 피처 엔지니어링 ──────────────────────────────────
USE_LOG_TARGET = True

LAG_FEATURES = [
    "low_battery_ratio", "battery_mean", "robot_idle",
    "order_inflow_15m", "congestion_score", "robot_charging",
    "max_zone_density",
]
LAG_STEPS = [1, 2]
ROLLING_WINDOWS = [3, 5]

EXPANDING_FEATURES = LAG_FEATURES

SCENARIO_STAT_FEATURES = [
    "order_inflow_15m", "congestion_score", "low_battery_ratio",
    "battery_mean", "robot_utilization",
]

LAYOUT_TYPE_MAP = {"grid": 0, "hub_spoke": 1, "hybrid": 2, "narrow": 3}

# ── 검증 ──────────────────────────────────────────────
SEED = 42
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 100

# ── 모델 ──────────────────────────────────────────────
LGB_PARAMS = {
    "objective": "mae",
    "metric": "mae",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "num_leaves": 127,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "min_child_samples": 50,
    "verbosity": -1,
    "random_state": SEED,
}

XGB_PARAMS = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "min_child_weight": 50,
    "tree_method": "hist",
    "random_state": SEED,
    "verbosity": 0,
}

CAT_PARAMS = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "subsample": 0.7,
    "random_seed": SEED,
    "verbose": 100,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}

ENSEMBLE_WEIGHTS = [0.4, 0.3, 0.3]  # LGB, XGB, CAT

# ── Sanity check ──────────────────────────────────────
SANITY_N_FOLDS = 2
SANITY_N_ESTIMATORS = 200
