"""
v5 실험 마스터 스크립트 — 6개 연속 실행 (Pseudo-labeling 완전 제거)
v5-1: Multi-Seed Baseline (Pseudo 제거)
v5-2: Adversarial Sample Weighting
v5-3: Target Transform Diversity
v5-4: High-Load Regime Features
v5-5: FT-Transformer
v5-6: Grand Ensemble + Distribution Calibration

규칙: test 데이터는 어떠한 형태로도 학습에 활용 불가
"""

import os, sys, time, argparse, traceback, warnings, pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import config as cfg
from dataset import load_raw_data, build_features

warnings.filterwarnings("ignore")

os.makedirs(cfg.LOG_DIR, exist_ok=True)
os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints_v5")  # main() 에서 sanity 시 변경
os.makedirs(CKPT_DIR, exist_ok=True)

# ── 전역 상태 ──────────────────────────────────────────────────────
EXPD_LB = 10.2428
RESULTS = {}
V5_CACHE = {}  # 실험 간 캐시 공유


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Checkpoint 유틸 ────────────────────────────────────────────────
def save_ckpt(name, data):
    path = os.path.join(CKPT_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    log(f"  [ckpt] 저장: {name}")


def load_ckpt(name):
    path = os.path.join(CKPT_DIR, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        log(f"  [ckpt] 로드: {name}")
        return data
    return None


# ── Target Transform ──────────────────────────────────────────────
def transform_target(y_raw, transform="log1p"):
    if transform == "log1p":
        return np.log1p(y_raw)
    elif transform == "sqrt":
        return np.sqrt(np.clip(y_raw, 0, None))
    return y_raw.copy()


def inverse_transform(preds, transform="log1p"):
    if transform == "log1p":
        preds = np.expm1(preds)
    elif transform == "sqrt":
        preds = np.clip(preds, 0, None) ** 2
    return np.clip(preds, 0, None)


# ── 학습 유틸 ─────────────────────────────────────────────────────
def train_lgb(X_tr, y_tr, X_val, y_val, params, early_stop, sample_weight=None):
    m = LGBMRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          sample_weight=sample_weight,
          callbacks=[lgb.early_stopping(early_stop), lgb.log_evaluation(200)])
    return m


def train_xgb(X_tr, y_tr, X_val, y_val, params, early_stop, sample_weight=None):
    p = params.copy()
    p["early_stopping_rounds"] = early_stop
    m = XGBRegressor(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          sample_weight=sample_weight, verbose=200)
    return m


def train_cat(X_tr, y_tr, X_val, y_val, params, sample_weight=None):
    m = CatBoostRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val), sample_weight=sample_weight)
    return m


def save_submission(test_fe, preds, name):
    sample_sub = pd.read_csv(cfg.SAMPLE_SUB_FILE)
    sub = pd.DataFrame({cfg.ID_COL: test_fe[cfg.ID_COL], cfg.TARGET: preds})
    assert len(sub) == len(sample_sub), f"행 수 불일치: {len(sub)} vs {len(sample_sub)}"
    assert sub[cfg.TARGET].isna().sum() == 0, "NaN 존재"
    assert (sub[cfg.TARGET] >= 0).all(), "음수 존재"
    path = os.path.join(cfg.SUBMISSION_DIR, f"{name}_submission.csv")
    sub.to_csv(path, index=False)
    log(f"  저장: {path}  mean={preds.mean():.2f} std={preds.std():.2f} max={preds.max():.2f}")
    return path


# ── Blend 최적화 ──────────────────────────────────────────────────
def optimize_blend_2way(oof_a, oof_b, y_raw):
    def mae_fn(alpha):
        return mean_absolute_error(y_raw, alpha * oof_a + (1 - alpha) * oof_b)
    result = minimize_scalar(mae_fn, bounds=(0.1, 0.95), method="bounded")
    best_a, best_mae = result.x, result.fun
    log(f"  2-way blend -> A:{best_a:.3f} / B:{1 - best_a:.3f}  CV MAE: {best_mae:.4f}")
    return best_a, best_mae


def optimize_blend_multi(oof_list, y_raw, names=None):
    n = len(oof_list)
    if names is None:
        names = [f"M{i}" for i in range(n)]

    def obj(w):
        w_abs = np.abs(w)
        w_norm = w_abs / w_abs.sum()
        pred = sum(w_norm[i] * oof_list[i] for i in range(n))
        return mean_absolute_error(y_raw, pred)

    x0 = np.ones(n) / n
    result = minimize(obj, x0, method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6})
    w = np.abs(result.x)
    w = w / w.sum()
    log(f"  {n}-way blend CV MAE: {result.fun:.4f}")
    for i, name in enumerate(names):
        log(f"    {name}: {w[i]:.3f}")
    return w, result.fun


# ── GBDT CV (pseudo 제거, transform 지원) ─────────────────────────
def run_gbdt_cv(X, y_raw, groups, X_test,
                transform="log1p",
                lgb_params=None, xgb_params=None, cat_params=None,
                weights=None, n_folds=5, early_stop=100,
                ckpt_name=None):
    if ckpt_name:
        cached = load_ckpt(ckpt_name)
        if cached is not None:
            return cached

    lgb_p = (lgb_params or cfg.LGB_PARAMS).copy()
    xgb_p = (xgb_params or cfg.XGB_PARAMS).copy()
    cat_p = (cat_params or cfg.CAT_PARAMS).copy()
    xgb_p["early_stopping_rounds"] = early_stop
    cat_p["early_stopping_rounds"] = early_stop

    y = transform_target(y_raw, transform)

    gkf = GroupKFold(n_splits=n_folds)
    oof_lgb = np.zeros(len(X));  test_lgb = np.zeros(len(X_test))
    oof_xgb = np.zeros(len(X));  test_xgb = np.zeros(len(X_test))
    oof_cat = np.zeros(len(X));  test_cat = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
        fold_ckpt = f"{ckpt_name}_fold{fold}" if ckpt_name else None
        if fold_ckpt:
            fc = load_ckpt(fold_ckpt)
            if fc is not None:
                oof_lgb[val_idx] = fc["oof_lgb"]; test_lgb += fc["test_lgb"] / n_folds
                oof_xgb[val_idx] = fc["oof_xgb"]; test_xgb += fc["test_xgb"] / n_folds
                oof_cat[val_idx] = fc["oof_cat"]; test_cat += fc["test_cat"] / n_folds
                print(f"  Fold {fold+1} ENS MAE: {fc['ens_mae']:.4f}  (캐시)")
                continue

        ft0 = time.time()
        print(f"\n  Fold {fold+1}/{n_folds}  (train={len(tr_idx)}, val={len(val_idx)})")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        y_val_raw = y_raw[val_idx]
        sw = weights[tr_idx] if weights is not None else None

        m_lgb = train_lgb(X_tr, y_tr, X_val, y_val, lgb_p, early_stop, sw)
        oof_lgb[val_idx] = inverse_transform(m_lgb.predict(X_val), transform)
        fold_test_lgb = inverse_transform(m_lgb.predict(X_test), transform)
        test_lgb += fold_test_lgb / n_folds

        m_xgb = train_xgb(X_tr, y_tr, X_val, y_val, xgb_p, early_stop, sw)
        oof_xgb[val_idx] = inverse_transform(m_xgb.predict(X_val), transform)
        fold_test_xgb = inverse_transform(m_xgb.predict(X_test), transform)
        test_xgb += fold_test_xgb / n_folds

        m_cat = train_cat(X_tr, y_tr, X_val, y_val, cat_p, sw)
        oof_cat[val_idx] = inverse_transform(m_cat.predict(X_val), transform)
        fold_test_cat = inverse_transform(m_cat.predict(X_test), transform)
        test_cat += fold_test_cat / n_folds

        w = cfg.ENSEMBLE_WEIGHTS
        p_ens = w[0]*oof_lgb[val_idx] + w[1]*oof_xgb[val_idx] + w[2]*oof_cat[val_idx]
        ens_mae = mean_absolute_error(y_val_raw, p_ens)
        print(f"  Fold {fold+1} ENS MAE: {ens_mae:.4f}  ({time.time()-ft0:.0f}s)")

        if fold_ckpt:
            save_ckpt(fold_ckpt, {
                "oof_lgb": oof_lgb[val_idx], "test_lgb": fold_test_lgb,
                "oof_xgb": oof_xgb[val_idx], "test_xgb": fold_test_xgb,
                "oof_cat": oof_cat[val_idx], "test_cat": fold_test_cat,
                "ens_mae": ens_mae,
            })

    w = cfg.ENSEMBLE_WEIGHTS
    oof_ens = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat
    test_ens = w[0]*test_lgb + w[1]*test_xgb + w[2]*test_cat
    cv_mae = mean_absolute_error(y_raw, oof_ens)

    result = (cv_mae, test_ens, oof_ens)
    if ckpt_name:
        save_ckpt(ckpt_name, result)
    return result


def run_gbdt_multiseed(X, y_raw, groups, X_test, seeds=(42, 123, 2026),
                       ckpt_prefix="gbdt", **kwargs):
    all_oof, all_test = [], []
    for seed in seeds:
        lgb_p = (kwargs.get("lgb_params") or cfg.LGB_PARAMS).copy()
        xgb_p = (kwargs.get("xgb_params") or cfg.XGB_PARAMS).copy()
        cat_p = (kwargs.get("cat_params") or cfg.CAT_PARAMS).copy()
        lgb_p["random_state"] = seed
        xgb_p["random_state"] = seed
        cat_p["random_seed"] = seed

        kw = {k: v for k, v in kwargs.items() if k not in ("lgb_params", "xgb_params", "cat_params")}
        cv, test, oof = run_gbdt_cv(
            X, y_raw, groups, X_test,
            lgb_params=lgb_p, xgb_params=xgb_p, cat_params=cat_p,
            ckpt_name=f"{ckpt_prefix}_seed{seed}", **kw)
        all_oof.append(oof)
        all_test.append(test)
        log(f"  GBDT seed={seed} CV: {cv:.4f}")

    oof_avg = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)
    cv_avg = mean_absolute_error(y_raw, oof_avg)
    log(f"  GBDT {len(seeds)}-seed avg CV: {cv_avg:.4f}")
    return cv_avg, test_avg, oof_avg


# ── TabNet CV (pseudo 제거) ────────────────────────────────────────
def run_tabnet_cv(X_arr, y_raw, groups, X_test_arr,
                  n_d=32, n_a=32, n_steps=5, patience=20, batch_size=4096,
                  n_folds=5, base_seed=42, max_epochs=200,
                  ckpt_name=None):
    if ckpt_name:
        cached = load_ckpt(ckpt_name)
        if cached is not None:
            return cached

    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  TabNet -- device:{device}  n_d={n_d}  n_steps={n_steps}  seed={base_seed}")

    y_log = np.log1p(y_raw).astype(np.float32).reshape(-1, 1)

    gkf = GroupKFold(n_splits=n_folds)
    oof_tab = np.zeros(len(X_arr))
    test_tab = np.zeros(len(X_test_arr))

    fold_ckpt_prefix = f"{ckpt_name}_fold" if ckpt_name else None
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_arr, groups=groups)):
        if fold_ckpt_prefix:
            fold_cached = load_ckpt(f"{fold_ckpt_prefix}_{fold}")
            if fold_cached is not None:
                oof_tab[val_idx] = fold_cached["oof"]
                test_tab += fold_cached["test"] / n_folds
                log(f"  Fold {fold+1} TabNet MAE: {fold_cached['mae']:.4f}  (캐시)")
                continue

        ft0 = time.time()
        X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
        y_tr, y_val = y_log[tr_idx], y_log[val_idx]
        y_val_raw = y_raw[val_idx]

        import torch
        tab = TabNetRegressor(
            n_d=n_d, n_a=n_a, n_steps=n_steps,
            gamma=1.5, n_independent=2, n_shared=2,
            lambda_sparse=1e-4,
            optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
            scheduler_params=dict(step_size=10, gamma=0.9),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax",
            device_name=device, verbose=0,
            seed=base_seed + fold,
        )
        tab.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=["mae"],
                max_epochs=max_epochs, patience=patience,
                batch_size=batch_size, virtual_batch_size=256)

        p = inverse_transform(tab.predict(X_val).flatten(), "log1p")
        test_contrib = inverse_transform(tab.predict(X_test_arr).flatten(), "log1p")
        oof_tab[val_idx] = p
        test_tab += test_contrib / n_folds
        fold_mae = mean_absolute_error(y_val_raw, p)
        log(f"  Fold {fold+1} TabNet MAE: {fold_mae:.4f}  ({time.time()-ft0:.0f}s)")

        if fold_ckpt_prefix:
            save_ckpt(f"{fold_ckpt_prefix}_{fold}", {
                "oof": p, "test": test_contrib, "mae": fold_mae,
            })

    cv_tab = mean_absolute_error(y_raw, oof_tab)
    log(f"  TabNet CV MAE: {cv_tab:.4f}")
    result = (cv_tab, test_tab, oof_tab)
    if ckpt_name:
        save_ckpt(ckpt_name, result)
    return result


def run_tabnet_multiseed(X_arr, y_raw, groups, X_test_arr, seeds=(42, 123, 2026),
                         ckpt_prefix="tabnet", **kwargs):
    all_oof, all_test = [], []
    for seed in seeds:
        cv, test, oof = run_tabnet_cv(
            X_arr, y_raw, groups, X_test_arr,
            base_seed=seed, ckpt_name=f"{ckpt_prefix}_seed{seed}", **kwargs)
        all_oof.append(oof)
        all_test.append(test)

    oof_avg = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)
    cv_avg = mean_absolute_error(y_raw, oof_avg)
    log(f"  TabNet {len(seeds)}-seed avg CV: {cv_avg:.4f}")
    return cv_avg, test_avg, oof_avg


# ── Adversarial Weighting ─────────────────────────────────────────
def compute_adversarial_weights(X_train, X_test, feature_cols, beta=1.0):
    X_all = pd.concat([X_train[feature_cols], X_test[feature_cols]], ignore_index=True)
    y_adv = np.array([0]*len(X_train) + [1]*len(X_test))
    clf = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        verbosity=-1, random_state=42,
    )
    clf.fit(X_all, y_adv)
    p_test = clf.predict_proba(X_train[feature_cols])[:, 1]
    auc = np.mean((p_test > 0.5).astype(int))
    log(f"  Adversarial AUC proxy: {auc:.3f}  (train이 test처럼 보이는 비율)")

    weights = 1.0 + beta * p_test
    log(f"  beta={beta}  weights: mean={weights.mean():.3f} min={weights.min():.3f} max={weights.max():.3f}")
    return weights, p_test


def tune_adversarial_beta(X, y_raw, groups, p_test, betas=(0.5, 1.0, 2.0, 3.0), n_folds=5):
    best_beta, best_cv = 0.0, float("inf")
    lgb_p = cfg.LGB_PARAMS.copy()
    lgb_p["n_estimators"] = 500

    for beta in betas:
        weights = 1.0 + beta * p_test
        y = np.log1p(y_raw)
        gkf = GroupKFold(n_splits=n_folds)
        fold_maes = []
        for _, (tr_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
            m = LGBMRegressor(**lgb_p)
            m.fit(X.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X.iloc[val_idx], y[val_idx])],
                  sample_weight=weights[tr_idx],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(9999)])
            preds = inverse_transform(m.predict(X.iloc[val_idx]), "log1p")
            fold_maes.append(mean_absolute_error(y_raw[val_idx], preds))
        avg = np.mean(fold_maes)
        log(f"  beta={beta:.1f}  avg CV: {avg:.4f}  worst: {max(fold_maes):.4f}")
        if avg < best_cv:
            best_cv, best_beta = avg, beta

    log(f"  최적 beta: {best_beta}")
    return best_beta


# ── High-Load Features (v5-4) ─────────────────────────────────────
def add_highload_features(train_fe, test_fe, feature_cols):
    """고부하 특화 피처 추가. 임계값은 train에서만 산출."""
    train_fe = train_fe.copy()
    test_fe = test_fe.copy()

    # Train 기준 임계값
    thresholds = {
        "order_p75": np.nanpercentile(train_fe["order_inflow_15m"], 75),
        "battery_p90": np.nanpercentile(train_fe["low_battery_ratio"], 90),
        "congestion_p90": np.nanpercentile(train_fe["congestion_score"], 90),
    }
    log(f"  임계값: order_P75={thresholds['order_p75']:.2f}  "
        f"battery_P90={thresholds['battery_p90']:.4f}  congestion_P90={thresholds['congestion_p90']:.2f}")

    for df in [train_fe, test_fe]:
        rt = df["robot_total"].replace(0, np.nan)
        cc = df["charger_count"].replace(0, np.nan)

        # 1. 포화 지표
        df["order_surge"] = (df["order_inflow_15m"] > thresholds["order_p75"]).astype(np.float32)
        df["battery_critical"] = (df["low_battery_ratio"] > thresholds["battery_p90"]).astype(np.float32)
        df["congestion_critical"] = (df["congestion_score"] > thresholds["congestion_p90"]).astype(np.float32)
        df["robot_capacity_used"] = (df["robot_active"] + df["robot_charging"]) / rt

        # 2. 스트레스 상호작용
        df["stress_index"] = df["order_inflow_15m"] * df["congestion_score"] * df["low_battery_ratio"]
        df["bottleneck_score"] = df["max_zone_density"] * (1 - df["robot_idle"] / rt)
        df["recovery_pressure"] = df["charge_queue_length"] * df["avg_charge_wait"] / (cc + 1)
        df["demand_supply_gap"] = df["order_inflow_15m"] - df["robot_active"]
        df["cascade_risk"] = df["fault_count_15m"] * df["congestion_score"] * df["blocked_path_15m"]

        # 3. 레이아웃 용량비
        df["pack_station_per_robot"] = df["pack_station_count"] / rt
        df["charger_per_robot"] = df["charger_count"] / rt

    # 4. 시간적 스트레스 (시나리오 내 변화)
    for df in [train_fe, test_fe]:
        grp = df.groupby(cfg.GROUP_COL)["stress_index"]
        df["stress_acceleration"] = df["stress_index"] - grp.shift(1)
        shifted = grp.shift(1)
        df["sustained_stress"] = shifted.groupby(df[cfg.GROUP_COL]).rolling(
            3, min_periods=1).mean().reset_index(level=0, drop=True)
        sc_max = grp.transform("max").replace(0, np.nan)
        df["peak_stress_ratio"] = df["stress_index"] / sc_max

    new_cols = [
        "order_surge", "battery_critical", "congestion_critical", "robot_capacity_used",
        "stress_index", "bottleneck_score", "recovery_pressure", "demand_supply_gap", "cascade_risk",
        "pack_station_per_robot", "charger_per_robot",
        "stress_acceleration", "sustained_stress", "peak_stress_ratio",
    ]
    extended_cols = feature_cols + new_cols
    log(f"  피처: {len(feature_cols)} -> {len(extended_cols)} (+{len(new_cols)})")
    return train_fe, test_fe, extended_cols


def adversarial_check_features(X_train, X_test, feature_cols, new_cols):
    """신규 피처가 train-test 구분력을 높이는지 확인."""
    X_all_base = pd.concat([X_train[feature_cols], X_test[feature_cols]], ignore_index=True)
    extended = feature_cols + new_cols
    X_all_ext = pd.concat([X_train[extended], X_test[extended]], ignore_index=True)
    y_adv = np.array([0]*len(X_train) + [1]*len(X_test))

    clf_base = LGBMClassifier(n_estimators=300, max_depth=5, verbosity=-1, random_state=42)
    clf_base.fit(X_all_base, y_adv)
    auc_base = clf_base.score(X_all_base, y_adv)

    clf_ext = LGBMClassifier(n_estimators=300, max_depth=5, verbosity=-1, random_state=42)
    clf_ext.fit(X_all_ext, y_adv)
    auc_ext = clf_ext.score(X_all_ext, y_adv)

    log(f"  Adversarial check: base={auc_base:.4f} -> extended={auc_ext:.4f} (diff={auc_ext-auc_base:+.4f})")

    # 개별 신규 피처 중요도 확인
    imp = pd.Series(clf_ext.feature_importances_, index=extended).sort_values(ascending=False)
    problematic = [c for c in new_cols if imp.get(c, 0) > imp.median() * 3]
    if problematic:
        log(f"  경고: 구분력 높은 신규 피처: {problematic}")
    return problematic, auc_ext - auc_base


# ── FT-Transformer ────────────────────────────────────────────────
def build_ft_transformer(n_features, d_model=64, n_heads=4, n_layers=2,
                         d_ff=128, dropout=0.2):
    import torch
    import torch.nn as nn

    class FTTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokenizers = nn.ModuleList([
                nn.Linear(1, d_model) for _ in range(n_features)
            ])
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            tokens = torch.stack([tok(x[:, i:i+1]) for i, tok in enumerate(self.tokenizers)], dim=1)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = self.transformer(tokens)
            return self.head(self.norm(tokens[:, 0])).squeeze(-1)

    return FTTransformer()


def run_ft_transformer_cv(X_arr, y_raw, groups, X_test_arr,
                          n_folds=5, base_seed=42, max_epochs=100, patience=15,
                          batch_size=512, d_model=64, n_heads=4, n_layers=2,
                          ckpt_name=None):
    if ckpt_name:
        cached = load_ckpt(ckpt_name)
        if cached is not None:
            return cached

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    # GPU 메모리 정리
    torch.cuda.empty_cache()
    import gc; gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_arr.shape[1]
    y_log = np.log1p(y_raw).astype(np.float32)

    gkf = GroupKFold(n_splits=n_folds)
    oof_ft = np.zeros(len(X_arr))
    test_ft = np.zeros(len(X_test_arr))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_arr, groups=groups)):
        fold_ckpt = f"{ckpt_name}_fold_{fold}" if ckpt_name else None
        if fold_ckpt:
            fc = load_ckpt(fold_ckpt)
            if fc is not None:
                oof_ft[val_idx] = fc["oof"]
                test_ft += fc["test"] / n_folds
                log(f"  Fold {fold+1} FT MAE: {fc['mae']:.4f}  (캐시)")
                continue

        ft0 = time.time()
        torch.manual_seed(base_seed + fold)
        np.random.seed(base_seed + fold)

        torch.cuda.empty_cache()
        X_tr_t = torch.FloatTensor(X_arr[tr_idx]).to(device)
        y_tr_t = torch.FloatTensor(y_log[tr_idx]).to(device)
        X_val_np = X_arr[val_idx]
        X_val_t = torch.FloatTensor(X_val_np).to(device)

        train_ds = TensorDataset(X_tr_t, y_tr_t)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        model = build_ft_transformer(n_features, d_model=d_model, n_heads=n_heads,
                                     n_layers=n_layers).to(device).to(torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.MSELoss()

        def _batch_predict_np(model_, X_np, bs=batch_size):
            """CPU numpy → GPU batch → CPU numpy"""
            preds = []
            for i in range(0, len(X_np), bs):
                xb = torch.FloatTensor(X_np[i:i+bs]).to(device).to(torch.bfloat16)
                preds.append(model_(xb).float().cpu().numpy())
            return np.concatenate(preds)

        best_mae, best_state, wait = float("inf"), None, 0
        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_dl:
                pred = model(xb.to(torch.bfloat16))
                loss = criterion(pred.float(), yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_pred = _batch_predict_np(model, X_val_np)
            val_mae = mean_absolute_error(y_raw[val_idx], inverse_transform(val_pred, "log1p"))

            if val_mae < best_mae:
                best_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p = inverse_transform(_batch_predict_np(model, X_val_np), "log1p")
            test_contrib = inverse_transform(_batch_predict_np(model, X_test_arr), "log1p")

        oof_ft[val_idx] = p
        test_ft += test_contrib / n_folds
        fold_mae = mean_absolute_error(y_raw[val_idx], p)
        log(f"  Fold {fold+1} FT MAE: {fold_mae:.4f}  ep={epoch+1} ({time.time()-ft0:.0f}s)")

        if fold_ckpt:
            save_ckpt(fold_ckpt, {"oof": p, "test": test_contrib, "mae": fold_mae})

        # GPU 메모리 정리
        del model, X_tr_t, y_tr_t
        torch.cuda.empty_cache()

    cv_ft = mean_absolute_error(y_raw, oof_ft)
    log(f"  FT-Transformer CV MAE: {cv_ft:.4f}")
    result = (cv_ft, test_ft, oof_ft)
    if ckpt_name:
        save_ckpt(ckpt_name, result)
    return result


def run_ft_multiseed(X_arr, y_raw, groups, X_test_arr, seeds=(42, 123, 2026),
                     ckpt_prefix="ft", **kwargs):
    all_oof, all_test = [], []
    for seed in seeds:
        cv, test, oof = run_ft_transformer_cv(
            X_arr, y_raw, groups, X_test_arr,
            base_seed=seed, ckpt_name=f"{ckpt_prefix}_seed{seed}", **kwargs)
        all_oof.append(oof)
        all_test.append(test)

    oof_avg = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)
    cv_avg = mean_absolute_error(y_raw, oof_avg)
    log(f"  FT {len(seeds)}-seed avg CV: {cv_avg:.4f}")
    return cv_avg, test_avg, oof_avg


# ── P99 Clipping ──────────────────────────────────────────────────
def apply_p99_clipping(preds, factor=1.1):
    p99 = np.percentile(preds, 99)
    clip_val = p99 * factor
    n_clip = (preds > clip_val).sum()
    clipped = np.clip(preds, 0, clip_val)
    log(f"  P99 clipping: p99={p99:.2f} clip={clip_val:.2f} n_clip={n_clip}")
    return clipped


# ══════════════════════════════════════════════════════════════════
# ── v5-1: Multi-Seed Baseline ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
def run_v5_1(data):
    log("=" * 60)
    log("v5-1: Multi-Seed GBDT + Multi-Seed TabNet Baseline (NO Pseudo)")
    log("=" * 60)
    t0 = time.time()

    train_fe, test_fe = data["train_fe"], data["test_fe"]
    feature_cols = data["feature_cols"]
    sanity = data.get("sanity", False)
    n_folds = 2 if sanity else 5
    max_ep = 10 if sanity else 200
    seeds = [42] if sanity else [42, 123, 2026]

    X = train_fe[feature_cols]
    y_raw = train_fe[cfg.TARGET].values
    groups = train_fe[cfg.GROUP_COL].values
    X_test = test_fe[feature_cols]

    # Multi-seed GBDT
    log("  [1/3] Multi-seed GBDT...")
    cv_gbdt, test_gbdt, oof_gbdt = run_gbdt_multiseed(
        X, y_raw, groups, X_test, seeds=seeds,
        ckpt_prefix="v5_1_gbdt", n_folds=n_folds)

    # Multi-seed TabNet
    log("  [2/3] Multi-seed TabNet...")
    X_arr = X.fillna(0).values.astype(np.float32)
    Xt_arr = X_test.fillna(0).values.astype(np.float32)
    cv_tab, test_tab, oof_tab = run_tabnet_multiseed(
        X_arr, y_raw, groups, Xt_arr, seeds=seeds,
        ckpt_prefix="v5_1_tabnet", n_folds=n_folds, max_epochs=max_ep)

    # Blend + Post-processing
    log("  [3/3] Blend + 후처리...")
    best_a, cv_ens = optimize_blend_2way(oof_gbdt, oof_tab, y_raw)
    test_ens = best_a * test_gbdt + (1 - best_a) * test_tab
    test_pp = apply_p99_clipping(test_ens)
    save_submission(test_fe, test_pp, "v5-1")

    # 캐시 저장
    V5_CACHE["v1_oof_gbdt"] = oof_gbdt
    V5_CACHE["v1_test_gbdt"] = test_gbdt
    V5_CACHE["v1_oof_tab"] = oof_tab
    V5_CACHE["v1_test_tab"] = test_tab

    log(f"v5-1 완료 -- GBDT:{cv_gbdt:.4f} | Tab:{cv_tab:.4f} | ENS:{cv_ens:.4f} ({(time.time()-t0)/60:.1f}분)")
    return cv_ens


# ══════════════════════════════════════════════════════════════════
# ── v5-2: Adversarial Sample Weighting ───────────────────────────
# ══════════════════════════════════════════════════════════════════
def run_v5_2(data):
    log("=" * 60)
    log("v5-2: Adversarial Sample Weighting")
    log("=" * 60)
    t0 = time.time()

    train_fe, test_fe = data["train_fe"], data["test_fe"]
    feature_cols = data["feature_cols"]
    sanity = data.get("sanity", False)
    n_folds = 2 if sanity else 5
    max_ep = 10 if sanity else 200
    seeds = [42] if sanity else [42, 123, 2026]

    X = train_fe[feature_cols]
    y_raw = train_fe[cfg.TARGET].values
    groups = train_fe[cfg.GROUP_COL].values
    X_test = test_fe[feature_cols]

    # 1) Adversarial weighting
    log("  [1/4] Adversarial weight 계산...")
    _, p_test = compute_adversarial_weights(train_fe, test_fe, feature_cols)

    # 2) Beta 튜닝
    log("  [2/4] Beta 튜닝...")
    betas = [0.5, 1.0] if sanity else [0.5, 1.0, 2.0, 3.0]
    best_beta = tune_adversarial_beta(X, y_raw, groups, p_test, betas=betas, n_folds=n_folds)
    weights = 1.0 + best_beta * p_test

    # 3) Multi-seed GBDT with weights
    log("  [3/4] Weighted multi-seed GBDT...")
    cv_gbdt, test_gbdt, oof_gbdt = run_gbdt_multiseed(
        X, y_raw, groups, X_test, seeds=seeds,
        weights=weights, ckpt_prefix="v5_2_gbdt", n_folds=n_folds)

    # 4) TabNet (oversampling으로 가중치 반영)
    log("  [4/4] Weighted TabNet...")
    X_arr = X.fillna(0).values.astype(np.float32)
    Xt_arr = X_test.fillna(0).values.astype(np.float32)
    # TabNet은 v5-1 캐시 재사용 (sample_weight 미지원이므로 동일)
    if "v1_oof_tab" in V5_CACHE:
        log("  TabNet: v5-1 캐시 재사용")
        oof_tab = V5_CACHE["v1_oof_tab"]
        test_tab = V5_CACHE["v1_test_tab"]
    else:
        _, test_tab, oof_tab = run_tabnet_multiseed(
            X_arr, y_raw, groups, Xt_arr, seeds=seeds,
            ckpt_prefix="v5_2_tabnet", n_folds=n_folds, max_epochs=max_ep)

    best_a, cv_ens = optimize_blend_2way(oof_gbdt, oof_tab, y_raw)
    test_ens = best_a * test_gbdt + (1 - best_a) * test_tab
    test_pp = apply_p99_clipping(test_ens)
    save_submission(test_fe, test_pp, "v5-2")

    V5_CACHE["v2_oof_gbdt"] = oof_gbdt
    V5_CACHE["v2_test_gbdt"] = test_gbdt
    V5_CACHE["adv_weights"] = weights

    log(f"v5-2 완료 -- GBDT:{cv_gbdt:.4f} | ENS:{cv_ens:.4f} beta={best_beta} ({(time.time()-t0)/60:.1f}분)")
    return cv_ens


# ══════════════════════════════════════════════════════════════════
# ── v5-3: Target Transform Diversity ─────────────────────────────
# ══════════════════════════════════════════════════════════════════
def run_v5_3(data):
    log("=" * 60)
    log("v5-3: Target Transform Diversity (log1p + sqrt + Huber)")
    log("=" * 60)
    t0 = time.time()

    train_fe, test_fe = data["train_fe"], data["test_fe"]
    feature_cols = data["feature_cols"]
    sanity = data.get("sanity", False)
    n_folds = 2 if sanity else 5

    X = train_fe[feature_cols]
    y_raw = train_fe[cfg.TARGET].values
    groups = train_fe[cfg.GROUP_COL].values
    X_test = test_fe[feature_cols]

    # Pipeline A: log1p + MAE (v5-1 캐시 재사용)
    log("  [1/3] Pipeline A: log1p + MAE (v5-1 캐시)...")
    if "v1_oof_gbdt" in V5_CACHE:
        oof_a = V5_CACHE["v1_oof_gbdt"]
        test_a = V5_CACHE["v1_test_gbdt"]
        cv_a = mean_absolute_error(y_raw, oof_a)
        log(f"  Pipeline A CV: {cv_a:.4f} (캐시)")
    else:
        cv_a, test_a, oof_a = run_gbdt_cv(
            X, y_raw, groups, X_test, transform="log1p",
            ckpt_name="v5_3_pipe_a", n_folds=n_folds)

    # Pipeline B: sqrt + MAE
    log("  [2/3] Pipeline B: sqrt + MAE...")
    cv_b, test_b, oof_b = run_gbdt_cv(
        X, y_raw, groups, X_test, transform="sqrt",
        ckpt_name="v5_3_pipe_b", n_folds=n_folds)
    log(f"  Pipeline B CV: {cv_b:.4f}")

    # Pipeline C: log1p + Huber
    log("  [3/3] Pipeline C: log1p + Huber...")
    lgb_huber = cfg.LGB_PARAMS.copy()
    lgb_huber["objective"] = "huber"
    lgb_huber["alpha"] = 1.35
    lgb_huber["metric"] = "mae"

    xgb_huber = cfg.XGB_PARAMS.copy()
    xgb_huber["objective"] = "reg:pseudohubererror"
    xgb_huber["huber_slope"] = 1.35

    cat_huber = cfg.CAT_PARAMS.copy()
    cat_huber["loss_function"] = "Huber:delta=1.35"
    cat_huber["eval_metric"] = "MAE"

    cv_c, test_c, oof_c = run_gbdt_cv(
        X, y_raw, groups, X_test, transform="log1p",
        lgb_params=lgb_huber, xgb_params=xgb_huber, cat_params=cat_huber,
        ckpt_name="v5_3_pipe_c", n_folds=n_folds)
    log(f"  Pipeline C CV: {cv_c:.4f}")

    # TabNet (v5-1 캐시)
    oof_tab = V5_CACHE.get("v1_oof_tab")
    test_tab = V5_CACHE.get("v1_test_tab")

    # Multi-way blend
    oof_list = [oof_a, oof_b, oof_c]
    test_list = [test_a, test_b, test_c]
    names = ["log1p+MAE", "sqrt+MAE", "log1p+Huber"]
    if oof_tab is not None:
        oof_list.append(oof_tab)
        test_list.append(test_tab)
        names.append("TabNet")

    w, cv_ens = optimize_blend_multi(oof_list, y_raw, names)
    test_ens = sum(w[i] * test_list[i] for i in range(len(w)))
    test_pp = apply_p99_clipping(test_ens)
    save_submission(test_fe, test_pp, "v5-3")

    V5_CACHE["v3_oof_sqrt"] = oof_b
    V5_CACHE["v3_test_sqrt"] = test_b
    V5_CACHE["v3_oof_huber"] = oof_c
    V5_CACHE["v3_test_huber"] = test_c

    log(f"v5-3 완료 -- A:{cv_a:.4f} B:{cv_b:.4f} C:{cv_c:.4f} | ENS:{cv_ens:.4f} ({(time.time()-t0)/60:.1f}분)")
    return cv_ens


# ══════════════════════════════════════════════════════════════════
# ── v5-4: High-Load Regime Features ──────────────────────────────
# ══════════════════════════════════════════════════════════════════
def run_v5_4(data):
    log("=" * 60)
    log("v5-4: High-Load Regime 특화 피처")
    log("=" * 60)
    t0 = time.time()

    train_fe, test_fe = data["train_fe"], data["test_fe"]
    feature_cols = data["feature_cols"]
    sanity = data.get("sanity", False)
    n_folds = 2 if sanity else 5
    max_ep = 10 if sanity else 200
    seeds = [42] if sanity else [42, 123, 2026]

    # 1) 피처 추가
    log("  [1/4] 고부하 피처 추가...")
    train_hl, test_hl, ext_cols = add_highload_features(train_fe, test_fe, feature_cols)

    X = train_hl[ext_cols]
    y_raw = train_hl[cfg.TARGET].values
    groups = train_hl[cfg.GROUP_COL].values
    X_test = test_hl[ext_cols]

    # 2) Adversarial check
    log("  [2/4] Adversarial check...")
    new_cols = [c for c in ext_cols if c not in feature_cols]
    problematic, auc_diff = adversarial_check_features(train_hl, test_hl, feature_cols, new_cols)

    # 구분력이 크게 높아지는 피처 제거
    if problematic:
        log(f"  제거 피처: {problematic}")
        ext_cols = [c for c in ext_cols if c not in problematic]
        X = train_hl[ext_cols]
        X_test = test_hl[ext_cols]
        log(f"  최종 피처 수: {len(ext_cols)}")

    # 3) Multi-seed GBDT + adversarial weights (v5-2에서 가져오기)
    log("  [3/4] Multi-seed GBDT...")
    weights = V5_CACHE.get("adv_weights")
    cv_gbdt, test_gbdt, oof_gbdt = run_gbdt_multiseed(
        X, y_raw, groups, X_test, seeds=seeds,
        weights=weights, ckpt_prefix="v5_4_gbdt", n_folds=n_folds)

    # 4) Multi-seed TabNet (StandardScaler 적용 — 신규 피처 스케일 차이 보정)
    log("  [4/4] Multi-seed TabNet (scaled)...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_arr = scaler.fit_transform(X.fillna(0)).astype(np.float32)
    Xt_arr = scaler.transform(X_test.fillna(0)).astype(np.float32)
    cv_tab, test_tab, oof_tab = run_tabnet_multiseed(
        X_arr, y_raw, groups, Xt_arr, seeds=seeds,
        ckpt_prefix="v5_4_tabnet_sc", n_folds=n_folds, max_epochs=max_ep)

    best_a, cv_ens = optimize_blend_2way(oof_gbdt, oof_tab, y_raw)
    test_ens = best_a * test_gbdt + (1 - best_a) * test_tab
    test_pp = apply_p99_clipping(test_ens)
    save_submission(test_fe, test_pp, "v5-4")

    V5_CACHE["v4_oof_gbdt"] = oof_gbdt
    V5_CACHE["v4_test_gbdt"] = test_gbdt
    V5_CACHE["v4_oof_tab"] = oof_tab
    V5_CACHE["v4_test_tab"] = test_tab

    log(f"v5-4 완료 -- GBDT:{cv_gbdt:.4f} | Tab:{cv_tab:.4f} | ENS:{cv_ens:.4f} ({(time.time()-t0)/60:.1f}분)")
    return cv_ens


# ══════════════════════════════════════════════════════════════════
# ── v5-5: FT-Transformer ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
def run_v5_5(data):
    log("=" * 60)
    log("v5-5: FT-Transformer 앙상블 다양성")
    log("=" * 60)
    t0 = time.time()

    train_fe, test_fe = data["train_fe"], data["test_fe"]
    feature_cols = data["feature_cols"]
    sanity = data.get("sanity", False)
    n_folds = 2 if sanity else 5
    max_ep = 10 if sanity else 100
    seeds = [42] if sanity else [42, 123, 2026]

    X = train_fe[feature_cols]
    y_raw = train_fe[cfg.TARGET].values
    groups = train_fe[cfg.GROUP_COL].values

    X_arr = X.fillna(0).values.astype(np.float32)
    Xt_arr = test_fe[feature_cols].fillna(0).values.astype(np.float32)

    # Multi-seed FT-Transformer
    log("  [1/2] Multi-seed FT-Transformer...")
    cv_ft, test_ft, oof_ft = run_ft_multiseed(
        X_arr, y_raw, groups, Xt_arr, seeds=seeds,
        ckpt_prefix="v5_5_ft", n_folds=n_folds, max_epochs=max_ep,
        batch_size=512)

    # 3-way blend: GBDT + TabNet + FT-Transformer
    log("  [2/2] 3-way blend...")
    oof_gbdt = V5_CACHE.get("v1_oof_gbdt")
    test_gbdt = V5_CACHE.get("v1_test_gbdt")
    oof_tab = V5_CACHE.get("v1_oof_tab")
    test_tab = V5_CACHE.get("v1_test_tab")

    if oof_gbdt is not None and oof_tab is not None:
        w, cv_ens = optimize_blend_multi(
            [oof_gbdt, oof_tab, oof_ft], y_raw,
            ["GBDT", "TabNet", "FT-Trans"])
        test_ens = w[0]*test_gbdt + w[1]*test_tab + w[2]*test_ft
    else:
        log("  v5-1 캐시 없음 -> FT + GBDT 2-way")
        cv_gbdt_s, test_gbdt_s, oof_gbdt_s = run_gbdt_cv(
            X, y_raw, groups, test_fe[feature_cols],
            ckpt_name="v5_5_gbdt_fallback", n_folds=n_folds)
        best_a, cv_ens = optimize_blend_2way(oof_gbdt_s, oof_ft, y_raw)
        test_ens = best_a * test_gbdt_s + (1 - best_a) * test_ft

    test_pp = apply_p99_clipping(test_ens)
    save_submission(test_fe, test_pp, "v5-5")

    V5_CACHE["v5_oof_ft"] = oof_ft
    V5_CACHE["v5_test_ft"] = test_ft

    log(f"v5-5 완료 -- FT:{cv_ft:.4f} | ENS:{cv_ens:.4f} ({(time.time()-t0)/60:.1f}분)")
    return cv_ens


# ══════════════════════════════════════════════════════════════════
# ── v5-6: Grand Ensemble + Distribution Calibration ──────────────
# ══════════════════════════════════════════════════════════════════
def run_v5_6(data):
    log("=" * 60)
    log("v5-6: Grand Ensemble + Distribution Calibration")
    log("=" * 60)
    t0 = time.time()

    train_fe = data["train_fe"]
    test_fe = data["test_fe"]
    y_raw = train_fe[cfg.TARGET].values
    groups = train_fe[cfg.GROUP_COL].values

    # 수집 가능한 모든 base model
    oof_models, test_models, model_names = [], [], []
    cache_map = [
        ("v1_oof_gbdt", "v1_test_gbdt", "v1-GBDT"),
        ("v1_oof_tab", "v1_test_tab", "v1-TabNet"),
        ("v2_oof_gbdt", "v2_test_gbdt", "v2-GBDT(adv)"),
        ("v3_oof_sqrt", "v3_test_sqrt", "v3-sqrt"),
        ("v3_oof_huber", "v3_test_huber", "v3-Huber"),
        ("v4_oof_gbdt", "v4_test_gbdt", "v4-GBDT(HL)"),
        ("v4_oof_tab", "v4_test_tab", "v4-TabNet(HL)"),
        ("v5_oof_ft", "v5_test_ft", "v5-FT-Trans"),
    ]
    for oof_key, test_key, name in cache_map:
        if oof_key in V5_CACHE and test_key in V5_CACHE:
            oof_models.append(V5_CACHE[oof_key])
            test_models.append(V5_CACHE[test_key])
            model_names.append(name)

    n_models = len(oof_models)
    log(f"  수집된 base models: {n_models}개 — {model_names}")

    if n_models < 2:
        log("  경고: base model 2개 미만 -> 건너뜀")
        return None

    # 1) Nelder-Mead blend
    log("  [1/3] Multi-way Nelder-Mead blend...")
    w_nm, cv_nm = optimize_blend_multi(oof_models, y_raw, model_names)

    # 2) ElasticNet stacking
    log("  [2/3] ElasticNet stacking...")
    meta_tr = np.column_stack(oof_models)
    meta_te = np.column_stack(test_models)

    scaler = StandardScaler()
    X_m = scaler.fit_transform(meta_tr)
    X_mt = scaler.transform(meta_te)

    gkf = GroupKFold(n_splits=5)
    oof_en = np.zeros(len(y_raw))
    test_en = np.zeros(len(meta_te))
    for _, (tr, va) in enumerate(gkf.split(X_m, groups=groups)):
        en = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9], alphas=[0.001, 0.01, 0.1, 1.0],
            cv=3, max_iter=5000, random_state=42)
        en.fit(X_m[tr], y_raw[tr])
        oof_en[va] = en.predict(X_m[va])
        test_en += en.predict(X_mt) / 5
    oof_en = np.clip(oof_en, 0, None)
    test_en = np.clip(test_en, 0, None)
    cv_en = mean_absolute_error(y_raw, oof_en)
    log(f"  ElasticNet CV: {cv_en:.4f}")

    # 최적 선택
    if cv_nm <= cv_en:
        log(f"  Nelder-Mead({cv_nm:.4f}) <= ElasticNet({cv_en:.4f}) -> Nelder-Mead 사용")
        test_final = sum(w_nm[i] * test_models[i] for i in range(n_models))
        oof_final = sum(w_nm[i] * oof_models[i] for i in range(n_models))
        cv_final = cv_nm
    else:
        log(f"  ElasticNet({cv_en:.4f}) < Nelder-Mead({cv_nm:.4f}) -> ElasticNet 사용")
        test_final = test_en
        oof_final = oof_en
        cv_final = cv_en

    # 3) Distribution calibration
    log("  [3/3] Distribution calibration...")
    residuals = y_raw - oof_final
    # Quantile별 잔차 패턴 분석
    q_bins = np.percentile(oof_final, [0, 25, 50, 75, 100])
    corrections = []
    for i in range(4):
        mask = (oof_final >= q_bins[i]) & (oof_final < q_bins[i+1] + (1 if i == 3 else 0))
        if mask.sum() > 100:
            med_resid = np.median(residuals[mask])
            corrections.append((q_bins[i], q_bins[i+1], med_resid, mask.sum()))
            log(f"    Q{i+1} [{q_bins[i]:.1f}-{q_bins[i+1]:.1f}]: "
                f"median resid={med_resid:+.3f}  n={mask.sum()}")

    # 보정 적용 (보수적: 잔차의 50%만 보정)
    test_calibrated = test_final.copy()
    for lo, hi, resid, _ in corrections:
        if abs(resid) > 0.1:  # 의미있는 보정만
            mask_test = (test_final >= lo) & (test_final < hi + (1 if hi == q_bins[-1] else 0))
            test_calibrated[mask_test] += resid * 0.5
            log(f"    보정: [{lo:.1f}-{hi:.1f}] += {resid*0.5:+.3f}")
    test_calibrated = np.clip(test_calibrated, 0, None)

    # 보정 효과 비교 (OOF 기준)
    oof_cal = oof_final.copy()
    for lo, hi, resid, _ in corrections:
        if abs(resid) > 0.1:
            mask_oof = (oof_final >= lo) & (oof_final < hi + (1 if hi == q_bins[-1] else 0))
            oof_cal[mask_oof] += resid * 0.5
    oof_cal = np.clip(oof_cal, 0, None)
    cv_cal = mean_absolute_error(y_raw, oof_cal)
    log(f"  Calibration 효과: {cv_final:.4f} -> {cv_cal:.4f}")

    if cv_cal < cv_final:
        log("  -> Calibration 적용")
        test_out = test_calibrated
        cv_out = cv_cal
    else:
        log("  -> Calibration 효과 없음, 원본 사용")
        test_out = test_final
        cv_out = cv_final

    test_pp = apply_p99_clipping(test_out)
    save_submission(test_fe, test_pp, "v5-6")

    log(f"v5-6 완료 -- NM:{cv_nm:.4f} | EN:{cv_en:.4f} | 최종:{cv_out:.4f} ({(time.time()-t0)/60:.1f}분)")
    return cv_out


# ══════════════════════════════════════════════════════════════════
# ── 메인 ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="v5 실험 (Pseudo-labeling 완전 제거)")
    parser.add_argument("--sanity", action="store_true", help="Sanity check (2-fold, 축소)")
    parser.add_argument("--exp", type=str, default=None,
                        help="특정 실험만 실행 (예: 1, 1-3, 3,5,6)")
    args = parser.parse_args()

    # sanity 모드면 체크포인트 디렉토리 분리
    global CKPT_DIR
    if args.sanity:
        CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints_v5_sanity")
        os.makedirs(CKPT_DIR, exist_ok=True)

    log("=" * 60)
    log("v5 실험 시작 — Pseudo-labeling 완전 제거")
    log(f"EXP-D 기준 (유효 baseline): LB MAE = {EXPD_LB}")
    if args.sanity:
        log("!! SANITY CHECK 모드 (2-fold, 축소 학습)")
    log("=" * 60)

    # 실행할 실험 결정
    all_exps = [
        ("v5-1", run_v5_1),
        ("v5-2", run_v5_2),
        ("v5-3", run_v5_3),
        ("v5-4", run_v5_4),
        ("v5-5", run_v5_5),
        ("v5-6", run_v5_6),
    ]

    if args.exp:
        # "1-3" → [1,2,3], "3,5,6" → [3,5,6]
        selected = set()
        for part in args.exp.split(","):
            if "-" in part:
                a, b = part.split("-")
                selected.update(range(int(a), int(b) + 1))
            else:
                selected.add(int(part))
        experiments = [(n, f) for i, (n, f) in enumerate(all_exps) if (i + 1) in selected]
        log(f"선택 실험: {[n for n, _ in experiments]}")
    else:
        experiments = all_exps

    # 데이터 로딩
    log("데이터 로딩...")
    train_raw, test_raw, layout = load_raw_data()
    train_fe, test_fe, feature_cols = build_features(train_raw, test_raw, layout)
    log(f"피처 수: {len(feature_cols)}")

    data = {
        "train_fe": train_fe,
        "test_fe": test_fe,
        "feature_cols": feature_cols,
        "sanity": args.sanity,
    }

    # 실험 실행
    for name, func in experiments:
        try:
            cv_mae = func(data)
            RESULTS[name] = cv_mae
        except Exception as e:
            log(f"\n{'!'*60}")
            log(f"{name} 실패: {e}")
            traceback.print_exc()
            log("!" * 60)
            RESULTS[name] = None

    # 비교표
    print(f"\n{'='*60}")
    print("  v5 실험 결과 비교")
    print(f"{'='*60}")
    print(f"  {'실험':<16} {'CV MAE':>10} {'vs EXP-D LB':>12}")
    print(f"  {'-'*40}")
    print(f"  {'EXP-D (기준)':<16} {'8.8827':>10} {'LB=10.2428':>12}")
    for name, mae in RESULTS.items():
        if mae is not None:
            print(f"  {name:<16} {mae:>10.4f}")
        else:
            print(f"  {name:<16} {'FAILED':>10}")
    print(f"{'='*60}")
    print(f"\n제출 파일: submissions/v5-{{1..6}}_submission.csv")


if __name__ == "__main__":
    main()
