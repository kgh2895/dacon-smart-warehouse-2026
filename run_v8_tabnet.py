"""
v8 TabNet — v8 714 피처로 TabNet 학습 후 v8 GBDT blend에 추가.
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
import run_experiments_v5 as v5
import run_experiments_strict as strict
import run_experiments_attack as attack
import run_experiments_v8 as v8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V8_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_v8")
V8_TABNET_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_v8_tabnet")


def log(msg):
    v5.log(msg)


def build_v8_features(sanity=False):
    log("  strict base features 빌드...")
    train_fe, test_fe, feature_cols = strict.build_strict_data(sanity=sanity)

    log("  attack3-style lead features 추가...")
    train_fe, test_fe, feature_cols = attack.add_lead_features_split(
        train_fe, test_fe, feature_cols,
        lead_mode="core", base_mode="expanded",
    )

    log("  PB-style features 추가...")
    train_fe, test_fe, feature_cols = v8.add_pb_style_features(
        train_fe, test_fe, feature_cols,
    )

    X_df = train_fe[feature_cols]
    X_test_df = strict.align_test_features(test_fe, feature_cols)
    y_raw = train_fe[cfg.TARGET].values
    groups = train_fe[cfg.GROUP_COL].values

    # StandardScaler fit on train only (strict-clean), fillna(0) before scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.fillna(0)).astype(np.float32)
    X_test = scaler.transform(X_test_df.fillna(0)).astype(np.float32)
    log(f"  StandardScaler 적용 완료")

    log(f"  피처: {len(feature_cols)}, train={X.shape}, test={X_test.shape}")
    return X, X_test, y_raw, groups, test_fe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    ckpt_dir = V8_TABNET_CKPT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    v5.CKPT_DIR = ckpt_dir

    log("=" * 60)
    log("v8 TabNet — 714 피처 TabNet 학습")
    if args.sanity:
        log("!! SANITY CHECK 모드")
    log("=" * 60)

    t0 = time.time()

    # 1. v8 피처 빌드
    X, X_test, y_raw, groups, test_fe = build_v8_features(sanity=args.sanity)

    # 2. TabNet 학습 (3 seed)
    seeds = [42] if args.sanity else [42, 123, 2026]
    n_folds = 2 if args.sanity else 5
    max_epochs = 50 if args.sanity else 200

    log(f"\n  === TabNet (n_d=32, seeds={seeds}, folds={n_folds}) ===")
    cv_tab, test_tab, oof_tab = v5.run_tabnet_multiseed(
        X, y_raw, groups, X_test,
        seeds=seeds,
        ckpt_prefix="v8_tabnet",
        n_folds=n_folds,
        max_epochs=max_epochs,
    )
    log(f"  v8 TabNet 3-seed avg CV: {cv_tab:.4f}")

    # 3. v8 GBDT OOF 로드해서 함께 blend (full 모드만)
    if args.sanity:
        log("\n  sanity 모드: 블렌드 스킵 (OOF 크기 불일치)")
        log(f"  v8 TabNet CV: {cv_tab:.4f}")
        log(f"  총 소요: {(time.time() - t0) / 60:.1f}분")
        return

    log("\n  === v8 GBDT + v8 TabNet 블렌드 ===")
    v5.CKPT_DIR = V8_CKPT_DIR
    model_oofs, model_tests, model_names = [], [], []

    for name in ["lgb_huber_log", "lgb_mae_log", "cat_mae_log", "lgb_mae_raw", "xgb_mae_raw"]:
        oofs, tests = [], []
        for seed in [42, 123, 2026]:
            ck = v5.load_ckpt(f"v8_{name}_seed{seed}")
            if ck is not None:
                _, tp, op = ck
                oofs.append(op); tests.append(tp)
        if oofs:
            model_oofs.append(np.mean(oofs, axis=0))
            model_tests.append(np.mean(tests, axis=0))
            model_names.append(name)

    # strict2 TabNet (198 피처)
    v5.CKPT_DIR = strict.STRICT_CKPT_DIR
    tab_oofs = []
    for seed in [42, 123, 2026]:
        ck = v5.load_ckpt(f"strict_2_focused_tabnet_seed{seed}")
        if ck is not None:
            _, _, op = ck
            tab_oofs.append(op)
    if tab_oofs:
        model_oofs.append(np.mean(tab_oofs, axis=0))
        model_tests.append(np.mean([v5.load_ckpt(f"strict_2_focused_tabnet_seed{s}")[1]
                                    for s in [42, 123, 2026]], axis=0))
        model_names.append("strict2_tabnet")

    # v8 TabNet (714 피처) 추가
    model_oofs.append(oof_tab)
    model_tests.append(test_tab)
    model_names.append("v8_tabnet")

    weights, blend_cv = v5.optimize_blend_multi(model_oofs, y_raw, model_names)
    log(f"  {len(model_names)}-way blend CV: {blend_cv:.4f}")
    for n, w in zip(model_names, weights):
        log(f"    {n}: {w:.3f}")

    # 4. 제출 파일 생성
    oof_blend = sum(w * o for w, o in zip(weights, model_oofs))
    test_blend = sum(w * t for w, t in zip(weights, model_tests))

    # noclip
    sub = pd.DataFrame({
        cfg.ID_COL: test_fe[cfg.ID_COL].astype(str),
        cfg.TARGET: np.clip(test_blend, 0, None),
    })
    path_noclip = os.path.join(cfg.SUBMISSION_DIR, "v8_tabnet_blend_noclip_submission.csv")
    sub.to_csv(path_noclip, index=False, float_format="%.10f", lineterminator="\n")
    log(f"  저장: {path_noclip}  mean={sub[cfg.TARGET].mean():.3f} std={sub[cfg.TARGET].std():.3f} max={sub[cfg.TARGET].max():.3f}")

    # clip130
    clip_th = np.percentile(oof_blend, 99) * 1.30
    sub_clip = sub.copy()
    sub_clip[cfg.TARGET] = np.clip(test_blend, 0, clip_th)
    path_clip = os.path.join(cfg.SUBMISSION_DIR, "v8_tabnet_blend_clip130_submission.csv")
    sub_clip.to_csv(path_clip, index=False, float_format="%.10f", lineterminator="\n")
    cv_clip = mean_absolute_error(y_raw, np.clip(oof_blend, 0, clip_th))
    log(f"  저장: {path_clip}  clip_th={clip_th:.2f} CV_clip={cv_clip:.4f}")

    log("\n" + "=" * 60)
    log("결과 요약")
    log("=" * 60)
    log(f"  v8 TabNet CV: {cv_tab:.4f}")
    log(f"  blend noclip CV: {blend_cv:.4f}")
    log(f"  blend clip130 CV: {cv_clip:.4f}")
    log(f"  총 소요: {(time.time() - t0) / 60:.1f}분")
    log("=" * 60)


if __name__ == "__main__":
    main()
