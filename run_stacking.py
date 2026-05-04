"""
Stacking meta-learner: OOF 8개 → LGB 메타모델
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

import config as cfg
import run_experiments_v5 as v5
import run_experiments_strict as strict
from run_v8_tabnet import build_v8_features
from run_v8_transformer import flat_to_seq, seq_to_flat, V8_TF_CKPT_DIR, V8_CKPT_DIR, V8_TAB_CKPT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def collect_oofs(groups, test_groups, train_usc, test_usc):
    model_oofs, model_tests, model_names = [], [], []

    # GBDT — v8b(시나리오 피처 추가) 있으면 우선 사용, 없으면 v8 폴백
    v5.CKPT_DIR = V8_CKPT_DIR
    for name in ["lgb_mae_log", "lgb_huber_log", "cat_mae_log", "lgb_mae_raw", "xgb_mae_raw"]:
        oofs, tests = [], []
        for seed in [42, 123, 2026]:
            ck = v5.load_ckpt(f"v8b_{name}_seed{seed}") or v5.load_ckpt(f"v8_{name}_seed{seed}")
            if ck is not None:
                _, tp, op = ck
                oofs.append(op); tests.append(tp)
        if oofs:
            model_oofs.append(np.mean(oofs, axis=0))
            model_tests.append(np.mean(tests, axis=0))
            model_names.append(name)

    # strict2 TabNet
    v5.CKPT_DIR = strict.STRICT_CKPT_DIR
    s2_oofs, s2_tests = [], []
    for seed in [42, 123, 2026]:
        ck = v5.load_ckpt(f"strict_2_focused_tabnet_seed{seed}")
        if ck is not None:
            _, tp, op = ck
            s2_oofs.append(op); s2_tests.append(tp)
    if s2_oofs:
        model_oofs.append(np.mean(s2_oofs, axis=0))
        model_tests.append(np.mean(s2_tests, axis=0))
        model_names.append("strict2_tabnet")

    # v8 TabNet
    v5.CKPT_DIR = V8_TAB_CKPT
    t8_oofs, t8_tests = [], []
    for seed in [42, 123, 2026]:
        ck = v5.load_ckpt(f"v8_tabnet_seed{seed}")
        if ck is not None:
            _, tp, op = ck
            t8_oofs.append(op); t8_tests.append(tp)
    if t8_oofs:
        model_oofs.append(np.mean(t8_oofs, axis=0))
        model_tests.append(np.mean(t8_tests, axis=0))
        model_names.append("v8_tabnet")

    # Transformer v4 — v8b(798 피처) 있으면 우선, 없으면 v8 폴백
    v5.CKPT_DIR = V8_TF_CKPT_DIR
    tf_oofs, tf_tests = [], []
    for seed in [42, 123, 2026]:
        ck = v5.load_ckpt(f"v8b_transformer_v4_seed{seed}") or v5.load_ckpt(f"v8_transformer_v4_seed{seed}")
        if ck is not None:
            _, oof_seq, test_seq = ck
            if oof_seq.shape[0] == len(train_usc):
                tf_oofs.append(seq_to_flat(oof_seq, groups, train_usc))
                tf_tests.append(seq_to_flat(test_seq, test_groups, test_usc))
    if tf_oofs:
        model_oofs.append(np.mean(tf_oofs, axis=0))
        model_tests.append(np.mean(tf_tests, axis=0))
        model_names.append("v8_transformer_v4")

    # MLP
    mlp_oofs, mlp_tests = [], []
    for seed in [42, 123, 2026]:
        ck = v5.load_ckpt(f"v8b_mlp_seed{seed}")
        if ck is not None:
            _, tp, op = ck
            mlp_oofs.append(op); mlp_tests.append(tp)
    if mlp_oofs:
        model_oofs.append(np.mean(mlp_oofs, axis=0))
        model_tests.append(np.mean(mlp_tests, axis=0))
        model_names.append("v8b_mlp")

    # CNN + v5 transformer (sequence → flat)
    for ckpt_key, label in [("v8b_cnn", "v8b_cnn"), ("v8b_transformer_v5", "v8b_tf_v5")]:
        seq_oofs, seq_tests = [], []
        for seed in [42, 123, 2026]:
            ck = v5.load_ckpt(f"{ckpt_key}_seed{seed}")
            if ck is not None:
                _, oof_seq, test_seq = ck
                if oof_seq.shape[0] == len(train_usc):
                    seq_oofs.append(seq_to_flat(oof_seq, groups, train_usc))
                    seq_tests.append(seq_to_flat(test_seq, test_groups, test_usc))
        if seq_oofs:
            model_oofs.append(np.mean(seq_oofs, axis=0))
            model_tests.append(np.mean(seq_tests, axis=0))
            model_names.append(label)

    return model_oofs, model_tests, model_names


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1", help="제출 파일 버전 태그 (예: v1, v2)")
    parser.add_argument("--method", default="lgb", choices=["lgb", "ridge", "blend"], help="메타러너 종류")
    args = parser.parse_args()

    v5.log("=" * 60)
    v5.log(f"Stacking meta-learner [tag={args.tag}, method={args.method}]")
    v5.log("=" * 60)

    # 1. 피처/그룹 로드 (OOF 정렬용)
    X_flat, X_test_flat, y_raw, groups, test_fe = build_v8_features()
    test_groups = test_fe[cfg.GROUP_COL].values
    X_seq, y_seq, train_usc = flat_to_seq(X_flat, y_raw, groups)
    X_test_seq, _, test_usc = flat_to_seq(X_test_flat, None, test_groups)

    # 2. OOF 수집
    model_oofs, model_tests, model_names = collect_oofs(groups, test_groups, train_usc, test_usc)
    v5.log(f"수집된 모델 ({len(model_names)}개): {model_names}")

    # 블렌드 CV 기준선
    weights, blend_cv = v5.optimize_blend_multi(model_oofs, y_raw, model_names)
    v5.log(f"Blend CV (기준선): {blend_cv:.4f}")

    # 3. 메타 피처 구성
    X_meta      = np.column_stack(model_oofs)
    X_test_meta = np.column_stack(model_tests)

    # 4. 메타러너 (method에 따라 분기)
    gkf      = GroupKFold(n_splits=5)
    oof_meta = np.zeros(len(y_raw))
    test_meta = np.zeros(len(test_groups))

    if args.method == "blend":
        oof_meta  = sum(w * o for w, o in zip(weights, model_oofs))
        test_meta = sum(w * t for w, t in zip(weights, model_tests))

    elif args.method == "ridge":
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_meta, groups=groups)):
            sc = StandardScaler()
            X_tr = sc.fit_transform(np.log1p(X_meta[tr_idx]))
            X_val = sc.transform(np.log1p(X_meta[val_idx]))
            y_tr = np.log1p(y_raw[tr_idx])

            model = Ridge(alpha=10.0)
            model.fit(X_tr, y_tr)

            oof_meta[val_idx] = np.expm1(model.predict(X_val)).clip(0)
            test_meta += np.expm1(model.predict(sc.transform(np.log1p(X_test_meta)))).clip(0) / gkf.n_splits

            fold_mae = mean_absolute_error(y_raw[val_idx], oof_meta[val_idx])
            v5.log(f"  Fold {fold+1} MAE: {fold_mae:.4f}")

    else:  # lgb
        params = {
            "objective": "mae", "metric": "mae",
            "num_leaves": 15, "learning_rate": 0.05,
            "min_child_samples": 100, "subsample": 0.8,
            "colsample_bytree": 1.0, "verbose": -1, "seed": 42,
        }
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_meta, groups=groups)):
            X_tr, y_tr = X_meta[tr_idx], np.log1p(y_raw[tr_idx])
            X_val, y_val = X_meta[val_idx], np.log1p(y_raw[val_idx])
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            model = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof_meta[val_idx] = np.expm1(model.predict(X_meta[val_idx]))
            test_meta += np.expm1(model.predict(X_test_meta)) / gkf.n_splits
            fold_mae = mean_absolute_error(y_raw[val_idx], oof_meta[val_idx])
            v5.log(f"  Fold {fold+1} MAE: {fold_mae:.4f}  (best_iter={model.best_iteration})")

    meta_cv = mean_absolute_error(y_raw, oof_meta)
    v5.log(f"\n{args.method} CV: {meta_cv:.4f}")
    v5.log(f"Blend CV (기준): {blend_cv:.4f}")
    v5.log(f"개선:             {blend_cv - meta_cv:+.4f}")

    # 5. 제출 파일 저장
    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)
    sub = pd.DataFrame({
        cfg.ID_COL: test_fe[cfg.ID_COL].astype(str),
        cfg.TARGET: np.clip(test_meta, 0, None),
    })
    p = os.path.join(cfg.SUBMISSION_DIR, f"v8_stacking_{args.tag}_submission.csv")
    sub.to_csv(p, index=False, float_format="%.10f", lineterminator="\n")
    v5.log(f"저장: {p}  mean={sub[cfg.TARGET].mean():.3f}  std={sub[cfg.TARGET].std():.3f}")
    v5.log("=" * 60)


if __name__ == "__main__":
    main()
