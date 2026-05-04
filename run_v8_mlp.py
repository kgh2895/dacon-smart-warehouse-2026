"""
v8b MLP — 798 피처 row-level TabMLP (3 seeds × 5-fold GroupKFold)
"""
import os, math, time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

import config as cfg
import run_experiments_v5 as v5
from run_v8_tabnet import build_v8_features

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MLP_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_v8_transformer")
V8_CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints_v8")
V8_TAB_CKPT  = os.path.join(BASE_DIR, "checkpoints_v8_tabnet")
import run_experiments_strict as strict
import run_experiments_attack as attack


class TabMLP(nn.Module):
    def __init__(self, n_feat, hidden=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden),   nn.BatchNorm1d(hidden),   nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),   nn.BatchNorm1d(hidden),   nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_mlp_cv(X_train, y_raw, X_test, groups,
               n_folds=5, seeds=(42, 123, 2026),
               hidden=512, dropout=0.2,
               peak_lr=3e-4, warmup=10, max_epochs=200, patience=30,
               batch_size=2048, ckpt_prefix="v8b_mlp"):

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp     = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    n_feat  = X_train.shape[1]

    all_oof, all_test = [], []

    for seed in seeds:
        ckpt_name = f"{ckpt_prefix}_seed{seed}"
        cached = v5.load_ckpt(ckpt_name)
        if cached is not None:
            cv_c, tp_c, op_c = cached
            v5.log(f"  [캐시] seed={seed} CV={cv_c:.4f}")
            all_oof.append(op_c); all_test.append(tp_c)
            continue

        torch.manual_seed(seed)
        gkf     = GroupKFold(n_splits=n_folds)
        oof     = np.zeros(len(y_raw))
        test_acc = np.zeros(len(X_test))
        v5.log(f"  seed={seed}  device={device}")

        X_tr_t  = torch.from_numpy(X_train.astype(np.float32))
        X_te_t  = torch.from_numpy(X_test.astype(np.float32)).to(device)
        y_log   = np.log1p(y_raw).astype(np.float32)

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, groups=groups)):
            ft0 = time.time()
            Xb = X_tr_t[tr_idx].to(device)
            yb = torch.from_numpy(y_log[tr_idx]).to(device)
            Xv = X_tr_t[val_idx].to(device)
            yv_raw = y_raw[val_idx]

            model   = TabMLP(n_feat, hidden, dropout).to(device)
            opt     = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=1e-4)
            def lr_fn(ep):
                if ep < warmup: return ep / max(warmup, 1)
                prog = (ep - warmup) / max(max_epochs - warmup, 1)
                return 0.5 * (1 + math.cos(math.pi * prog))
            sched   = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
            loss_fn = nn.L1Loss()

            best_mae, best_state, p_cnt = float("inf"), None, 0
            for epoch in range(max_epochs):
                model.train()
                perm = torch.randperm(len(Xb), device=device)
                for i in range(0, len(Xb), batch_size):
                    xb_, yb_ = Xb[perm[i:i+batch_size]], yb[perm[i:i+batch_size]]
                    with amp:
                        loss = loss_fn(model(xb_), yb_)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                if (epoch + 1) % 5 == 0:
                    model.eval()
                    with torch.no_grad(), amp:
                        vp = np.expm1(model(Xv).cpu().float().numpy()).clip(0)
                    vm = mean_absolute_error(yv_raw, vp)
                    if vm < best_mae:
                        best_mae, p_cnt = vm, 0
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        p_cnt += 1
                    if p_cnt >= patience // 5:
                        v5.log(f"    early stop ep={epoch+1}  best={best_mae:.4f}")
                        break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad(), amp:
                oof[val_idx]  = np.expm1(model(Xv).cpu().float().numpy()).clip(0)
                test_acc += np.expm1(model(X_te_t).cpu().float().numpy()).clip(0) / n_folds

            v5.log(f"  Fold {fold+1} MAE: {mean_absolute_error(yv_raw, oof[val_idx]):.4f}  ({time.time()-ft0:.0f}s)")

        cv = mean_absolute_error(y_raw, oof)
        v5.log(f"  seed={seed} CV: {cv:.4f}")
        v5.save_ckpt(ckpt_name, (cv, test_acc, oof))
        all_oof.append(oof); all_test.append(test_acc)

    oof_avg  = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)
    cv_avg   = mean_absolute_error(y_raw, oof_avg)
    v5.log(f"  MLP 3-seed avg CV: {cv_avg:.4f}")
    return cv_avg, oof_avg, test_avg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    v5.CKPT_DIR = MLP_CKPT_DIR
    os.makedirs(MLP_CKPT_DIR, exist_ok=True)
    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    v5.log("=" * 60)
    v5.log("v8b MLP [798 피처, row-level, L1Loss]")
    if args.sanity: v5.log("!! SANITY 모드")
    v5.log("=" * 60)
    t0 = time.time()

    X_flat, X_test_flat, y_raw, groups, test_fe = build_v8_features(sanity=args.sanity)
    seeds    = [42] if args.sanity else [42, 123, 2026]
    n_folds  = 2    if args.sanity else 5
    patience = 5    if args.sanity else 30
    max_ep   = 20   if args.sanity else 200

    cv, oof, test_pred = run_mlp_cv(
        X_flat, y_raw, X_test_flat, groups,
        n_folds=n_folds, seeds=seeds,
        peak_lr=3e-4, warmup=10, max_epochs=max_ep, patience=patience,
        batch_size=2048, ckpt_prefix="v8b_mlp",
    )

    if args.sanity:
        v5.log(f"  총 소요: {(time.time()-t0)/60:.1f}분"); return

    # N-way blend
    v5.log("\n  === N-way blend ===")
    model_oofs, model_tests, model_names = [], [], []

    v5.CKPT_DIR = V8_CKPT_DIR
    for name in ["lgb_mae_log", "lgb_huber_log", "cat_mae_log", "lgb_mae_raw", "xgb_mae_raw"]:
        oofs, tests = [], []
        for s in [42, 123, 2026]:
            ck = v5.load_ckpt(f"v8b_{name}_seed{s}") or v5.load_ckpt(f"v8_{name}_seed{s}")
            if ck: _, tp, op = ck; oofs.append(op); tests.append(tp)
        if oofs:
            model_oofs.append(np.mean(oofs, axis=0)); model_tests.append(np.mean(tests, axis=0))
            model_names.append(name)

    v5.CKPT_DIR = strict.STRICT_CKPT_DIR
    s2o, s2t = [], []
    for s in [42, 123, 2026]:
        ck = v5.load_ckpt(f"strict_2_focused_tabnet_seed{s}")
        if ck: _, tp, op = ck; s2o.append(op); s2t.append(tp)
    if s2o:
        model_oofs.append(np.mean(s2o, axis=0)); model_tests.append(np.mean(s2t, axis=0))
        model_names.append("strict2_tabnet")

    v5.CKPT_DIR = V8_TAB_CKPT
    t8o, t8t = [], []
    for s in [42, 123, 2026]:
        ck = v5.load_ckpt(f"v8_tabnet_seed{s}")
        if ck: _, tp, op = ck; t8o.append(op); t8t.append(tp)
    if t8o:
        model_oofs.append(np.mean(t8o, axis=0)); model_tests.append(np.mean(t8t, axis=0))
        model_names.append("v8_tabnet")

    v5.CKPT_DIR = MLP_CKPT_DIR
    from run_v8_transformer import flat_to_seq, seq_to_flat, V8_TF_CKPT_DIR
    v5.CKPT_DIR = V8_TF_CKPT_DIR
    test_groups = test_fe[cfg.GROUP_COL].values
    X_seq, y_seq, train_usc = flat_to_seq(X_flat, y_raw, groups)
    X_test_seq, _, test_usc = flat_to_seq(X_test_flat, None, test_groups)
    tfo, tft = [], []
    for s in [42, 123, 2026]:
        ck = v5.load_ckpt(f"v8b_transformer_v4_seed{s}") or v5.load_ckpt(f"v8_transformer_v4_seed{s}")
        if ck:
            _, oof_seq, test_seq = ck
            if oof_seq.shape[0] == len(train_usc):
                tfo.append(seq_to_flat(oof_seq, groups, train_usc))
                tft.append(seq_to_flat(test_seq, test_groups, test_usc))
    if tfo:
        model_oofs.append(np.mean(tfo, axis=0)); model_tests.append(np.mean(tft, axis=0))
        model_names.append("v8b_transformer_v4")

    model_oofs.append(oof); model_tests.append(test_pred); model_names.append("v8b_mlp")
    weights, blend_cv = v5.optimize_blend_multi(model_oofs, y_raw, model_names)

    test_blend = sum(w * t for w, t in zip(weights, model_tests))
    sub = pd.DataFrame({cfg.ID_COL: test_fe[cfg.ID_COL].astype(str),
                        cfg.TARGET: np.clip(test_blend, 0, None)})
    p = os.path.join(cfg.SUBMISSION_DIR, "v8b_mlp_blend_submission.csv")
    sub.to_csv(p, index=False, float_format="%.10f", lineterminator="\n")

    v5.log(f"\n  MLP CV:      {cv:.4f}")
    v5.log(f"  blend CV:    {blend_cv:.4f}")
    v5.log(f"  저장: {p}")
    v5.log(f"  총 소요: {(time.time()-t0)/60:.1f}분")
    v5.log("=" * 60)


if __name__ == "__main__":
    main()
