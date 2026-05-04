"""
v8 Strong Transformer — 25 timestep 시퀀스 모델.
(batch, 25, 714) 입력, pre-norm Transformer Encoder, learnable positional embedding.
"""
import argparse, math, os, time
from contextlib import nullcontext

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

import config as cfg
import run_experiments_v5 as v5
import run_experiments_strict as strict
from sklearn.preprocessing import StandardScaler
from run_v8_tabnet import build_v8_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V8_CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints_v8")
V8_TAB_CKPT    = os.path.join(BASE_DIR, "checkpoints_v8_tabnet")
V8_TF_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_v8_transformer")


def build_base_features(sanity=False):
    """198 base features only — lag/rolling 없이 시퀀스 모델이 직접 시간 패턴 학습"""
    train_fe, test_fe, feature_cols = strict.build_strict_data(sanity=sanity)
    X_df      = train_fe[feature_cols]
    X_test_df = strict.align_test_features(test_fe, feature_cols)
    y_raw     = train_fe[cfg.TARGET].values
    groups    = train_fe[cfg.GROUP_COL].values
    scaler    = StandardScaler()
    X         = scaler.fit_transform(X_df.fillna(0)).astype(np.float32)
    X_test    = scaler.transform(X_test_df.fillna(0)).astype(np.float32)
    v5.log(f"  base features: {len(feature_cols)}, train={X.shape}, test={X_test.shape}")
    return X, X_test, y_raw, groups, test_fe
N_TS = 25


# ── Models ────────────────────────────────────────────────────────────
class Conv1DNet(nn.Module):
    """Multi-scale 1D CNN — 인접 timestep 로컬 패턴 학습 (kernel 3+5)"""
    def __init__(self, n_feat, channels=128, dropout=0.2, **_):
        super().__init__()
        self.proj  = nn.Linear(n_feat, channels)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x):              # x: (B, 25, n_feat)
        h = self.proj(x)               # (B, 25, C)
        ht = h.transpose(1, 2)         # (B, C, 25)
        h = self.norm1(h + self.conv3(ht).transpose(1, 2))
        h = self.norm2(h + self.conv5(h.transpose(1, 2)).transpose(1, 2))
        return self.head(self.drop(h)).squeeze(-1)  # (B, 25)


class BiLSTMAttention(nn.Module):
    def __init__(self, n_feat, hidden=256, n_layers=2, n_heads=8, dropout=0.2, **_):
        super().__init__()
        d = hidden * 2  # bidirectional output dim
        self.proj = nn.Linear(n_feat, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 2, d))
        self.norm2 = nn.LayerNorm(d)
        self.head  = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):              # x: (B, 25, n_feat)
        h, _ = self.lstm(self.proj(x)) # (B, 25, hidden*2)
        a, _ = self.attn(h, h, h)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ffn(h))
        return self.head(h).squeeze(-1)  # (B, 25)


class StrongTransformer(nn.Module):
    def __init__(self, n_feat, d_model=256, n_heads=8, n_layers=6, d_ff=1024, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos  = nn.Embedding(N_TS, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout,
            batch_first=True, norm_first=True)   # pre-LayerNorm
        self.enc  = nn.TransformerEncoder(enc, n_layers)
        # 2-layer MLP head with GELU
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):          # x: (B, 25, n_feat)
        pos = torch.arange(x.shape[1], device=x.device)
        h0 = self.proj(x) + self.pos(pos)
        return self.head(self.enc(h0) + h0).squeeze(-1)  # skip conn + (B, 25)


# ── Data helpers ───────────────────────────────────────────────────────
def flat_to_seq(X_flat, y_flat, groups):
    """row-level → (n_sc, N_TS, n_feat), (n_sc, N_TS)|None, unique_sc"""
    unique_sc, sc_to_i = [], {}
    for sc in groups:
        if sc not in sc_to_i:
            sc_to_i[sc] = len(unique_sc)
            unique_sc.append(sc)
    n_sc = len(unique_sc)
    X_seq = np.zeros((n_sc, N_TS, X_flat.shape[1]), dtype=np.float32)
    y_seq = np.zeros((n_sc, N_TS), dtype=np.float32) if y_flat is not None else None
    cnt = np.zeros(n_sc, dtype=np.int32)
    for r, sc in enumerate(groups):
        i = sc_to_i[sc]
        X_seq[i, cnt[i]] = X_flat[r]
        if y_seq is not None:
            y_seq[i, cnt[i]] = y_flat[r]
        cnt[i] += 1
    return X_seq, y_seq, np.array(unique_sc)


def seq_to_flat(pred_seq, groups, unique_sc):
    """(n_sc, N_TS) → row-level (N,) in original row order"""
    sc_to_i = {sc: i for i, sc in enumerate(unique_sc)}
    out, cnt = np.zeros(len(groups)), {}
    for r, sc in enumerate(groups):
        p = cnt.get(sc, 0)
        out[r] = pred_seq[sc_to_i[sc], p]
        cnt[sc] = p + 1
    return out


# ── CV training ────────────────────────────────────────────────────────
def run_transformer_cv_seq(X_seq, y_seq, X_test_seq, scenario_ids,
                            model_cls=StrongTransformer, model_kwargs=None,
                            n_folds=5, base_seed=42, max_epochs=200,
                            batch_size=512, patience=30, peak_lr=3e-4, warmup=10,
                            ckpt_name=None, loss_fn=None):
    if ckpt_name:
        cached = v5.load_ckpt(ckpt_name)
        if cached is not None:
            cv_c, oof_c, test_c = cached
            if oof_c.shape[0] == len(X_seq) and test_c.shape[0] == len(X_test_seq):
                return cached
            v5.log(f"  {ckpt_name}: 크기 불일치 ({oof_c.shape[0]} vs {len(X_seq)}), 재학습")

    if model_kwargs is None:
        model_kwargs = {}
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_sc, _, n_feat = X_seq.shape
    use_amp = device.type == "cuda"
    v5.log(f"  {model_cls.__name__}  device:{device}  kwargs={model_kwargs}  seed={base_seed}")
    torch.manual_seed(base_seed)

    gkf       = GroupKFold(n_splits=n_folds)
    oof_seq   = np.zeros((n_sc, N_TS))
    test_seq  = np.zeros((len(X_test_seq), N_TS))
    fold_pfx  = f"{ckpt_name}_fold" if ckpt_name else None
    X_test_t  = torch.from_numpy(X_test_seq).to(device)
    amp       = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(n_sc), groups=scenario_ids)):
        if fold_pfx:
            fc = v5.load_ckpt(f"{fold_pfx}_{fold}")
            if fc is not None:
                if fc["oof"].shape[0] == len(val_idx):
                    oof_seq[val_idx] = fc["oof"]
                    test_seq += fc["test"] / n_folds
                    v5.log(f"  Fold {fold+1} MAE: {fc['mae']:.4f}  (캐시)")
                    continue
                v5.log(f"  Fold {fold+1} 캐시 크기 불일치, 재학습")

        ft0     = time.time()
        X_tr    = torch.from_numpy(X_seq[tr_idx]).to(device)
        y_tr    = torch.from_numpy(np.log1p(y_seq[tr_idx]).astype(np.float32)).to(device)
        X_val_t = torch.from_numpy(X_seq[val_idx]).to(device)
        y_val_r = y_seq[val_idx]

        model = model_cls(n_feat, **model_kwargs).to(device)
        wd    = model_kwargs.get("weight_decay", 1e-4)
        opt   = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=wd)
        def lr_lambda(ep):
            if ep < warmup:
                return ep / max(warmup, 1)
            prog = (ep - warmup) / max(max_epochs - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * prog))
        sched   = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        loss_fn = nn.HuberLoss(delta=0.5)

        best_mae, best_state, p_cnt = float("inf"), None, 0
        _loss_fn = loss_fn if loss_fn is not None else nn.HuberLoss(delta=0.5)

        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(len(X_tr), device=device)
            for i in range(0, len(X_tr), batch_size):
                xb, yb = X_tr[perm[i:i+batch_size]], y_tr[perm[i:i+batch_size]]
                with amp:
                    loss = _loss_fn(model(xb), yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad(), amp:
                    vp = np.expm1(model(X_val_t).cpu().float().numpy()).clip(0)
                vm = mean_absolute_error(y_val_r.flatten(), vp.flatten())
                if vm < best_mae:
                    best_mae, p_cnt = vm, 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    p_cnt += 1
                if p_cnt >= patience // 5:
                    v5.log(f"    early stop ep={epoch+1}  best={best_mae:.4f}")
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad(), amp:
            oof_pred  = np.expm1(model(X_val_t).cpu().float().numpy()).clip(0)
            test_pred = np.expm1(model(X_test_t).cpu().float().numpy()).clip(0)

        oof_seq[val_idx]  = oof_pred
        test_seq         += test_pred / n_folds
        fold_mae = mean_absolute_error(y_val_r.flatten(), oof_pred.flatten())
        v5.log(f"  Fold {fold+1} MAE: {fold_mae:.4f}  ({time.time()-ft0:.0f}s)")
        if fold_pfx:
            v5.save_ckpt(f"{fold_pfx}_{fold}", {"oof": oof_pred, "test": test_pred, "mae": fold_mae})

    cv = mean_absolute_error(y_seq.flatten(), oof_seq.flatten())
    v5.log(f"  Transformer CV: {cv:.4f}")
    result = (cv, oof_seq, test_seq)
    if ckpt_name:
        v5.save_ckpt(ckpt_name, result)
    return result


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--v2", action="store_true", help="warmup+cosine, 6layers, patience30, 200ep")
    parser.add_argument("--v3", action="store_true", help="v2 + dropout0.3, wd5e-4, lr1e-4, patience50, 300ep")
    parser.add_argument("--v4", action="store_true", help="v2 구조 + lr=2e-4, patience35, ep250, L1Loss")
    parser.add_argument("--v5", action="store_true", help="소형: d=128, L=4, lr=3e-4, L1Loss")
    parser.add_argument("--cnn", action="store_true", help="1D CNN: channels=128, kernel 3+5, L1Loss")
    parser.add_argument("--lstm", action="store_true", help="BiLSTM+Attention, hidden=256, 2layers")
    parser.add_argument("--base", action="store_true", help="198 base features only (lag없는 원시 피처 → LSTM이 시간패턴 학습)")
    args = parser.parse_args()

    os.makedirs(V8_TF_CKPT_DIR, exist_ok=True)
    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    v5.CKPT_DIR = V8_TF_CKPT_DIR

    tag = "base_lstm" if args.base else ("lstm" if args.lstm else ("cnn" if args.cnn else ("v5" if args.v5 else ("v4" if args.v4 else ("v3" if args.v3 else ("v2" if args.v2 else "v1"))))))
    v5.log("=" * 60)
    v5.log(f"v8 Sequence Model [{tag}]")
    if args.sanity: v5.log("!! SANITY 모드")
    if args.base: v5.log("  Base 198 피처 BiLSTM+Attention (lag/rolling 없음 — 시퀀스가 직접 학습)")
    elif args.lstm: v5.log("  BiLSTM+Attention: hidden=256 layers=2 heads=8 / lr=3e-4 warmup=10 patience=30")
    elif args.cnn: v5.log("  1D CNN: channels=128 kernel=3+5 / lr=3e-4 warmup=10 dropout=0.2 patience=30 L1Loss")
    elif args.v5: v5.log("  v5: d=128 L=4 heads=4 / lr=3e-4 warmup=10 dropout=0.2 patience=30 ep=200 L1Loss")
    elif args.v4: v5.log("  v4: lr=2e-4 / warmup=10 / dropout=0.2 / patience=35 / ep=250 / L1Loss")
    elif args.v3: v5.log("  v3: lr=1e-4 / warmup=20 / dropout=0.3 / wd=5e-4 / patience=50 / ep=300")
    elif args.v2: v5.log("  v2: lr=3e-4 / warmup=10 / dropout=0.2 / patience=30 / ep=200")
    v5.log("=" * 60)
    t0 = time.time()

    # 1. 피처 빌드
    if args.base:
        X_flat, X_test_flat, y_raw, groups, test_fe = build_base_features(sanity=args.sanity)
    else:
        X_flat, X_test_flat, y_raw, groups, test_fe = build_v8_features(sanity=args.sanity)
    test_groups = test_fe[cfg.GROUP_COL].values

    # 2. row → sequence 변환
    X_seq, y_seq, train_usc = flat_to_seq(X_flat, y_raw, groups)
    X_test_seq, _, test_usc = flat_to_seq(X_test_flat, None, test_groups)
    v5.log(f"  train seq: {X_seq.shape}  test seq: {X_test_seq.shape}")

    # 3. 하이퍼파라미터
    # defaults (Transformer 계열)
    model_cls, model_kwargs = StrongTransformer, {}

    if args.sanity:
        seeds, n_folds, max_epochs = [42], 2, 10
        model_kwargs = {"d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 128, "dropout": 0.2}
        peak_lr, warmup, patience = 3e-4, 2, 10
        ckpt_prefix = "v8_transformer_sanity"
    elif args.base:
        seeds, n_folds, max_epochs = [42, 123, 2026], 5, 200
        model_cls    = BiLSTMAttention
        model_kwargs = {"hidden": 256, "n_layers": 2, "n_heads": 8, "dropout": 0.2}
        peak_lr, warmup, patience = 3e-4, 10, 30
        ckpt_prefix  = "v8_base_lstm"
    elif args.lstm:
        seeds, n_folds, max_epochs = [42, 123, 2026], 5, 300
        model_cls    = BiLSTMAttention
        model_kwargs = {"hidden": 256, "n_layers": 2, "n_heads": 8, "dropout": 0.2}
        peak_lr, warmup, patience = 1e-4, 20, 50
        ckpt_prefix  = "v8_lstm_attn_v2"
    elif args.cnn:
        seeds, n_folds, max_epochs = [42, 123, 2026], 5, 200
        model_cls    = Conv1DNet
        model_kwargs = {"channels": 512, "dropout": 0.2}
        peak_lr, warmup, patience = 3e-4, 10, 30
        ckpt_prefix  = "v8b_cnn"
    elif args.v5:
        seeds, n_folds, max_epochs = [42, 123, 2026], 5, 200
        model_kwargs = {"d_model": 128, "n_heads": 4, "n_layers": 4, "d_ff": 512, "dropout": 0.2}
        peak_lr, warmup, patience = 3e-4, 10, 30
        ckpt_prefix  = "v8b_transformer_v5"
    elif args.v4:
        seeds, n_folds, max_epochs = [42, 123, 2026], 5, 250
        model_kwargs = {"d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 1024, "dropout": 0.2}
        peak_lr, warmup, patience = 2e-4, 10, 35
        ckpt_prefix = "v8b_transformer_v4"
    elif args.v3:
        seeds, n_folds, max_epochs = [42, 123, 2026], 5, 300
        model_kwargs = {"d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 1024, "dropout": 0.3}
        peak_lr, warmup, patience = 1e-4, 20, 50
        ckpt_prefix = "v8_transformer_v3"
    elif args.v2:
        seeds, n_folds, max_epochs = [42, 123, 2026], 5, 200
        model_kwargs = {"d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 1024, "dropout": 0.2}
        peak_lr, warmup, patience = 3e-4, 10, 30
        ckpt_prefix = "v8_transformer_v2"
    else:
        seeds, n_folds, max_epochs = [42, 123, 2026], 5, 100
        model_kwargs = {"d_model": 256, "n_heads": 8, "n_layers": 4, "d_ff": 1024, "dropout": 0.2}
        peak_lr, warmup, patience = 1e-3, 0, 15
        ckpt_prefix = "v8_transformer"

    # 4. Transformer 학습
    train_loss_fn = nn.L1Loss() if (args.v4 or args.v5 or args.cnn) else None

    all_oof, all_test = [], []
    for seed in seeds:
        cv, oof_seq, test_seq = run_transformer_cv_seq(
            X_seq, y_seq, X_test_seq, train_usc,
            model_cls=model_cls, model_kwargs=model_kwargs,
            n_folds=n_folds, base_seed=seed, max_epochs=max_epochs,
            peak_lr=peak_lr, warmup=warmup, patience=patience,
            ckpt_name=f"{ckpt_prefix}_seed{seed}",
            loss_fn=train_loss_fn,
        )
        oof_flat  = seq_to_flat(oof_seq,  groups,      train_usc)
        test_flat = seq_to_flat(test_seq, test_groups, test_usc)
        all_oof.append(oof_flat); all_test.append(test_flat)
        v5.log(f"  seed={seed} CV(row): {mean_absolute_error(y_raw, oof_flat):.4f}")

    oof_tf  = np.mean(all_oof,  axis=0)
    test_tf = np.mean(all_test, axis=0)
    cv_tf   = mean_absolute_error(y_raw, oof_tf)
    v5.log(f"  {model_cls.__name__} 3-seed avg CV: {cv_tf:.4f}")

    if args.sanity:
        v5.log(f"  총 소요: {(time.time()-t0)/60:.1f}분")
        return

    # 5. N-way blend (GBDT + strict2_tabnet + v8_tabnet + transformer)
    v5.log("\n  === N-way blend ===")
    model_oofs, model_tests, model_names = [], [], []

    v5.CKPT_DIR = V8_CKPT_DIR
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

    # lstm 모드: v8_transformer_v2도 포함 (9-way blend)
    v5.CKPT_DIR = V8_TF_CKPT_DIR
    if args.lstm:
        tf_oofs, tf_tests = [], []
        for seed in [42, 123, 2026]:
            ck = v5.load_ckpt(f"v8_transformer_v2_seed{seed}")
            if ck is not None:
                _, oof_c, test_c = ck
                if oof_c.shape[0] == len(X_seq):
                    tf_oofs.append(seq_to_flat(oof_c, groups, train_usc))
                    tf_tests.append(seq_to_flat(test_c, test_groups, test_usc))
        if tf_oofs:
            model_oofs.append(np.mean(tf_oofs, axis=0))
            model_tests.append(np.mean(tf_tests, axis=0))
            model_names.append("v8_transformer_v2")

    model_oofs.append(oof_tf);  model_tests.append(test_tf)
    model_names.append(ckpt_prefix)
    weights, blend_cv = v5.optimize_blend_multi(model_oofs, y_raw, model_names)

    # 6. 제출 파일
    oof_blend  = sum(w * o for w, o in zip(weights, model_oofs))
    test_blend = sum(w * t for w, t in zip(weights, model_tests))

    sub = pd.DataFrame({cfg.ID_COL: test_fe[cfg.ID_COL].astype(str),
                        cfg.TARGET: np.clip(test_blend, 0, None)})
    p_nc = os.path.join(cfg.SUBMISSION_DIR, f"{ckpt_prefix}_blend_noclip_submission.csv")
    sub.to_csv(p_nc, index=False, float_format="%.10f", lineterminator="\n")
    v5.log(f"  저장: {p_nc}  mean={sub[cfg.TARGET].mean():.3f} std={sub[cfg.TARGET].std():.3f}")

    clip_th = np.percentile(oof_blend, 99) * 1.30
    sub_c = sub.copy(); sub_c[cfg.TARGET] = np.clip(test_blend, 0, clip_th)
    p_c  = os.path.join(cfg.SUBMISSION_DIR, f"{ckpt_prefix}_blend_clip130_submission.csv")
    sub_c.to_csv(p_c, index=False, float_format="%.10f", lineterminator="\n")
    cv_clip = mean_absolute_error(y_raw, np.clip(oof_blend, 0, clip_th))
    v5.log(f"  저장: {p_c}  clip_th={clip_th:.2f} CV_clip={cv_clip:.4f}")

    v5.log("\n" + "=" * 60)
    v5.log(f"  {model_cls.__name__} CV: {cv_tf:.4f}")
    v5.log(f"  blend noclip CV:  {blend_cv:.4f}")
    v5.log(f"  blend clip130 CV: {cv_clip:.4f}")
    v5.log(f"  총 소요: {(time.time()-t0)/60:.1f}분")
    v5.log("=" * 60)


if __name__ == "__main__":
    main()
