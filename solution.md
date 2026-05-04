# 스마트 창고 출고 지연 예측 AI 경진대회 - 솔루션

## 1. 문제 정의 및 접근 전략

### 1.1 문제 요약
- **과제**: 스마트 물류창고 운영 스냅샷(15분 단위, 시나리오당 25 타임스텝) → 향후 30분 평균 출고 지연 시간(분) 예측
- **데이터**: 정형 (train 250,000행 × 94컬럼, test 50,000행, 보조 layout_info 300행)
- **타깃**: `avg_delay_minutes_next_30m` (연속값 회귀)
- **평가**: MAE (낮을수록 우수) / Public 30% · Private 70%

### 1.2 접근 전략
1. **PB-style 피처 엔지니어링**: lag/rolling/비율/충전 압력 등 도메인 특화 피처 714개 구성
2. **시나리오 집계 피처**: 25 타임스텝 전체에 걸친 mean/std/max/min 통계 84개 추가 (→ 798개)
3. **이종 모델 다양성**: GBDT 3종 + TabNet + Transformer + MLP의 9개 모델군, 3-seed × 5-fold
4. **LGB 메타 스태킹**: OOF 예측 + 예측 불확실성(std/range)을 메타 피처로 활용
5. **GroupKFold**: scenario_id 기준으로 시나리오 단위 정보 누수 방지

---

## 2. 피처 엔지니어링 (798개)

### 2.1 피처 구성 요약

| 그룹 | 수 | 설명 |
|------|-----|------|
| strict base | 198개 | lag(1,2), rolling(3,5), expanding 통계, timestep, layout |
| attack lead/forward | 107개 | 미래 타임스텝 예측을 위한 ops lead/forward 피처 |
| PB-style lag/rolling/ratio | 409개 | 배터리 압력, 충전 대기 압력, 수요-로봇 비율, onset 피처 등 |
| **시나리오 집계** | **84개** | 25 타임스텝 전체 mean/std/max/min (21개 컬럼 × 4 통계) |

### 2.2 핵심 피처 아이디어

**PB-style 피처 (sections 1~12)**
- `charge_pressure_pb`: 충전 중인 로봇 비율 × 충전 큐 × 배터리 긴급도
- `battery_pressure_pb`: 저배터리 비율의 지수 함수 가중치
- `demand_mass_per_robot`: 주문 유입량 / 활성 로봇 수 (로봇당 부하)
- `congestion_x_lowbat`: 혼잡도 × 저배터리 비율 (이중 압박 지표)
- Onset 피처: 충전/큐가 처음 발생하는 타임스텝 위치

**시나리오 집계 피처 (section 13)** — 가장 큰 단독 개선 효과
```
시나리오 내 25 타임스텝의 mean/std/max/min → 
현재 타임스텝 값과 전체 시나리오 패턴을 동시에 학습 가능
```

---

## 3. 모델 구성

### 3.1 GBDT 모델 (checkpoints_v8/)

| 모델명 | objective | target 변환 | seeds |
|--------|-----------|-------------|-------|
| `lgb_mae_log` | LightGBM MAE | log1p | 42, 123, 2026 |
| `lgb_huber_log` | LightGBM Huber | log1p | 42, 123, 2026 |
| `cat_mae_log` | CatBoost MAE | log1p | 42, 123, 2026 |
| `lgb_mae_raw` | LightGBM MAE | raw | 42, 123, 2026 |
| `xgb_mae_raw` | XGBoost MAE | raw | 42, 123, 2026 |

- 피처: 798개 (v8b prefix)
- 검증: GroupKFold(5, groups=scenario_id)
- 체크포인트: `checkpoints_v8/v8b_{name}_seed{seed}.pkl`

### 3.2 TabNet 모델 (checkpoints_v8_tabnet/, checkpoints_strict/)

| 모델명 | 피처 | 비고 |
|--------|------|------|
| `v8_tabnet` | 714개 (attack 기반) | pytorch-tabnet |
| `strict2_tabnet` | 198개 (strict base) | 구 피처셋 |

### 3.3 Transformer v4 (checkpoints_v8_transformer/)

```
Input: (B, 25, 798)  ← 25 timestep sequence
        |
  Linear projection → d=256
        |
  Pre-Norm Transformer (L=6, heads=8, d_ff=1024)
        |
  Sequence output → (B, 25, 1)
        |
    Target per timestep
```

- **핵심 설정**: L1Loss (MAE loss), lr=2e-4, warmup=10 epoch, cosine LR, patience=35, epoch=250
- bfloat16 AMP, GroupKFold(5)
- 체크포인트: `checkpoints_v8_transformer/v8b_transformer_v4_seed{seed}.pkl`

### 3.4 MLP (checkpoints_v8_transformer/)

```
Input: (B, 798)  ← flat feature vector (row-level)
        |
  512 → BN+GELU+Drop → 512 → BN+GELU+Drop → 256 → BN+GELU+Drop → 1
```

- L1Loss, lr=3e-4, cosine warmup, patience=30, epoch=300
- 체크포인트: `checkpoints_v8_transformer/v8b_mlp_seed{seed}.pkl`

---

## 4. 스태킹 메타러너

### 4.1 메타 피처 구성 (11개)

| 피처 | 수 |
|------|-----|
| 9개 모델의 OOF 예측값 | 9개 |
| **예측 std** (모델 불확실성) | 1개 |
| **예측 range** (max-min) | 1개 |

불확실성 피처가 핵심 개선 기여:
- CV: 8.3961 → 8.3942
- Public LB: 9.821 → **9.802**

### 4.2 메타러너: LightGBM

```python
params = {
    "objective": "mae", "num_leaves": 15,
    "learning_rate": 0.05, "min_child_samples": 100,
    "subsample": 0.8, "colsample_bytree": 1.0, "seed": 42,
}
```

GroupKFold(5), early_stopping=50, num_boost_round=3000

---

## 5. 핵심 발견 및 실험 이력

### 5.1 핵심 발견

1. **시나리오 집계 피처** (section 13): GBDT blend CV 8.5426 → 8.4806 (+0.0620). 가장 큰 단독 개선
2. **예측 불확실성 메타 피처**: 메타러너가 모델 불일치 상황을 더 잘 처리 → LB 9.821 → 9.802
3. **LGB 스태킹 > blend > ridge**: 메타러너 복잡도와 성능이 비례하지 않음 (num_leaves=15 최적)
4. **1D CNN 부적합**: 25 timestep이 너무 짧아 로컬 컨볼루션 패턴 추출 불가 (CV ≥ 10.21)
5. **CV-LB 일관성**: GroupKFold CV가 LB 방향과 잘 일치 → CV 기반 의사결정 신뢰 가능
6. **메타 모델 수 주의**: 9→11개 증가 시 오히려 CV 악화 (과적합)

### 5.2 제출 이력

| # | 파일 | Public LB | CV | 핵심 |
|---|------|----------|----|------|
| 1 | v8_transformer_v4_blend_noclip | 9.9850 | 8.5372 | TF v4 블렌드 기준선 |
| 2 | v8_stacking_v1 | 9.8347 | 8.4230 | LGB 스태킹 8모델 도입 |
| 3 | v8_stacking_v2 | 9.8215 | 8.4050 | v8b transformer(798피처) 반영 |
| 4 | v8_stacking_v3 | 9.8254 | 8.3961 | MLP 추가 (9모델) |
| **5** | **v8_stacking_v9_unc** | **9.8020** | **8.3942** | **예측 std/range 메타 피처** |

---

## 6. 최종 결과

| 항목 | 값 |
|------|-----|
| **최고 Public LB** | **9.802014278** |
| **Private Score** | **10.01488** |
| **최종 순위** | **18등** |
| **최고 CV** | **8.3942** |
| **제출 파일** | `v8_stacking_v9_unc_submission.csv` |
| **시작 대비 개선** | 10.2990 → 9.8020 (Public) / 10.01488 (Private) |

---

## 7. 개발 환경 및 재현 방법

### 7.1 개발 환경

| 항목 | 값 |
|------|-----|
| OS | Ubuntu (WSL2) / Linux 6.6.87.2-microsoft-standard-WSL2 |
| GPU | NVIDIA RTX 5060 Ti 16GB |
| CUDA | 12.8 |
| Python | 3.13.12 |
| PyTorch | 2.10.0+cu128 (bfloat16 AMP) |
| LightGBM | 4.6.0 |
| XGBoost | 3.2.0 |
| CatBoost | 1.2.10 |
| pytorch-tabnet | 4.1.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.1 |
| numpy | 2.4.2 |

### 7.2 재현 순서

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 배치
# data/train.csv, data/test.csv, data/layout_info.csv, data/sample_submission.csv

# 3. GBDT 학습 (798 피처, v8b prefix)
python run_experiments_v8.py --ckpt_prefix v8b

# 4. Transformer v4 학습 (798 피처, L1Loss)
python run_v8_transformer.py --v4

# 5. MLP 학습 (798 피처, L1Loss)
python run_v8_mlp.py

# 6. TabNet 학습 (선택 — 체크포인트 포함)
# python run_v8_tabnet.py

# 7. Stacking (불확실성 메타 피처 포함)
python run_stacking.py --tag v9
# → submissions/v8_stacking_v9_submission.csv
```

### 7.3 파일 구조

```
smart_warehouse/
├── config.py                   # 하이퍼파라미터 중앙 관리 (경로, 컬럼, 모델 파라미터)
├── dataset.py                  # 데이터 로드 (사용하지 않는 경우 run_experiments_v8.py가 직접 로드)
├── run_experiments_v8.py       # GBDT 학습 (--ckpt_prefix v8b, 798 피처)
├── run_v8_transformer.py       # Transformer v4/MLP 학습 (--v4, --mlp)
├── run_v8_mlp.py               # MLP 전용 학습 스크립트
├── run_v8_tabnet.py            # TabNet 학습
├── run_stacking.py             # 스태킹 메타러너 (--tag v9)
├── run_experiments_v5.py       # 공용 유틸리티 (load_ckpt, optimize_blend 등)
├── run_experiments_strict.py   # strict2_tabnet 체크포인트 경로 참조용
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── layout_info.csv
│   └── sample_submission.csv
├── checkpoints_v8/             # GBDT v8b 체크포인트 (90개 파일)
├── checkpoints_v8_transformer/ # TF v4 + MLP 체크포인트 (21개 파일)
├── checkpoints_v8_tabnet/      # TabNet 체크포인트 (18개 파일)
├── checkpoints_strict/         # strict2_tabnet 체크포인트 (18개 파일)
└── submissions/
    └── v8_stacking_v9_unc_submission.csv  ← 최고 LB (9.8020)
```

---

## 8. 코드 설명 (1000자 제한용)

## PB-style 피처 엔지니어링
로봇 운영 지표(배터리·혼잡도·충전 큐 등)의 lag/rolling/비율 피처 714개에 시나리오 내 25개 타임스텝 집계(mean/std/max/min) 84개를 추가, 총 798개 피처를 구성했습니다. 시나리오 집계 피처가 GBDT blend CV 8.54 → 8.48 개선의 핵심이었습니다.

## 이종 모델 9종
LightGBM(MAE·Huber), XGBoost(MAE), CatBoost(MAE), TabNet, Transformer(Pre-Norm d=256 L=6 L1Loss), MLP(TabMLP L1Loss)를 3-seed × 5-fold GroupKFold로 학습하여 다양성을 확보했습니다.

## 불확실성 메타 스태킹
9개 OOF 예측값 외에 예측 std(모델 불확실성)와 range(max-min)를 메타 피처로 추가한 LGB 스태킹(num_leaves=15) 적용. 불확실성 피처 추가가 Public LB 9.821 → 9.802 개선의 핵심 기여입니다.

## GroupKFold
scenario_id 기준 GroupKFold(5)로 시나리오 단위 정보 누수를 방지. GBDT는 log1p 타깃 변환, Transformer/MLP는 L1Loss 직접 학습.

## 재현
Score 복원: run_experiments_v8.py → run_v8_transformer.py --v4 → run_v8_mlp.py → run_stacking.py --tag v9
폴더: ./data/(데이터), ./checkpoints_v8/(GBDT), ./checkpoints_v8_transformer/(DL), ./checkpoints_v8_tabnet/(TabNet), ./checkpoints_strict/(strict2_tabnet)
최종 성적: Public LB 9.802014278 / Private 10.01488 / 18등
