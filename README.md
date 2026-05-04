# 스마트 창고 출고 지연 예측 AI 경진대회

**DACON** | 최종 순위: **18등** | Public LB: **9.8020** | Private: **10.0149**

---

## 문제 정의

스마트 물류창고 운영 스냅샷(15분 단위, 시나리오당 25 타임스텝)을 입력으로, 향후 30분 평균 출고 지연 시간(분)을 예측하는 회귀 문제.

- **데이터**: train 250,000행 × 94컬럼 / test 50,000행 / 보조 layout_info 300행
- **타깃**: `avg_delay_minutes_next_30m`
- **평가**: MAE (낮을수록 우수) / Public 30% · Private 70%

---

## 솔루션 요약

### 1. 피처 엔지니어링 (798개)

| 그룹 | 수 | 설명 |
|------|----|------|
| strict base | 198개 | lag(1,2), rolling(3,5), expanding 통계, timestep, layout |
| attack lead/forward | 107개 | 미래 타임스텝 예측을 위한 ops lead/forward 피처 |
| PB-style lag/rolling/ratio | 409개 | 배터리 압력, 충전 대기 압력, 수요-로봇 비율, onset 피처 |
| **시나리오 집계** | **84개** | 25 타임스텝 전체 mean/std/max/min (21컬럼 × 4통계) |

핵심 피처:
- `charge_pressure_pb`: 충전 중 로봇 비율 × 충전 큐 × 배터리 긴급도
- `battery_pressure_pb`: 저배터리 비율의 지수 함수 가중치
- `demand_mass_per_robot`: 주문 유입량 / 활성 로봇 수
- `congestion_x_lowbat`: 혼잡도 × 저배터리 비율
- **시나리오 집계 피처**: 단독으로 GBDT blend CV 8.5426 → 8.4806 개선

### 2. 모델 구성 (9종)

| 모델 | objective | target 변환 |
|------|-----------|-------------|
| LightGBM MAE | MAE | log1p |
| LightGBM Huber | Huber | log1p |
| CatBoost MAE | MAE | log1p |
| LightGBM MAE raw | MAE | raw |
| XGBoost MAE raw | MAE | raw |
| TabNet (v8) | — | raw |
| TabNet (strict2) | — | raw |
| Transformer v4 | L1Loss | raw |
| MLP | L1Loss | raw |

- 검증: **GroupKFold(5, groups=scenario_id)** — 시나리오 단위 정보 누수 방지
- 3 seed (42, 123, 2026) × 5 fold = 모델당 15개 체크포인트

**Transformer v4 구조**
```
Input: (B, 25, 798)
  → Linear projection (d=256)
  → Pre-Norm Transformer (L=6, heads=8, d_ff=1024)
  → Sequence output: (B, 25, 1)
```

### 3. 스태킹 메타러너

메타 피처 11개: 9개 모델 OOF 예측 + **예측 std** + **예측 range**

불확실성 메타 피처 추가가 Public LB 9.821 → **9.802** 개선의 핵심.

```python
meta_params = {
    "objective": "mae", "num_leaves": 15,
    "learning_rate": 0.05, "min_child_samples": 100,
    "subsample": 0.8, "colsample_bytree": 1.0,
}
```

---

## 제출 이력

| # | Public LB | CV | 핵심 |
|---|----------|----|------|
| 1 | 9.9850 | 8.5372 | Transformer v4 블렌드 기준선 |
| 2 | 9.8347 | 8.4230 | LGB 스태킹 8모델 도입 |
| 3 | 9.8215 | 8.4050 | v8b Transformer(798피처) 반영 |
| 4 | 9.8254 | 8.3961 | MLP 추가 (9모델) |
| **5** | **9.8020** | **8.3942** | **예측 std/range 메타 피처** |

---

## 재현 방법

```bash
pip install -r requirements.txt

# data/ 폴더에 배치: train.csv, test.csv, layout_info.csv, sample_submission.csv

python run_experiments_v8.py --ckpt_prefix v8b   # GBDT 학습
python run_v8_transformer.py --v4                 # Transformer v4
python run_v8_mlp.py                              # MLP
python run_stacking.py --tag v9                   # 스태킹 → submissions/
```

---

## 파일 구조

```
.
├── config.py                  # 하이퍼파라미터 중앙 관리
├── dataset.py                 # 데이터 로드 유틸
├── run_experiments_v8.py      # GBDT 학습 (798 피처)
├── run_v8_transformer.py      # Transformer v4 학습
├── run_v8_mlp.py              # MLP 학습
├── run_v8_tabnet.py           # TabNet 학습
├── run_stacking.py            # 스태킹 메타러너
├── run_experiments_v5.py      # 공용 유틸 (load_ckpt, optimize_blend 등)
├── run_experiments_strict.py  # strict2_tabnet 참조용
├── solution.md                # 상세 솔루션 문서
├── submission_package.ipynb   # 제출용 노트북
└── v8_stacking_v9_unc_submission.csv  # 최고 LB 제출 파일
```

> `data/`, `checkpoints_*/` 폴더는 .gitignore 처리

---

## 개발 환경

| 항목 | 값 |
|------|-----|
| OS | Ubuntu (WSL2) |
| GPU | NVIDIA RTX 5060 Ti 16GB |
| CUDA | 12.8 |
| Python | 3.13.12 |
| PyTorch | 2.10.0+cu128 |
| LightGBM | 4.6.0 |
| XGBoost | 3.2.0 |
| CatBoost | 1.2.10 |
