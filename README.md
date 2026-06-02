# Gridlock Hackathon 2.0 — Traffic Demand Prediction

End-to-end pipeline for the [HackerEarth Gridlock Hackathon 2.0](https://www.hackerearth.com/community/challenges/competitive/gridlock-hackathon-20/) traffic-demand-prediction problem.

The task: given 1.4 days of 15-minute traffic observations across ~1,250 Bengaluru geohashes, predict the `demand` for the rest of day 49 (qhour 9–55, i.e. 02:15–13:45). Metric: `max(0, 100 × R²)`.

---

## Final result

The pipeline ends with `submission.csv` (41,778 rows, columns `Index, demand`). The estimated leaderboard score is in `blend_weights.json` (`val_fold_r2.blend_with_sat` × 100), computed on a daytime-matched holdout that mirrors the test distribution.

---

## Setup

```bash
python -m pip install -r requirements.txt
```

Place the dataset zip somewhere on disk and extract it. Either:
- keep the folder named `e88186124ec611f1 (1)/dataset/` next to the notebooks (the default the notebooks assume), or
- update the `DATA_DIR` constant at the top of each notebook.

The dataset should contain `train.csv`, `test.csv`, `sample_submission.csv`.

---

## Execution order

Run the notebooks **strictly in this order**. Each consumes the outputs of the previous ones from `processed/`, `oof/`, or `predictions/`.

| Order | Notebook | What it does | Approx. runtime (CPU) |
|---|---|---|---|
| 1 | `eda.ipynb` | Exploratory data analysis; documents all anomalies and design decisions | ~1 min |
| 2 | `cleaning_v2.ipynb` | Leakage-free cleaning + train-only imputation | ~30 s |
| 3 | `val_split_v2.ipynb` | Daytime-matched validation holdout | ~5 s |
| 4 | `feature_engineering_v2.ipynb` | Reindex grid, lags, rolling, OOF target encodings | ~1 min |
| 5 | `model_train_lgbm.ipynb` | 3 LightGBM base models (Huber / MAE / Quantile-0.9) with Optuna search | ~20 min |
| 6 | `model_train_moe.ipynb` | RoadType Mixture-of-Experts (Highway / Street / Residential / Global) | ~25 min |
| 7 | `model_train_saturation.ipynb` | Binary classifier for `demand ≥ 0.99` on saturating geohashes | ~2 min |
| 8 | `model_blend.ipynb` | NNLS blend of base learners + saturation residual correction | ~5 s |
| 9 | `model_inference.ipynb` | Writes `submission.csv` | ~5 s |
| 10 | `final_verification.ipynb` | Independent audit of submission + leakage spot-checks | ~10 s |

Total CPU runtime: **~50–60 minutes** on a modern laptop.

You can execute each notebook either via Jupyter Lab/Notebook UI, or non-interactively:

```bash
python -m jupyter nbconvert --to notebook --execute eda.ipynb --output eda.ipynb --ExecutePreprocessor.timeout=3600
```

For the heavy training notebooks (5, 6) you may want to bump the timeout to ~3600s.

---

## Theory & design rationale

The competition has three structural features that drove every modelling decision:

### 1. It's a *forecasting* problem, not a regression problem

Train covers **day 48 fully (96 quarter-hour slots) + day 49 first 9 slots (00:00–02:00)**. Test is **day 49 slots 9–55 (02:15–13:45)**. There is zero timestamp overlap between train and test on day 49. So we're predicting future quarter-hours from past ones.

**Consequences:**
- A random K-fold validation would leak future-into-past. We use a **time-block CV** (`feature_engineering_v2`) where each fold is a contiguous range of `t_abs`.
- Our **validation holdout** (`val_split_v2`) is a 47-slot daytime window from day 48 chosen to mirror the test horizon's `qhour` distribution.
- Lag features must respect causality. We reindex each geohash onto the full 15-minute grid, compute lags on the dense grid, then drop the synthetic rows.

### 2. The features that look static aren't

`RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks` *seem* like properties of the road — they should be constant per geohash. But in the data, they **vary row-by-row** within the same geohash (in >96% of geohashes for some of these columns). Likely explanation: a 6-char geohash tile (~0.6 km × 1.2 km) contains multiple roads and observations sample from different ones.

**Consequences:**
- We keep them as **per-row features** (they ARE informative per row).
- We *also* compute the **per-geohash modal value** as a separate feature (`*_modal`) — the location's dominant character is still useful.
- Critically, modal values are computed **train-only** (`cleaning_v2.ipynb`), with a hierarchical fallback (prefix5 → prefix4 → global) for the 10 cold-start test geohashes that have no train rows. The initial `cleaning.ipynb` had a leakage bug here — it used train+test combined.

### 3. The target is heavy-tailed and right-censored

Target distribution: median 0.048, p95 ≈ 0.34, max = 1.0 with **585 records exactly at 1.0**, almost all on Highway and 4–5-lane geohashes. R² penalises errors on these high-value rows quadratically, so a single mis-prediction here hurts a lot.

**Consequences:**
- We train **three diverse LightGBM bases** (Huber on log1p, MAE on raw, Quantile-0.9 on log1p). Each captures a different part of the distribution; a non-negative blend exceeds any individual.
- We add a **saturation residual head**: a binary classifier P(demand ≥ 0.99) trained on the ~23 known-saturating geohashes. Applied at inference as `pred + P_sat × (1 - pred)` only on Highway-saturating cells. The correction can only pull predictions up (and never lowers them), so it's safe.

---

## Architecture summary

```
                           ┌──────────────────────────┐
                           │  raw train.csv + test.csv │
                           └────────────┬─────────────┘
                                        ▼
                           ┌──────────────────────────┐
                           │  cleaning_v2.ipynb       │
                           │  (train-only imputation) │
                           └────────────┬─────────────┘
                                        ▼
                           ┌──────────────────────────┐
                           │  val_split_v2.ipynb      │
                           │  (daytime-matched fold)  │
                           └────────────┬─────────────┘
                                        ▼
                           ┌──────────────────────────┐
                           │  feature_engineering_v2  │
                           │  lags, rolling, OOF enc  │
                           └────────────┬─────────────┘
                                        ▼
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
   ┌──────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
   │ model_train_lgbm │    │ model_train_moe      │    │ model_train_satur..│
   │  Huber, MAE, Q90 │    │  Hwy/Str/Res/Global  │    │  P(demand ≥ 0.99)  │
   └────────┬─────────┘    └──────────┬───────────┘    └──────────┬──────────┘
            └────────────────┬────────┴───────────────────────────┘
                             ▼
                     ┌─────────────────┐
                     │  model_blend    │  NNLS + saturation residual
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │ model_inference │  writes submission.csv
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │ final_verify    │  independent audit
                     └─────────────────┘
```

### Feature engineering (the heart of the pipeline)

70 model features split into seven groups (see `processed/feature_meta_v2.json`):

| Group | Examples | Notes |
|---|---|---|
| Time | `qhour_of_day`, `qhour_sin/cos`, `tod_bucket_code` | `hour` is dropped because it's redundant with `qhour_of_day` |
| Geo | `geohash_code`, `gh_prefix5_code`, `lat`, `lon`, `gh_train_count` | Lat/lon decode to anonymised coords (NOT real Bengaluru) — use only relative geometry |
| Road (per-row) | `RoadType_code`, `NumberofLanes`, `LargeVehicles_code` | Vary row-by-row even within the same geohash |
| Road (modal) | `RoadType_modal_code`, `NumberofLanes_modal` | The dominant value per geohash (train-only) |
| Weather | `Weather_code`, `Temperature`, `*_was_missing` | Imputation is train-only |
| Lags & rolling | `lag_1/2/4/8/16/96`, `rolling_mean_4/8/16/32`, `ewm_demand`, `boundary_demand`, `time_since_boundary`, `last_seen_gap_to_boundary` | All causal (no future leakage) |
| OOF target enc | `gh_mean_demand_oof`, `gh_p90_demand_oof`, `gh_qhour_mean_oof`, `qhour_roadtype_mean_oof`, etc. | Empirical-Bayes shrunk; computed via 5-fold CV |

### Validation

Three validation signals are reported by each model:

1. **OOF R² (all train)** — every row's prediction comes from a fold that didn't include it. The most pessimistic estimate (uses all 5 folds, including the morning fold most similar to test).
2. **OOF R² (val_fold subset)** — restricted to the daytime-matched holdout. Most likely to track leaderboard.
3. **NNLS-blend val_fold R²** — the final number to compare across model variants. Saved in `blend_weights.json`.

### Why an NNLS blend (and not Ridge or LinearRegression)

Non-negative weights:
- Are interpretable as a mixture (which model is contributing what fraction).
- Naturally zero-out a redundant base learner without manual feature selection.
- Are stable under small data perturbations.

---

## File / folder layout after a full run

```
.
├── README.md                              # this file
├── requirements.txt
├── e88186124ec611f1 (1)/dataset/          # raw data
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── eda.ipynb                              # the EDA / dataset audit
├── cleaning_v2.ipynb
├── val_split_v2.ipynb
├── feature_engineering_v2.ipynb
├── model_train_lgbm.ipynb
├── model_train_moe.ipynb
├── model_train_saturation.ipynb
├── model_blend.ipynb
├── model_inference.ipynb
├── final_verification.ipynb
│
├── processed/                             # outputs of stages 2–4
│   ├── train_clean_v2.csv
│   ├── test_clean_v2.csv
│   ├── imputation_lookups.json
│   ├── feature_meta_v2.json
│   ├── train_features.parquet
│   ├── test_features.parquet
│   ├── cv_folds.parquet
│   ├── saturating_geohashes.json
│   ├── val_split_v2.json
│   └── val_split_v2_indices.parquet
│
├── oof/                                   # OOF predictions from training (stage 5–7)
│   ├── lgbm_huber.parquet
│   ├── lgbm_mae.parquet
│   ├── lgbm_quantile.parquet
│   ├── lgbm_summary.json
│   ├── moe_blended.parquet
│   ├── moe_expert.parquet
│   ├── moe_global.parquet
│   ├── moe_summary.json
│   ├── saturation_prob.parquet
│   └── sat_fire_mask_train.npy
│
├── predictions/                           # test predictions from each model + final
│   ├── lgbm_*_test.parquet
│   ├── moe_*_test.parquet
│   ├── saturation_prob_test.parquet
│   ├── sat_fire_mask_test.npy
│   └── final_test_predictions.parquet
│
├── models/                                # serialized LightGBM boosters
│   └── *.txt
│
├── blend_weights.json                     # NNLS weights + val R² breakdown
└── submission.csv                         # ← UPLOAD THIS to HackerEarth
```

---

## Reproducibility notes

- All randomness seeded with `SEED = 42` (LightGBM `random_state`, Optuna `TPESampler(seed=...)`).
- Optuna uses 20 trials per LGBM model and 15 trials per MoE expert. Bump if you have more time; gains will be small.
- The whole pipeline is deterministic except for tiny numerical differences from multi-threaded LightGBM on different hardware.

---

## Credits

Built with Claude (Anthropic) as a pair-programmer over multiple iteration cycles. Architecture refined via adversarial-audit workflows that found and fixed three critical leakage bugs in the initial cleaning pipeline.
