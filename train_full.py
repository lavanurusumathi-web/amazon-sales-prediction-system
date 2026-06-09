#!/usr/bin/env python3
"""Full pipeline: generate large dataset, train models, predict, launch server."""
import sys, os, time, json, logging
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("full-pipeline")

N_PRODUCTS = 3000
N_DAYS = 365
TUNE = False
TRAIN_SAMPLE = 150000

# ── 1. Generate ──────────────────────────────────────────────────
logger.info("=" * 70)
logger.info(f"STEP 1: Generating dataset ({N_PRODUCTS} products x {N_DAYS} days)")
t0 = time.time()

from data.dataset import AmazonSalesDataset
ds = AmazonSalesDataset(data_dir="data")
if ds._cache_path and os.path.exists(ds._cache_path):
    os.remove(ds._cache_path)

df = ds.generate(n_products=N_PRODUCTS, n_days=N_DAYS, force=True)
elapsed = time.time() - t0
size_mb = os.path.getsize(ds._cache_path) / 1024 / 1024
logger.info(f"  Done in {elapsed:.0f}s: {len(df):,} rows, {df['product_id'].nunique():,} products, {size_mb:.1f} MB")
logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
logger.info(f"  Total sales: {df['sales_units'].sum():,} units, Revenue: ${df['revenue'].sum():,.0f}")

# ── 2. Feature Engineering ──────────────────────────────────────
logger.info("=" * 70)
logger.info("STEP 2: Feature engineering")
from features.engineering import FeatureEngineer
fe = FeatureEngineer()

df_sample = df.sort_values("date").tail(TRAIN_SAMPLE).copy()
feat_df = fe.create_features(df_sample)
logger.info(f"  Features: {feat_df.shape}")

# ── 3. Preprocessing ────────────────────────────────────────────
logger.info("=" * 70)
logger.info("STEP 3: Preprocessing")
from data.preprocessor import DataPreprocessor
pp = DataPreprocessor()
processed = pp.fit_transform(feat_df)
X, y = pp.prepare_for_training(processed, target_col="sales_units")

split_idx = int(len(X) * 0.8)
val_idx = int(split_idx * 0.875)
X_tr, y_tr = X.iloc[:val_idx], y.iloc[:val_idx]
X_val, y_val = X.iloc[val_idx:split_idx], y.iloc[val_idx:split_idx]
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
logger.info(f"  Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ── 4. Train ────────────────────────────────────────────────────
logger.info("=" * 70)
logger.info("STEP 4: Training ensemble")
from models.train import ModelTrainer
trainer = ModelTrainer(model_dir="models")
t0 = time.time()
trainer.train_ensemble(X_tr, y_tr, X_val, y_val, tune=TUNE)
elapsed = time.time() - t0
logger.info(f"  Training done in {elapsed:.0f}s")
logger.info(f"  Models: {list(trainer.models.keys())}")

# ── 5. Evaluate ─────────────────────────────────────────────────
logger.info("=" * 70)
logger.info("STEP 5: Evaluation")
from models.evaluate import ModelEvaluator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

evaluator = ModelEvaluator()
for name, model in trainer.models.items():
    if model is None or name == "ensemble_weights":
        continue
    try:
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"  {name:20s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    except Exception as e:
        logger.warning(f"  {name}: eval failed - {e}")

# Ensemble
weights = trainer.models.get("ensemble_weights", {})
logger.info(f"  Ensemble weights: {weights}")

# ── 6. Save ─────────────────────────────────────────────────────
save_path = trainer.save_models(tag="pretrained")
logger.info(f"  Models saved to: {save_path}")

# ── 7. Predict ──────────────────────────────────────────────────
logger.info("=" * 70)
logger.info("STEP 6: Future sales predictions")
from models.predict import SalesPredictor

predictor = SalesPredictor(
    models={k: v for k, v in trainer.models.items() if v is not None and k != "ensemble_weights"},
    weights=weights,
    feature_engineer=fe,
    preprocessor=pp,
)

# Top products by recent sales
from datetime import timedelta
recent = df[df["date"] >= df["date"].max() - timedelta(days=60)]
top_pids = recent.groupby("product_id")["sales_units"].sum().sort_values(ascending=False).head(15).index.tolist()

DAYS = 30
forecasts = []
for pid in top_pids:
    fc = predictor.predict_future(df, days_ahead=DAYS, product_id=pid)
    last = df[df["product_id"] == pid].sort_values("date").iloc[-1]
    fc["product_id"] = pid
    fc["title"] = last["product_title"]
    fc["cat"] = last["category"]
    fc["price"] = last["price"]
    forecasts.append(fc)

fc_df = pd.concat(forecasts, ignore_index=True)
summary = fc_df.groupby(["product_id", "title", "cat", "price"]).agg(
    total_pred=("predicted_sales", "sum"),
    avg_daily=("predicted_sales", "mean"),
    avg_lo=("lower_bound", "mean"),
    avg_hi=("upper_bound", "mean"),
).reset_index().sort_values("total_pred", ascending=False)
summary["revenue"] = summary["total_pred"] * summary["price"]

import pandas as pd
print()
print(f"{'#':<4} {'Product':<50} {'Category':<20} {'Price':>8} {'30-Day':>9} {'Revenue':>12} {'±CI':>8}")
print("-" * 125)
for i, (_, r) in enumerate(summary.iterrows()):
    print(f"{i+1:<4} {r['title'][:48]:<50} {r['cat'][:18]:<20} ${r['price']:>7.2f} {int(r['total_pred']):>9,} ${r['revenue']:>11,.0f} ±{int(r['avg_hi']-r['avg_lo']):>6}")

total_u = summary["total_pred"].sum()
total_r = summary["revenue"].sum()
print("-" * 125)
print(f"\n  TOP 15 PRODUCTS → {total_u:,.0f} units | ${total_r:,.0f} revenue | {summary['avg_daily'].mean():.1f} avg daily/prod")
print(f"  Dataset: {N_PRODUCTS:,} products | {len(df):,} rows | {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Model: {weights}")
print("=" * 70)

# Save summary CSV
summary.to_csv("data/forecast_results.csv", index=False)
logger.info("Forecast saved to data/forecast_results.csv")
logger.info("DONE!")
