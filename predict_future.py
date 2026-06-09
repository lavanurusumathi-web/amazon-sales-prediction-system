#!/usr/bin/env python3
"""Load trained models and predict future sales for top products."""
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from data.dataset import AmazonSalesDataset
from features.engineering import FeatureEngineer
from data.preprocessor import DataPreprocessor
from models.predict import SalesPredictor
from models.train import ModelTrainer

MODEL_DIR = "models/models_pretrained"
DAYS_AHEAD = 30
TOP_N = 15

print("=" * 70)
print("   AMAZON SALES PREDICTOR - Future Sales Forecast")
print("=" * 70)

# Load dataset
print("\n[1/4] Loading dataset...")
ds = AmazonSalesDataset(data_dir="data")
df = ds.load_from_parquet("data/amazon_sales_data.parquet")
print(f"  Loaded {len(df):,} rows, {df['product_id'].nunique()} products")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Load models
print("\n[2/4] Loading trained models...")
trainer = ModelTrainer(model_dir="models")
trainer.load_models(MODEL_DIR)

# Load weights
weights_path = os.path.join(MODEL_DIR, "ensemble_weights.json")
weights = {}
if os.path.exists(weights_path):
    with open(weights_path) as f:
        weights = json.load(f)
    print(f"  Ensemble weights: {weights}")

print(f"  Models loaded: {list(trainer.models.keys())}")

# Setup predictor
print("\n[3/4] Running predictions...")
fe = FeatureEngineer()
pp = DataPreprocessor()

# Fit preprocessor on sample
sample = df.sample(min(10000, len(df)), random_state=42).copy()
sample_feat = fe.create_features(sample)
pp.fit_transform(sample_feat)

predictor = SalesPredictor(
    models=trainer.models,
    weights=weights,
    feature_engineer=fe,
    preprocessor=pp,
    confidence_level=0.80
)

# Get top products by recent sales
recent = df[df["date"] >= df["date"].max() - timedelta(days=60)]
top_products = (
    recent.groupby("product_id")["sales_units"].sum()
    .sort_values(ascending=False)
    .head(TOP_N)
    .index.tolist()
)

print(f"  Forecasting {DAYS_AHEAD} days ahead for top {len(top_products)} products...")

all_forecasts = []
for pid in top_products:
    prod_data = df[df["product_id"] == pid].sort_values("date")
    last_row = prod_data.iloc[-1]
    forecast = predictor.predict_future(df, days_ahead=DAYS_AHEAD, product_id=pid)
    forecast["product_id"] = pid
    forecast["product_title"] = last_row["product_title"]
    forecast["category"] = last_row["category"]
    forecast["current_price"] = last_row["price"]
    forecast["current_rating"] = last_row["rating"]
    all_forecasts.append(forecast)

forecasts_df = pd.concat(all_forecasts, ignore_index=True)

# Aggregate by product
print("\n[4/4] Results:\n")
summary = (
    forecasts_df.groupby(["product_id", "product_title", "category", "current_price", "current_rating"])
    .agg(
        total_predicted=("predicted_sales", "sum"),
        avg_daily=("predicted_sales", "mean"),
        avg_lower=("lower_bound", "mean"),
        avg_upper=("upper_bound", "mean"),
    )
    .reset_index()
    .sort_values("total_predicted", ascending=False)
)

summary["predicted_revenue"] = summary["total_predicted"] * summary["current_price"]
summary["confidence_range"] = summary["avg_upper"] - summary["avg_lower"]

# Print summary
print(f"{'Rank':<5} {'Product':<50} {'Category':<18} {'Price':>7} {'30-Day':>8} {'Revenue':>10} {'±Range':>8}")
print("-" * 120)
for i, row in summary.head(TOP_N).iterrows():
    rank = summary.index.get_loc(i) + 1
    title = row['product_title'][:48]
    cat = row['category'][:16]
    print(f"{rank:<5} {title:<50} {cat:<18} ${row['current_price']:>6.2f} {int(row['total_predicted']):>8} ${row['predicted_revenue']:>9,.0f} ±{int(row['confidence_range']):>6}")

# Overall totals
total_units = summary["total_predicted"].sum()
total_revenue = summary["predicted_revenue"].sum()
avg_confidence = summary["confidence_range"].mean()

print("-" * 120)
print(f"\nTOP {TOP_N} PRODUCTS - 30-DAY FORECAST SUMMARY:")
print(f"  Total predicted sales:     {total_units:,.0f} units")
print(f"  Total predicted revenue:   ${total_revenue:,.2f}")
print(f"  Avg daily sales per prod:  {summary['avg_daily'].mean():.1f} units")
print(f"  Avg confidence interval:   ±{avg_confidence:.0f} units")
print(f"  Avg growth trend:          {summary['total_predicted'].mean() / (recent[recent['product_id'].isin(top_products)].groupby('product_id')['sales_units'].sum().mean()):.1%} of recent 60-day avg")

# Category breakdown
print(f"\nCATEGORY BREAKDOWN:")
cat_summary = summary.groupby("category").agg(
    products=("product_id", "nunique"),
    total_sales=("total_predicted", "sum"),
    total_revenue=("predicted_revenue", "sum"),
    avg_price=("current_price", "mean")
).sort_values("total_revenue", ascending=False)
for cat, row in cat_summary.iterrows():
    print(f"  {cat:<20} {int(row['products']):>3} products | {int(row['total_sales']):>8,} units | ${row['total_revenue']:>10,.0f} revenue | ${row['avg_price']:>6.2f} avg price")

# Save to CSV
out_path = "data/forecast_results.csv"
summary.to_csv(out_path, index=False)
print(f"\nFull forecast saved to: {out_path}")
print("=" * 70)
