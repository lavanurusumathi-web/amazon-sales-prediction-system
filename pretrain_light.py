"""Lightweight pre-training for Render free tier (512MB RAM)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import AmazonSalesDataset
from features.engineering import FeatureEngineer
from data.preprocessor import DataPreprocessor
from models.train import ModelTrainer

N_PRODUCTS = 100
N_DAYS = 90
TRAIN_SAMPLE = 8000

print(f"Generating dataset ({N_PRODUCTS} products x {N_DAYS} days)...")
ds = AmazonSalesDataset(data_dir="data")
df = ds.generate(n_products=N_PRODUCTS, n_days=N_DAYS, force=True)
print(f"  {len(df):,} rows, {df['product_id'].nunique()} products")

print("Feature engineering...")
fe = FeatureEngineer()
df_sample = df.sort_values("date").tail(min(TRAIN_SAMPLE, len(df))).copy()
feat_df = fe.create_features(df_sample)
print(f"  Shape: {feat_df.shape}")

print("Preprocessing...")
pp = DataPreprocessor()
processed = pp.fit_transform(feat_df)
X, y = pp.prepare_for_training(processed, target_col="sales_units")

split_idx = int(len(X) * 0.8)
val_idx = int(split_idx * 0.875)
X_tr, y_tr = X.iloc[:val_idx], y.iloc[:val_idx]
X_val, y_val = X.iloc[val_idx:split_idx], y.iloc[val_idx:split_idx]
print(f"  Train: {len(X_tr):,} | Val: {len(X_val):,}")

print("Training models...")
t0 = time.time()
trainer = ModelTrainer(model_dir="models")
trainer.train_ensemble(X_tr, y_tr, X_val, y_val, tune=False)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s")
print(f"  Models: {list(trainer.models.keys())}")

save_path = trainer.save_models(tag="pretrained")
print(f"Saved: {save_path}")
print("DONE!")
