"""Quick pre-training: generate small dataset, train fast models, save to disk."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import AmazonSalesDataset
from features.engineering import FeatureEngineer
from data.preprocessor import DataPreprocessor
from models.train import ModelTrainer

print("Generating small training dataset...")
ds = AmazonSalesDataset(data_dir="data")
df = ds.generate(n_products=2000, n_days=365, force=True)
print(f"  {len(df):,} rows, {df['product_id'].nunique()} products")

print("Feature engineering...")
fe = FeatureEngineer()
feat_df = fe.create_features(df)
print(f"  {feat_df.shape}")

print("Preprocessing...")
pp = DataPreprocessor()
processed = pp.fit_transform(feat_df)
X, y = pp.prepare_for_training(processed, target_col="sales_units")

split_idx = int(len(X) * 0.8)
val_idx = int(split_idx * 0.875)
X_tr, y_tr = X.iloc[:val_idx], y.iloc[:val_idx]
X_val, y_val = X.iloc[val_idx:split_idx], y.iloc[val_idx:split_idx]

print("Training models (no tuning)...")
t0 = time.time()
trainer = ModelTrainer(model_dir="models")
trainer.train_ensemble(X_tr, y_tr, X_val, y_val, tune=False)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Models: {list(trainer.models.keys())}")
save_path = trainer.save_models(tag="pretrained")
print(f"Saved to: {save_path}")
print("DONE!")
