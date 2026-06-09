"""
Model Training Module

Trains multiple model types for sales prediction:
- XGBoost Regressor (primary)
- LightGBM Regressor
- Ensemble combining multiple models
- Hyperparameter tuning with Optuna
- Time-series cross-validation
"""
import numpy as np
import pandas as pd
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

logger = logging.getLogger(__name__)


# Optional imports - warn if not available
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")


class ModelTrainer:
    """
    Trains and tunes ML models for sales prediction.

    Supports:
    - XGBoost with hyperparameter tuning
    - LightGBM with hyperparameter tuning
    - Random Forest
    - Ensemble averaging
    - Time-series cross validation
    - Feature importance analysis
    """

    def __init__(
        self,
        model_dir: str = "models",
        random_state: int = 42
    ):
        self.model_dir = model_dir
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.best_model_name: Optional[str] = None
        os.makedirs(model_dir, exist_ok=True)

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        tune_hyperparams: bool = True,
        n_trials: int = 20,
        **kwargs
    ) -> Any:
        """Train XGBoost model with optional hyperparameter tuning."""
        if not XGB_AVAILABLE:
            logger.error("XGBoost is not installed. Cannot train.")
            return None

        logger.info("Training XGBoost model...")

        if tune_hyperparams and OPTUNA_AVAILABLE:
            best_params = self._tune_xgboost(X_train, y_train, X_val, y_val, n_trials)
        else:
            best_params = {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": self.random_state,
            }
            if X_val is not None:
                best_params["early_stopping_rounds"] = 50
                best_params["eval_metric"] = "rmse"

        # Update with any user-provided kwargs
        best_params.update(kwargs)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        model = xgb.XGBRegressor(**best_params, verbosity=0)

        if eval_set:
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        self.models["xgboost"] = model
        logger.info("XGBoost training complete")
        return model

    def _tune_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        n_trials: int = 20
    ) -> Dict[str, Any]:
        """Hyperparameter tuning for XGBoost using Optuna."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
                "random_state": self.random_state,
            }

            if X_val is not None and y_val is not None:
                params["early_stopping_rounds"] = 30
                params["eval_metric"] = "rmse"

                model = xgb.XGBRegressor(**params, verbosity=0)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False
                )
                best_iter = model.best_iteration + 1 if hasattr(model, "best_iteration") else params["n_estimators"]
                y_pred = model.predict(X_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            else:
                # Cross-validation if no validation set
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    model = xgb.XGBRegressor(**params, verbosity=0)
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_v)
                    scores.append(np.sqrt(mean_squared_error(y_v, y_pred)))
                return np.mean(scores)

        study = optuna.create_study(direction="minimize", study_name="xgboost_tuning")
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best XGBoost params: {study.best_params}")
        logger.info(f"Best RMSE: {study.best_value:.4f}")

        return study.best_params

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        tune_hyperparams: bool = True,
        n_trials: int = 20,
        **kwargs
    ) -> Any:
        """Train LightGBM model."""
        if not LGB_AVAILABLE:
            logger.error("LightGBM is not installed. Cannot train.")
            return None

        logger.info("Training LightGBM model...")

        if tune_hyperparams and OPTUNA_AVAILABLE:
            best_params = self._tune_lightgbm(X_train, y_train, X_val, y_val, n_trials)
        else:
            best_params = {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": self.random_state,
                "verbose": -1,
            }

        best_params.update(kwargs)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=eval_set)

        self.models["lightgbm"] = model
        logger.info("LightGBM training complete")
        return model

    def _tune_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        n_trials: int = 20
    ) -> Dict[str, Any]:
        """Hyperparameter tuning for LightGBM."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 2.0),
                "random_state": self.random_state,
                "verbose": -1,
            }

            if X_val is not None and y_val is not None:
                params["early_stopping_rounds"] = 30
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                y_pred = model.predict(X_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            else:
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_v)
                    scores.append(np.sqrt(mean_squared_error(y_v, y_pred)))
                return np.mean(scores)

        study = optuna.create_study(direction="minimize", study_name="lightgbm_tuning")
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best LightGBM params: {study.best_params}")
        logger.info(f"Best RMSE: {study.best_value:.4f}")

        return study.best_params

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> Any:
        """Train a Random Forest model."""
        logger.info("Training Random Forest model...")

        params = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        params.update(kwargs)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        self.models["random_forest"] = model
        logger.info("Random Forest training complete")
        return model

    def train_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        tune: bool = True,
        n_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Train an ensemble of models and find optimal weights.

        Returns dict with 'models' and 'weights'.
        """
        logger.info("Training ensemble of models...")

        # Train individual models
        self.train_xgboost(X_train, y_train, X_val, y_val, tune_hyperparams=tune, n_trials=n_trials)
        self.train_lightgbm(X_train, y_train, X_val, y_val, tune_hyperparams=tune, n_trials=n_trials)
        self.train_random_forest(X_train, y_train)

        # Find optimal weights using validation set
        if X_val is not None and len(self.models) > 1:
            predictions = {}
            for name, model in self.models.items():
                if model is not None:
                    predictions[name] = model.predict(X_val)

            # Simple averaging first, then optimize with grid search
            weights = {name: 1.0 / len(predictions) for name in predictions}
            best_rmse = float("inf")
            best_weights = weights.copy()

            # Coarse grid search for best combination
            for w1 in np.arange(0, 1.1, 0.2):
                for w2 in np.arange(0, 1.1, 0.2):
                    w3 = 1.0 - w1 - w2
                    if w3 < 0:
                        continue
                    names = list(predictions.keys())
                    if len(names) >= 2:
                        w = {names[0]: w1, names[1]: w2}
                        if len(names) >= 3:
                            w[names[2]] = w3

                    ensemble_pred = np.zeros(len(y_val))
                    for name, weight in w.items():
                        ensemble_pred += weight * predictions[name]

                    rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_weights = w

            logger.info(f"Ensemble weights: {best_weights}")
            logger.info(f"Ensemble validation RMSE: {best_rmse:.4f}")

            self.models["ensemble_weights"] = best_weights
        else:
            best_weights = {
                name: 1.0 / len(self.models)
                for name in self.models
            }

        return {
            "models": {k: v for k, v in self.models.items() if v is not None},
            "weights": best_weights
        }

    def get_feature_importance(self, model_name: str) -> Optional[pd.DataFrame]:
        """Get feature importance DataFrame for a trained model."""
        model = self.models.get(model_name)
        if model is None:
            logger.warning(f"Model '{model_name}' not found")
            return None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            logger.warning(f"Model '{model_name}' doesn't expose feature importance")
            return None

        # Try to get feature names
        if hasattr(model, "feature_names_in_"):
            features = model.feature_names_in_
        else:
            features = [f"feature_{i}" for i in range(len(importances))]

        importance_df = pd.DataFrame({
            "feature": features,
            "importance": importances
        }).sort_values("importance", ascending=False)

        # Normalize to percentages
        importance_df["importance_pct"] = (
            importance_df["importance"] / importance_df["importance"].sum() * 100
        )

        self.feature_importance[model_name] = importance_df
        return importance_df

    def save_models(self, tag: Optional[str] = None):
        """Save all trained models to disk."""
        timestamp = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.model_dir, f"models_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        for name, model in self.models.items():
            if model is None or name == "ensemble_weights":
                continue
            if hasattr(model, "save_model"):
                model.save_model(os.path.join(save_dir, f"{name}.json"))
            else:
                joblib.dump(model, os.path.join(save_dir, f"{name}.joblib"))

        # Save ensemble weights separately
        if "ensemble_weights" in self.models and self.models["ensemble_weights"] is not None:
            with open(os.path.join(save_dir, "ensemble_weights.json"), "w") as f:
                json.dump(self.models["ensemble_weights"], f)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "models": [k for k in self.models.keys() if k != "ensemble_weights"],
            "metrics": self.metrics,
            "best_model": self.best_model_name
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Models saved to {save_dir}")
        return save_dir

    def load_models(self, load_dir: str):
        """Load all models from a directory."""
        if not os.path.exists(load_dir):
            logger.error(f"Directory {load_dir} does not exist")
            return

        # Load metadata first to know which files are actual models
        meta_path = os.path.join(load_dir, "metadata.json")
        model_names = set()
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
                self.metrics = metadata.get("metrics", {})
                self.best_model_name = metadata.get("best_model")
                model_names = set(metadata.get("models", []))

        # Load ensemble weights from JSON if present
        weights_path = os.path.join(load_dir, "ensemble_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path) as f:
                self.models["ensemble_weights"] = json.load(f)

        for file in os.listdir(load_dir):
            name = file.replace(".json", "").replace(".joblib", "")
            if file == "metadata.json" or file == "ensemble_weights.json":
                continue
            if file.endswith(".json"):
                if not model_names or name in model_names:
                    model = xgb.XGBRegressor()
                    model.load_model(os.path.join(load_dir, file))
                    self.models[name] = model
            elif file.endswith(".joblib"):
                self.models[name] = joblib.load(os.path.join(load_dir, file))

        logger.info(f"Loaded {len(self.models)} models from {load_dir}")
