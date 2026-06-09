"""
Model Evaluation Module

Evaluates sales prediction models with comprehensive metrics and visualizations.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    explained_variance_score
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates sales prediction model performance.

    Provides:
    - Regression metrics (RMSE, MAE, MAPE, R2, EV)
    - Time-series specific metrics (bias, tracking signal)
    - Error distribution analysis
    - Category-level performance breakdown
    - Comparison across models
    - Report generation
    """

    def __init__(self):
        self.results: Dict[str, Dict] = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        dataset_name: str = "test",
        y_true_original_scale: Optional[np.ndarray] = None,
        y_pred_original_scale: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate predictions with multiple metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            dataset_name: Name of dataset (train/val/test)
            y_true_original_scale: Actual values in original (unscaled) units
            y_pred_original_scale: Predicted values in original units

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Core regression metrics
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["mape"] = float(self._mape(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["explained_variance"] = float(explained_variance_score(y_true, y_pred))
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))

        # Additional metrics
        metrics["bias"] = float(np.mean(y_pred - y_true))
        metrics["bias_pct"] = float(np.mean((y_pred - y_true) / (y_true + 1) * 100))
        metrics["max_error"] = float(np.max(np.abs(y_true - y_pred)))
        metrics["median_abs_error"] = float(np.median(np.abs(y_true - y_pred)))

        # If original scale is provided, compute metrics in original units too
        if y_true_original_scale is not None and y_pred_original_scale is not None:
            metrics["rmse_original"] = float(np.sqrt(mean_squared_error(y_true_original_scale, y_pred_original_scale)))
            metrics["mae_original"] = float(mean_absolute_error(y_true_original_scale, y_pred_original_scale))
            metrics["mape_original"] = float(self._mape(y_true_original_scale, y_pred_original_scale))

        # Store results
        key = f"{model_name}_{dataset_name}"
        self.results[key] = metrics

        logger.info(f"[{model_name} - {dataset_name}] RMSE: {metrics['rmse']:.4f}, "
                     f"MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.2f}%, "
                     f"R2: {metrics['r2']:.4f}")

        return metrics

    def _mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error, handling zeros."""
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def evaluate_by_category(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        categories: np.ndarray,
        model_name: str = "model"
    ) -> pd.DataFrame:
        """
        Evaluate performance broken down by product category.

        Returns:
            DataFrame with metrics per category
        """
        results = []
        unique_cats = np.unique(categories)

        for cat in unique_cats:
            mask = categories == cat
            if mask.sum() < 5:
                continue

            cat_true = y_true[mask]
            cat_pred = y_pred[mask]

            results.append({
                "category": cat,
                "count": mask.sum(),
                "rmse": np.sqrt(mean_squared_error(cat_true, cat_pred)),
                "mae": mean_absolute_error(cat_true, cat_pred),
                "mape": self._mape(cat_true, cat_pred),
                "r2": r2_score(cat_true, cat_pred),
                "bias": np.mean(cat_pred - cat_true),
                "actual_mean": np.mean(cat_true),
                "pred_mean": np.mean(cat_pred),
            })

        result_df = pd.DataFrame(results).sort_values("rmse")
        self.results[f"{model_name}_by_category"] = result_df

        return result_df

    def evaluate_by_time_period(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: np.ndarray,
        period: str = "month",
        model_name: str = "model"
    ) -> pd.DataFrame:
        """
        Evaluate performance broken down by time period.

        Args:
            dates: Array of date strings/datetimes
            period: 'month', 'week', or 'quarter'

        Returns:
            DataFrame with metrics per time period
        """
        df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "date": pd.to_datetime(dates)
        })

        if period == "month":
            df["period"] = df["date"].dt.strftime("%Y-%m")
        elif period == "week":
            df["period"] = df["date"].dt.strftime("%Y-W%W")
        elif period == "quarter":
            df["period"] = df["date"].dt.to_period("Q").astype(str)
        else:
            df["period"] = df["date"].dt.strftime("%Y-%m")

        results = df.groupby("period").apply(
            lambda g: pd.Series({
                "count": len(g),
                "rmse": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
                "mae": mean_absolute_error(g["y_true"], g["y_pred"]),
                "mape": self._mape(g["y_true"].values, g["y_pred"].values),
                "bias": np.mean(g["y_pred"] - g["y_true"]),
                "actual_mean": np.mean(g["y_true"]),
                "pred_mean": np.mean(g["y_pred"]),
            })
        ).reset_index()

        key = f"{model_name}_by_{period}"
        self.results[key] = results

        return results

    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models side by side."""
        rows = []
        for key, metrics in self.results.items():
            if isinstance(metrics, dict) and not key.endswith("_by_category") and not "_by_" in key:
                row = {"model_dataset": key, **metrics}
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values("rmse")

    def get_best_model(self, metric: str = "rmse") -> Optional[str]:
        """Get the name of the best-performing model."""
        comparison = self.compare_models()
        if comparison.empty:
            return None
        return comparison.loc[comparison[metric].idxmin(), "model_dataset"]

    def summary_report(self) -> str:
        """Generate a human-readable evaluation summary."""
        comparison = self.compare_models()

        lines = []
        lines.append("=" * 60)
        lines.append("MODEL EVALUATION SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        if not comparison.empty:
            lines.append("Model Performance Comparison:")
            lines.append("-" * 40)
            for _, row in comparison.iterrows():
                lines.append(
                    f"  {row['model_dataset']:30s} | "
                    f"RMSE: {row['rmse']:7.2f} | "
                    f"MAE: {row['mae']:7.2f} | "
                    f"MAPE: {row['mape']:5.1f}% | "
                    f"R²: {row['r2']:.4f}"
                )
            lines.append("")

            best = comparison.loc[comparison["rmse"].idxmin()]
            lines.append(f"🏆 Best model: {best['model_dataset']} (RMSE: {best['rmse']:.4f})")

        return "\n".join(lines)

    def error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Detailed error analysis."""
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        pct_errors = np.where(y_true != 0, abs_errors / y_true * 100, 0)

        return {
            "error_stats": {
                "mean_error": float(np.mean(errors)),
                "std_error": float(np.std(errors)),
                "skewness": float(pd.Series(errors).skew()),
                "kurtosis": float(pd.Series(errors).kurtosis()),
            },
            "percentiles": {
                "p5": float(np.percentile(abs_errors, 5)),
                "p25": float(np.percentile(abs_errors, 25)),
                "p50": float(np.percentile(abs_errors, 50)),
                "p75": float(np.percentile(abs_errors, 75)),
                "p95": float(np.percentile(abs_errors, 95)),
            },
            "accuracy_thresholds": {
                "within_10pct": float(np.mean(pct_errors <= 10) * 100),
                "within_25pct": float(np.mean(pct_errors <= 25) * 100),
                "within_50pct": float(np.mean(pct_errors <= 50) * 100),
            },
            "outliers": {
                "n_outliers_3sigma": int(np.sum(np.abs(errors - np.mean(errors)) > 3 * np.std(errors))),
                "outlier_pct": float(np.mean(np.abs(errors - np.mean(errors)) > 3 * np.std(errors)) * 100),
            }
        }
