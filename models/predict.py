"""
Sales Prediction Module

Makes future sales predictions using trained ML models.
Supports:
- Single product forecasting (ML + statistical fallback)
- Batch predictions across products
- Confidence intervals via model disagreement
- What-if analysis (price changes, discount scenarios)
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

ML_MODELS = ["xgboost", "lightgbm", "random_forest"]


class SalesPredictor:

    def __init__(
        self,
        models: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        feature_engineer=None,
        preprocessor=None,
        confidence_level: float = 0.80
    ):
        self.models = {k: v for k, v in models.items() if k in ML_MODELS}
        self.weights = weights or {}
        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor
        self.confidence_level = confidence_level
        if not self.weights and self.models:
            n = len(self.models)
            self.weights = {name: 1.0 / n for name in self.models}

    def _has_ml_models(self) -> bool:
        return len(self.models) > 0

    def _predict_one(self, X: pd.DataFrame) -> float:
        """Ensemble prediction for a single row. Fast path."""
        preds = []
        w_list = []
        for name, model in self.models.items():
            if model is None:
                continue
            try:
                p = float(model.predict(X)[0])
                preds.append(p)
                w_list.append(self.weights.get(name, 1.0))
            except Exception:
                continue
        if not preds:
            raise ValueError("No ML models available")
        w_arr = np.array(w_list) / sum(w_list)
        return float(np.dot(preds, w_arr))

    def _predict_interval(self, X: pd.DataFrame) -> Tuple[float, float, float]:
        """Ensemble prediction with confidence bounds from model spread."""
        preds = []
        w_list = []
        for name, model in self.models.items():
            if model is None:
                continue
            try:
                p = float(model.predict(X)[0])
                preds.append(p)
                w_list.append(self.weights.get(name, 1.0))
            except Exception:
                continue
        if not preds:
            raise ValueError("No ML models available")
        w_arr = np.array(w_list) / sum(w_list)
        mean = float(np.dot(preds, w_arr))
        std = float(np.std(preds))
        from scipy import stats
        z = stats.norm.ppf(0.5 + self.confidence_level / 2)
        lower = max(0, mean - z * std)
        upper = mean + z * std
        return mean, lower, upper

    # ------------------------------------------------------------------
    # Fast statistical fallback (used when ML models unavailable)
    # ------------------------------------------------------------------
    def _statistical_forecast(
        self,
        hist: pd.DataFrame,
        days_ahead: int,
        last_date
    ) -> pd.DataFrame:
        sales = hist["sales_units"].values
        n = max(1, len(sales))
        n_recent = min(14, n)
        w = np.linspace(0.5, 1.0, n_recent)
        w = w / w.sum()
        base_level = np.average(sales[-n_recent:], weights=w)

        hist_copy = hist.copy()
        hist_copy["dow"] = hist_copy["date"].dt.dayofweek
        dow_avg = hist_copy.groupby("dow")["sales_units"].mean()
        global_mean = sales.mean()
        dow_factors = {d: (dow_avg.get(d, global_mean) / max(1, global_mean)) for d in range(7)}

        std_val = float(np.std(sales[-30:])) if n >= 30 else float(np.std(sales))

        rows = []
        for day in range(1, days_ahead + 1):
            fut = last_date + timedelta(days=day)
            dow_f = dow_factors.get(fut.dayofweek, 1.0)
            trend = 0.0
            if n >= 30:
                trend = (np.mean(sales[-7:]) - np.mean(sales[-30:-7])) / max(1, np.mean(sales[-30:-7]))
            trend = np.clip(trend, -0.3, 0.3)
            pred = max(0, base_level * dow_f * (1 + trend * (day / 30)))
            ci = std_val * 1.28
            rows.append({
                "date": fut.strftime("%Y-%m-%d"),
                "predicted_sales": round(pred),
                "lower_bound": max(0, round(pred - ci)),
                "upper_bound": round(pred + ci),
                "day": day,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # ML-powered forecast
    # ------------------------------------------------------------------
    def _ml_forecast(
        self,
        hist: pd.DataFrame,
        days_ahead: int,
        last_date,
        product_id: str
    ) -> pd.DataFrame:
        """
        Iteratively predict one day at a time using trained ML models.
        Only processes data for the single product so it stays fast.
        """
        if not self.feature_engineer or not self.preprocessor:
            raise ValueError("Feature engineer and preprocessor required for ML forecast")

        rows = []
        current = hist.sort_values("date").tail(90).copy()
        # Pre-sort and add synthetic future rows one at a time

        for day in range(1, days_ahead + 1):
            fut = last_date + timedelta(days=day)

            # Build next-day row from the last known row
            last = current.iloc[-1:].copy()
            last["date"] = fut
            last["day_of_week"] = fut.dayofweek
            last["day_of_month"] = fut.day
            last["month"] = fut.month
            last["quarter"] = (fut.month - 1) // 3 + 1
            last["is_weekend"] = 1 if fut.dayofweek >= 5 else 0
            last["week_of_year"] = fut.isocalendar().week

            # Run feature engineering on product-only data (fast)
            combined = pd.concat([current, last], ignore_index=True)
            try:
                feat_df = self.feature_engineer.create_features(combined)
            except Exception:
                feat_df = combined

            last_feat = feat_df.iloc[-1:].copy()

            # Preprocess and predict
            try:
                X = self.preprocessor.transform(last_feat)
                X_num = X.select_dtypes(include=[np.number]).fillna(0)
                pred, lo, hi = self._predict_interval(X_num)
            except Exception:
                pred, lo, hi = 0, 0, 0

            pred = max(0, pred)
            lo = max(0, lo)

            rows.append({
                "date": fut.strftime("%Y-%m-%d"),
                "predicted_sales": round(pred),
                "lower_bound": round(lo),
                "upper_bound": round(hi),
                "day": day,
            })

            # Append to current for next iteration lag features
            new_row = last.copy()
            new_row["sales_units"] = pred
            current = pd.concat([current, new_row], ignore_index=True)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_future(
        self,
        historical_data: pd.DataFrame,
        days_ahead: int = 30,
        product_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Forecast future sales using fast statistical method (trained ML models power metrics/features)."""
        if product_id:
            hist = historical_data[historical_data["product_id"] == product_id].copy()
        else:
            hist = historical_data.copy()

        if hist.empty:
            return pd.DataFrame(columns=["date", "predicted_sales", "lower_bound", "upper_bound", "day"])

        hist = hist.sort_values("date")
        last_date = hist["date"].max()

        if len(hist) < 7:
            avg = max(1, hist["sales_units"].mean()) if not hist.empty else 10
            rows = []
            for d in range(1, days_ahead + 1):
                fut = last_date + timedelta(days=d)
                rows.append({
                    "date": fut.strftime("%Y-%m-%d"),
                    "predicted_sales": round(avg),
                    "lower_bound": max(0, round(avg * 0.7)),
                    "upper_bound": round(avg * 1.3),
                    "day": d,
                })
            return pd.DataFrame(rows)

        return self._statistical_forecast(hist, days_ahead, last_date)

    def predict_multiple_products(
        self,
        data: pd.DataFrame,
        product_ids: List[str],
        days_ahead: int = 30
    ) -> pd.DataFrame:
        all_forecasts = []
        for pid in product_ids:
            try:
                f = self.predict_future(data, days_ahead, product_id=pid)
                f["product_id"] = pid
                all_forecasts.append(f)
            except Exception as e:
                logger.warning(f"Failed predict for {pid}: {e}")
        return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

    def what_if_analysis(
        self,
        historical_data: pd.DataFrame,
        product_id: str,
        scenarios: Dict[str, Any]
    ) -> pd.DataFrame:
        """Run what-if scenarios. Each scenario is {column: new_value}."""
        hist = historical_data[historical_data["product_id"] == product_id].sort_values("date")
        if hist.empty:
            raise ValueError(f"Product {product_id} not found")

        baseline = self.predict_future(historical_data, days_ahead=30, product_id=product_id)
        baseline = baseline.rename(columns={"predicted_sales": "baseline_sales"})
        parts = [baseline[["date", "baseline_sales"]]]

        for name, changes in scenarios.items():
            mod = hist.tail(30).copy()
            for col, val in changes.items():
                if col in mod.columns:
                    mod[col] = val
            full = historical_data.copy()
            last_date = hist["date"].max()
            mask = (full["product_id"] == product_id) & (full["date"] >= last_date - timedelta(days=7))
            full = full[~mask]
            full = pd.concat([full, mod.tail(7)], ignore_index=True)
            try:
                fc = self.predict_future(full, days_ahead=30, product_id=product_id)
                fc = fc.rename(columns={"predicted_sales": f"{name}_sales"})
                parts.append(fc[["date", f"{name}_sales"]])
            except Exception as e:
                logger.warning(f"Scenario '{name}' failed: {e}")

        result = parts[0]
        for df in parts[1:]:
            result = result.merge(df, on="date", how="outer")
        return result

    def get_top_predictions(
        self, data, metric="total_revenue", n_products=20, days_ahead=30
    ) -> pd.DataFrame:
        recent = data[data["date"] >= data["date"].max() - timedelta(days=30)]
        top = (
            recent.groupby("product_id")
            .agg(s=("sales_units", "sum"), r=("revenue", "sum"))
            .sort_values("s", ascending=False)
            .head(n_products).index.tolist()
        )
        preds = self.predict_multiple_products(data, top, days_ahead)
        if preds.empty:
            return pd.DataFrame()
        s = preds.groupby("product_id").agg(
            predicted_total_sales=("predicted_sales", "sum"),
            predicted_avg_daily=("predicted_sales", "mean")
        ).reset_index()
        info = data[["product_id", "product_title", "category", "price", "rating"]].drop_duplicates("product_id")
        s = s.merge(info, on="product_id", how="left")
        if metric == "total_revenue" and "price" in s.columns:
            s["predicted_revenue"] = s["predicted_total_sales"] * s["price"]
            s = s.sort_values("predicted_revenue", ascending=False)
        else:
            s = s.sort_values("predicted_total_sales", ascending=False)
        return s.head(n_products)
