"""
Feature Engineering Module

Creates predictive features from raw Amazon sales data:
- Lag features (sales from previous days)
- Rolling statistics (moving averages, trends)
- Seasonal features (day of week, month, holidays)
- Product-level features (price elasticity, rating momentum)
- Cross-product features (category averages)
- Interaction features
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates predictive features for sales forecasting.

    Features are organized into groups:
    1. **Lag Features**: sales_lag_1d, sales_lag_7d, sales_lag_30d
    2. **Rolling Statistics**: sales_rolling_mean_7d, sales_rolling_std_7d
    3. **Price Features**: price_ratio, discount_depth, price_elasticity
    4. **Rating Features**: rating_momentum, review_growth
    5. **Temporal Features**: day_of_week, month, quarter, is_holiday_season
    6. **Cross-Product Features**: category_avg_price, category_avg_sales
    7. **Interaction Features**: price * rating, discount * review_count
    """

    def __init__(self):
        self.feature_names: List[str] = []
        self._holiday_dates = self._get_us_holidays()

    def _get_us_holidays(self) -> Dict[str, str]:
        """Get major US shopping holidays (month-day format)."""
        return {
            "new_year": "01-01",
            "valentine": "02-14",
            "easter_approx": "04-01",
            "memorial_day": "05-27",
            "prime_day": "07-15",
            "labor_day": "09-02",
            "halloween": "10-31",
            "black_friday": "11-24",
            "cyber_monday": "11-27",
            "christmas": "12-25",
            "boxing_day": "12-26"
        }

    def create_features(
        self,
        df: pd.DataFrame,
        product_id_col: str = "product_id",
        date_col: str = "date",
        target_col: str = "sales_units",
        price_col: str = "price",
        rating_col: str = "rating",
        review_col: str = "review_count",
        lag_days: List[int] = None,
        rolling_windows: List[int] = None,
        include_advanced: bool = True
    ) -> pd.DataFrame:
        """
        Create all predictive features from raw data.

        Args:
            df: DataFrame with raw sales data per product per day
            product_id_col: Column identifying each product
            date_col: Column with dates
            target_col: Sales column to create lag features for
            lag_days: Days to lag (e.g., [1, 3, 7, 14, 30, 60, 90])
            rolling_windows: Window sizes for rolling statistics
            include_advanced: Whether to include advanced features

        Returns:
            DataFrame with original columns + engineered features
        """
        if lag_days is None:
            lag_days = [1, 3, 7, 14, 30, 60, 90]
        if rolling_windows is None:
            rolling_windows = [7, 14, 30]

        data = df.copy()
        data = data.sort_values([product_id_col, date_col])

        self.feature_names = []

        # Ensure correct types
        data[date_col] = pd.to_datetime(data[date_col])

        # ============================================================
        # 1. TEMPORAL FEATURES
        # ============================================================
        data["day_of_week"] = data[date_col].dt.dayofweek
        data["day_of_month"] = data[date_col].dt.day
        data["day_of_year"] = data[date_col].dt.dayofyear
        data["week_of_year"] = data[date_col].dt.isocalendar().week.astype(int)
        data["month"] = data[date_col].dt.month
        data["quarter"] = data[date_col].dt.quarter
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
        data["is_month_start"] = data[date_col].dt.is_month_start.astype(int)
        data["is_month_end"] = data[date_col].dt.is_month_end.astype(int)

        # Days since / until known shopping events
        # Use the first year in the data to compute day-of-year for holidays
        data_years = data[date_col].dt.year.unique()
        ref_year = max(data_years) if len(data_years) > 0 else 2024
        for holiday_name, holiday_md in self._holiday_dates.items():
            holiday_dt = pd.to_datetime(f"{ref_year}-{holiday_md}")
            doy = holiday_dt.dayofyear
            data[f"days_to_{holiday_name}"] = data["day_of_year"] - doy
            data[f"days_since_{holiday_name}"] = data["day_of_year"] - doy
            # Proximity feature (Gaussian kernel around holiday)
            data[f"holiday_proximity_{holiday_name}"] = np.exp(
                -0.5 * ((data["day_of_year"] - doy) / 5) ** 2
            )

        self.feature_names.extend([
            "day_of_week", "day_of_month", "week_of_year",
            "month", "quarter", "is_weekend",
            "is_month_start", "is_month_end"
        ])
        self.feature_names.extend([
            f"holiday_proximity_{h}" for h in self._holiday_dates
        ])

        # ============================================================
        # 2. LAG FEATURES (per product)
        # ============================================================
        for lag in lag_days:
            data[f"sales_lag_{lag}d"] = data.groupby(product_id_col)[target_col].shift(lag)
            self.feature_names.append(f"sales_lag_{lag}d")

        # Also lag price and rating
        for col in [price_col, rating_col, review_col]:
            for lag in [1, 7, 30]:
                data[f"{col}_lag_{lag}d"] = data.groupby(product_id_col)[col].shift(lag)
                self.feature_names.append(f"{col}_lag_{lag}d")

        # ============================================================
        # 3. ROLLING WINDOW STATISTICS (per product)
        # ============================================================
        for window in rolling_windows:
            # Sales rolling stats
            data[f"sales_rolling_mean_{window}d"] = (
                data.groupby(product_id_col)[target_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            data[f"sales_rolling_std_{window}d"] = (
                data.groupby(product_id_col)[target_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            )
            data[f"sales_rolling_max_{window}d"] = (
                data.groupby(product_id_col)[target_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
            )
            data[f"sales_rolling_min_{window}d"] = (
                data.groupby(product_id_col)[target_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).min())
            )

            # Price rolling average
            data[f"price_rolling_mean_{window}d"] = (
                data.groupby(product_id_col)[price_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

            self.feature_names.extend([
                f"sales_rolling_mean_{window}d",
                f"sales_rolling_std_{window}d",
                f"sales_rolling_max_{window}d",
                f"sales_rolling_min_{window}d",
                f"price_rolling_mean_{window}d"
            ])

        # ============================================================
        # 4. TREND AND ACCELERATION FEATURES
        # ============================================================
        # Sales trend (slope over last 7 days)
        data["sales_trend_7d"] = (
            data[f"sales_lag_1d"] - data[f"sales_lag_7d"]
        ) / (data[f"sales_lag_7d"] + 1)
        self.feature_names.append("sales_trend_7d")

        # Coefficient of variation (relative volatility)
        for window in [7, 30]:
            mean_col = f"sales_rolling_mean_{window}d"
            std_col = f"sales_rolling_std_{window}d"
            data[f"sales_cv_{window}d"] = data[std_col] / (data[mean_col] + 1)
            self.feature_names.append(f"sales_cv_{window}d")

        # ============================================================
        # 5. PRICE FEATURES
        # ============================================================
        if "list_price" in data.columns and price_col in data.columns:
            data["discount_pct"] = (
                (data["list_price"] - data[price_col]) / data["list_price"].clip(lower=0.01) * 100
            )
            self.feature_names.append("discount_pct")

        # Price relative to category average
        if "category" in data.columns:
            cat_price_avg = data.groupby("category")[price_col].transform("mean")
            data["price_vs_category"] = (data[price_col] - cat_price_avg) / cat_price_avg
            self.feature_names.append("price_vs_category")

        # Price change rate (daily)
        data["price_change_1d"] = data.groupby(product_id_col)[price_col].pct_change()
        data["price_change_7d"] = data.groupby(product_id_col)[price_col].pct_change(periods=7)
        self.feature_names.extend(["price_change_1d", "price_change_7d"])

        # ============================================================
        # 6. RATING AND REVIEW FEATURES
        # ============================================================
        if rating_col in data.columns:
            # Rating momentum (change over time)
            data["rating_change_7d"] = (
                data.groupby(product_id_col)[rating_col].diff(7)
            )
            self.feature_names.append("rating_change_7d")

        if review_col in data.columns:
            # Review growth rate
            data["review_growth_7d"] = (
                data.groupby(product_id_col)[review_col].diff(7)
            )
            data["review_growth_30d"] = (
                data.groupby(product_id_col)[review_col].diff(30)
            )
            self.feature_names.extend(["review_growth_7d", "review_growth_30d"])

        # ============================================================
        # 7. CROSS-PRODUCT / CATEGORY FEATURES
        # ============================================================
        if "category" in data.columns:
            # Category-level aggregates (excluding current product)
            cat_agg = data.groupby(["category", date_col]).agg({
                target_col: ["mean", "std"],
                price_col: "mean",
                rating_col: "mean"
            }).fillna(0)
            cat_agg.columns = [
                "cat_avg_sales", "cat_std_sales",
                "cat_avg_price", "cat_avg_rating"
            ]
            cat_agg = cat_agg.reset_index()

            data = data.merge(
                cat_agg,
                on=["category", date_col],
                how="left"
            )

            # Product sales relative to category
            data["sales_vs_category"] = (
                data[target_col] - data["cat_avg_sales"]
            ) / (data["cat_avg_sales"] + 1)

            self.feature_names.extend([
                "cat_avg_sales", "cat_std_sales",
                "cat_avg_price", "cat_avg_rating",
                "sales_vs_category"
            ])

        # ============================================================
        # 8. INTERACTION FEATURES
        # ============================================================
        if include_advanced:
            # Price × Rating interaction
            data["price_rating_interaction"] = data[price_col] * data[rating_col]
            self.feature_names.append("price_rating_interaction")

            # Discount × Review count (deals on popular items)
            if "discount_pct" in data.columns:
                data["discount_review_interaction"] = (
                    data["discount_pct"] * np.log1p(data[review_col])
                )
                self.feature_names.append("discount_review_interaction")

            # Sales velocity relative to price
            data["sales_per_dollar"] = data[target_col] / (data[price_col] + 0.01)
            self.feature_names.append("sales_per_dollar")

            # Days since last deal
            if "days_since_last_deal" in data.columns:
                self.feature_names.append("days_since_last_deal")

            # Weekend × discount (deals on weekends)
            data["weekend_discount"] = data["is_weekend"] * data.get("discount_pct", 0)
            self.feature_names.append("weekend_discount")

        # ============================================================
        # 9. CLEAN UP INFINITE / NaN VALUES
        # ============================================================
        for col in self.feature_names:
            if col in data.columns:
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                data[col] = data[col].fillna(0)

        # Drop rows with NaN in original target (from lag creation)
        data = data.dropna(subset=[target_col])

        logger.info(f"Created {len(self.feature_names)} features")
        logger.debug(f"Features: {self.feature_names}")

        return data

    def get_feature_names(self) -> List[str]:
        """Get list of all created feature names."""
        return self.feature_names.copy()

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get features organized by group."""
        groups = {
            "temporal": [f for f in self.feature_names if any(
                x in f for x in ["day_of", "week_of", "month", "quarter", "is_", "holiday"]
            )],
            "lag": [f for f in self.feature_names if "lag" in f],
            "rolling": [f for f in self.feature_names if "rolling" in f],
            "trend": [f for f in self.feature_names if any(x in f for x in ["trend", "cv_", "change_"])],
            "price": [f for f in self.feature_names if "price" in f or "discount" in f],
            "rating": [f for f in self.feature_names if "rating" in f or "review" in f],
            "cross_product": [f for f in self.feature_names if "cat_" in f or "vs_category" in f],
            "interaction": [f for f in self.feature_names if "interaction" in f or "per_dollar" in f or "weekend_discount" in f],
        }
        return groups
