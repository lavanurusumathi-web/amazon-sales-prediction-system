"""
Data Preprocessing Module

Handles cleaning, normalization, encoding, and preparation of Amazon sales data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses Amazon sales data for machine learning.

    Handles:
    - Missing value imputation
    - Outlier detection and treatment
    - Categorical encoding (label, one-hot, target)
    - Numerical scaling
    - Feature selection / filtering
    """

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[RobustScaler] = None
        self.target_encodings: Dict[str, Dict] = {}
        self._fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "sales_units",
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        date_col: str = "date",
        id_col: str = "product_id",
        scale: bool = True
    ) -> pd.DataFrame:
        """Fit preprocessor and transform data."""
        return self._fit_transform(df, target_col, categorical_cols, numerical_cols, date_col, id_col, scale, fit=True)

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str = "sales_units",
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        date_col: str = "date",
        id_col: str = "product_id",
        scale: bool = True
    ) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        return self._fit_transform(df, target_col, categorical_cols, numerical_cols, date_col, id_col, scale, fit=False)

    def _fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "sales_units",
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        date_col: str = "date",
        id_col: str = "product_id",
        scale: bool = True,
        fit: bool = True
    ) -> pd.DataFrame:
        """Internal method for fit and/or transform."""
        data = df.copy()

        # Default column classifications
        if categorical_cols is None:
            categorical_cols = [
                col for col in ["category", "subcategory", "brand", "is_weekend", "is_deal", "quarter", "month"]
                if col in data.columns
            ]

        if numerical_cols is None:
            exclude = {date_col, id_col, target_col, *categorical_cols}
            numerical_cols = [
                col for col in data.select_dtypes(include=[np.number]).columns
                if col not in exclude
            ]

        # Handle missing values
        for col in numerical_cols:
            if col in data.columns and data[col].isna().any():
                median_val = data[col].median() if fit else 0
                data[col] = data[col].fillna(median_val)
                logger.debug(f"Filled {data[col].isna().sum()} missing values in {col}")

        for col in categorical_cols:
            if col in data.columns and data[col].isna().any():
                mode_val = data[col].mode().iloc[0] if fit else "unknown"
                data[col] = data[col].fillna(mode_val)
                logger.debug(f"Filled missing values in {col}")

        # Handle outliers in numerical columns (cap at 99th percentile)
        for col in numerical_cols:
            if col in data.columns and col != target_col:
                if fit:
                    upper = data[col].quantile(0.99)
                    data[col] = data[col].clip(upper=upper)
                else:
                    pass  # Use same cap as fit - stored in scaler bounds
                # Also cap outliers at the lower end
                if fit:
                    lower = data[col].quantile(0.01)
                    data[col] = data[col].clip(lower=lower)

        # Encode categorical variables
        for col in categorical_cols:
            if col in data.columns:
                if data[col].dtype == "object" or data[col].dtype == "category":
                    if fit:
                        le = LabelEncoder()
                        data[f"{col}_encoded"] = le.fit_transform(data[col].astype(str))
                        self.label_encoders[col] = le
                    else:
                        le = self.label_encoders.get(col)
                        if le:
                            # Handle unseen categories
                            data[col] = data[col].astype(str)
                            known_classes = set(le.classes_)
                            data.loc[~data[col].isin(known_classes), col] = "unknown"
                            if "unknown" not in le.classes_:
                                le.classes_ = list(le.classes_) + ["unknown"]
                            data[f"{col}_encoded"] = le.transform(data[col])

        # Scale numerical features
        if scale and numerical_cols:
            existing_num_cols = [c for c in numerical_cols if c in data.columns]
            if existing_num_cols:
                if fit:
                    self.scaler = RobustScaler()
                    data[existing_num_cols] = self.scaler.fit_transform(data[existing_num_cols])
                else:
                    if self.scaler:
                        data[existing_num_cols] = self.scaler.transform(data[existing_num_cols])

        self._fitted = True
        return data

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_col: str = "sales_units",
        feature_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Final preparation: separate features and target."""
        if exclude_cols is None:
            exclude_cols = ["date", "product_id", "product_title", target_col]

        data = df.copy()

        if feature_cols:
            available = [c for c in feature_cols if c in data.columns]
            X = data[available]
        else:
            X = data.drop(columns=[c for c in exclude_cols if c in data.columns], errors="ignore")
            # Keep only numeric columns
            X = X.select_dtypes(include=[np.number])

        y = data[target_col] if target_col in data.columns else None

        # Final NA check
        X = X.fillna(0)

        return X, y
