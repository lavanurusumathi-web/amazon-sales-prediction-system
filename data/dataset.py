"""
Amazon Sales Dataset Module

Provides realistic synthetic Amazon product data (since real sales data is proprietary)
and a Kaggle dataset downloader for when users have access to real data.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import json
import os
import logging

logger = logging.getLogger(__name__)

# Realistic Amazon product categories and subcategories
PRODUCT_CATEGORIES = {
    "Electronics": ["Smartphones", "Laptops", "Headphones", "Tablets", "Smartwatches", "Cameras", "Speakers"],
    "Home & Kitchen": ["Cookware", "Small Appliances", "Home Decor", "Furniture", "Bedding", "Storage"],
    "Books": ["Fiction", "Non-Fiction", "Educational", "Children's Books", "Comics"],
    "Clothing & Accessories": ["Men's Clothing", "Women's Clothing", "Shoes", "Accessories", "Kids' Fashion"],
    "Sports & Outdoors": ["Exercise Equipment", "Camping Gear", "Team Sports", "Cycling", "Fitness Trackers"],
    "Beauty & Personal Care": ["Skincare", "Hair Care", "Makeup", "Fragrances", "Grooming"],
    "Toys & Games": ["Action Figures", "Board Games", "Educational Toys", "Video Games", "Puzzles"],
    "Automotive": ["Car Accessories", "Tools", "Maintenance", "Electronics", "Interior"],
    "Health & Household": ["Vitamins", "Household Supplies", "Medical Supplies", "Wellness", "Cleaning"],
    "Office Products": ["Office Furniture", "Stationery", "Printer Supplies", "Desk Accessories"]
}

# Brand names by category
BRANDS = {
    "Electronics": ["TechPro", "AudioMax", "VisionGo", "ConnectPlus", "SmartWave", "InnoTech", "PowerUp"],
    "Home & Kitchen": ["HomeChef", "KitchenArt", "CozyHome", "EliteCook", "FreshLiving"],
    "Books": ["PageTurner Press", "KnowledgeFirst", "StoryCraft", "LearnWell Publishing"],
    "Clothing & Accessories": ["StyleWear", "FashionFirst", "TrendSetter", "UrbanFit", "ClassicThreads"],
    "Sports & Outdoors": ["FitGear", "OutdoorPro", "EnduroMax", "SportFlex", "TrailBlazer"],
    "Beauty & Personal Care": ["GloSkin", "PureBeauty", "NaturaEssence", "RadiancePro"],
    "Toys & Games": ["FunFactory", "PlayWorld", "CreativeMinds", "GameOn"],
    "Automotive": ["AutoPro", "DriveMaster", "CarCare", "RoadWarrior"],
    "Health & Household": ["WellnessPlus", "CleanLife", "VitalHealth", "PureLiving"],
    "Office Products": ["WorkSmart", "OfficeElite", "DeskPro", "EfficiencyPlus"]
}


def generate_sample_data(
    n_products: int = 5000,
    n_days_history: int = 365 * 2,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate a realistic synthetic Amazon product dataset with daily sales data.

    The data captures realistic patterns:
    - Power-law distribution of sales (few products sell a lot, most sell a little)
    - Price elasticity (discounts boost sales)
    - Seasonality (holidays, day-of-week effects)
    - Review effects (more/better reviews → more sales)
    - Category-specific trends

    Args:
        n_products: Number of unique products to generate
        n_days_history: Number of days of historical data per product
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns:
        - date, product_id, product_title, category, subcategory, brand
        - price, list_price, discount_pct
        - rating, review_count, sentiment_score
        - sales_units, revenue, best_seller_rank
        - is_weekend, month, day_of_week, day_of_month, quarter
        - days_since_last_deal, avg_price_last_30d, sales_lag_7d, etc.
    """
    rng = np.random.default_rng(random_seed)

    # Generate product catalog
    products = []
    for pid in range(1, n_products + 1):
        cat = rng.choice(list(PRODUCT_CATEGORIES.keys()))
        subcat = rng.choice(PRODUCT_CATEGORIES[cat])
        brand = rng.choice(BRANDS[cat])

        # Base price varies by category
        base_price = {
            "Electronics": rng.uniform(15, 1500),
            "Home & Kitchen": rng.uniform(8, 400),
            "Books": rng.uniform(5, 60),
            "Clothing & Accessories": rng.uniform(10, 250),
            "Sports & Outdoors": rng.uniform(12, 500),
            "Beauty & Personal Care": rng.uniform(5, 120),
            "Toys & Games": rng.uniform(8, 200),
            "Automotive": rng.uniform(10, 350),
            "Health & Household": rng.uniform(5, 100),
            "Office Products": rng.uniform(8, 300)
        }[cat]

        title = f"{brand} {subcat} - Premium Quality #{pid}"

        products.append({
            "product_id": f"AMZ{pid:06d}",
            "product_title": title,
            "category": cat,
            "subcategory": subcat,
            "brand": brand,
            "base_price": round(base_price, 2),
            "base_rating": round(min(5.0, max(1.0, rng.normal(3.8, 0.8))), 1),
            "base_review_count": int(max(0, rng.pareto(2.0) * 50)),
            "base_sales_velocity": max(0.1, rng.pareto(1.5) * 20)
        })

    products_df = pd.DataFrame(products)

    # Generate daily time series for each product
    rows = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days_history)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Seasonal multipliers
    holidays = {
        "New Year": ("01-01", 1.8),
        "Valentine's Day": ("02-14", 1.4),
        "Easter": ("03-31", 1.3),
        "Prime Day": ("07-15", 2.5),
        "Back to School": ("08-20", 1.6),
        "Halloween": ("10-31", 1.5),
        "Black Friday": ("11-24", 3.0),
        "Cyber Monday": ("11-27", 2.8),
        "Christmas": ("12-25", 2.2)
    }

    # Convert holidays to day-of-year
    holiday_doy = {}
    for name, (date_str, _) in holidays.items():
        dt = datetime.strptime(date_str, "%m-%d")
        holiday_doy[name] = dt.timetuple().tm_yday

    logger.info(f"Generating {n_products} products x {len(date_range)} days of data...")

    for _, product in products_df.iterrows():
        pid = product["product_id"]
        base_velocity = product["base_sales_velocity"]

        # Individual product seasonality pattern
        product_season_amplitude = rng.uniform(0.2, 0.6)
        product_season_phase = rng.uniform(0, 2 * np.pi)
        product_trend = rng.uniform(-0.0003, 0.0003)  # slight growth/decline
        product_noise_scale = rng.uniform(0.1, 0.4)

        # Price fluctuation parameters
        price_volatility = rng.uniform(0.05, 0.15)
        discount_frequency = rng.choice([0, 0.05, 0.1, 0.15], p=[0.3, 0.3, 0.25, 0.15])
        current_price = product["base_price"]

        # Rating drifts slowly over time
        current_rating = product["base_rating"]
        current_reviews = product["base_review_count"]

        # Track if we're in a deal period
        days_since_deal = 90
        recent_prices = [current_price] * 30

        for i, date in enumerate(date_range):
            day_of_year = date.timetuple().tm_yday
            month = date.month
            day_of_week = date.dayofweek
            is_weekend = day_of_week >= 5

            # --- Price dynamics ---
            # Random price changes
            price_change = rng.normal(0, price_volatility)
            current_price *= (1 + price_change)
            current_price = max(product["base_price"] * 0.5, min(product["base_price"] * 1.5, current_price))

            # Occasional discounts
            is_deal = rng.random() < discount_frequency
            if is_deal:
                deal_depth = rng.uniform(0.1, 0.4)
                deal_price = current_price * (1 - deal_depth)
                days_since_deal = 0
            else:
                deal_price = current_price
                days_since_deal += 1

            list_price = max(current_price, deal_price)
            discount_pct = (list_price - deal_price) / list_price * 100

            recent_prices.append(deal_price)
            recent_prices = recent_prices[-30:]

            # --- Seasonality factors ---
            # Day-of-week effect
            dow_factor = 1.0 + {0: -0.1, 1: -0.15, 2: -0.15, 3: -0.1, 4: 0.0, 5: 0.2, 6: 0.15}.get(day_of_week, 0)

            # Holiday effect
            holiday_factor = 1.0
            for name, (_, multiplier) in holidays.items():
                # Effect spreads over ~10 days centered on the holiday
                days_to_holiday = abs(day_of_year - holiday_doy[name])
                if days_to_holiday < 10:
                    holiday_factor += (multiplier - 1) * (1 - days_to_holiday / 10) * 0.5
                # Also check if the holiday was in the past year
                days_to_holiday_alt = abs(day_of_year - holiday_doy[name] - 365)
                if days_to_holiday_alt < 10:
                    holiday_factor += (multiplier - 1) * (1 - days_to_holiday_alt / 10) * 0.3

            # Yearly seasonality
            season_factor = 1.0 + product_season_amplitude * np.sin(
                2 * np.pi * day_of_year / 365 + product_season_phase
            )

            # --- Rating & review dynamics ---
            # Small random walk in rating
            current_rating += rng.normal(0, 0.02)
            current_rating = min(5.0, max(1.0, current_rating))

            # Reviews accumulate over time
            review_inflow = max(0, int(rng.poisson(max(0.05, base_velocity * 0.1))))
            current_reviews += review_inflow

            # --- Sales calculation ---
            # Price elasticity: sales decrease as price increases
            price_ratio = deal_price / product["base_price"]
            price_elasticity = np.exp(-0.8 * (price_ratio - 1))  # ~20% drop for 25% price increase
            if is_deal:
                price_elasticity *= (1 + deal_depth * 1.5)  # deals boost sales

            # Rating effect: higher rating → more sales
            rating_effect = 1.0 + (current_rating - 3.5) * 0.3

            # Review count effect: more reviews → more sales (diminishing returns)
            review_effect = 1.0 + np.log10(max(1, current_reviews)) * 0.1

            # Trend effect
            trend_effect = 1.0 + product_trend * i

            # Random noise
            noise = 1.0 + rng.normal(0, product_noise_scale)

            # Calculate daily sales
            daily_sales = (
                base_velocity
                * dow_factor
                * holiday_factor
                * season_factor
                * price_elasticity
                * rating_effect
                * review_effect
                * trend_effect
                * noise
            )
            daily_sales = max(0, int(round(daily_sales)))

            # Best Seller Rank (inverse of sales, with noise)
            bsr = max(1, int(10000 / max(1, daily_sales) * rng.uniform(0.5, 2.0)))

            # Sentiment score from rating
            sentiment = (current_rating - 1) / 4  # 0-1 scale

            rows.append({
                "date": date,
                "product_id": pid,
                "product_title": product["product_title"],
                "category": product["category"],
                "subcategory": product["subcategory"],
                "brand": product["brand"],
                "price": round(deal_price, 2),
                "list_price": round(list_price, 2),
                "discount_pct": round(discount_pct, 1),
                "rating": round(current_rating, 1),
                "review_count": current_reviews,
                "sentiment_score": round(sentiment, 3),
                "sales_units": daily_sales,
                "revenue": round(daily_sales * deal_price, 2),
                "best_seller_rank": bsr,
                "is_weekend": is_weekend,
                "month": month,
                "day_of_week": day_of_week,
                "day_of_year": day_of_year,
                "quarter": (month - 1) // 3 + 1,
                "is_deal": is_deal,
                "days_since_last_deal": days_since_deal,
                "avg_price_last_30d": round(np.mean(recent_prices), 2)
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    logger.info(f"Generated dataset: {len(df):,} rows, {df['product_id'].nunique():,} products")
    logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"Total sales: {df['sales_units'].sum():,} units")
    logger.info(f"Total revenue: ${df['revenue'].sum():,.2f}")

    return df


class AmazonSalesDataset:
    """
    Central data loading and management class.

    Can:
    - Generate synthetic data for development/testing
    - Load pre-downloaded Kaggle datasets
    - Cache data to disk for faster reloading
    - Split into train/val/test time-based sets
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._data: Optional[pd.DataFrame] = None
        self._cache_path = os.path.join(data_dir, "amazon_sales_data.parquet")

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise ValueError(
                "Data not loaded. Call load() or generate() first."
            )
        return self._data

    def generate(
        self,
        n_products: int = 3000,
        n_days: int = 730,
        seed: int = 42,
        force: bool = False
    ) -> pd.DataFrame:
        """Generate and cache synthetic dataset."""
        if os.path.exists(self._cache_path) and not force:
            logger.info(f"Loading cached data from {self._cache_path}")
            self._data = pd.read_parquet(self._cache_path)
            return self._data

        self._data = generate_sample_data(
            n_products=n_products,
            n_days_history=n_days,
            random_seed=seed
        )

        os.makedirs(self.data_dir, exist_ok=True)
        self._data.to_parquet(self._cache_path, index=False)
        logger.info(f"Cached data to {self._cache_path}")

        return self._data

    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load data from a CSV file (e.g., Kaggle dataset)."""
        self._data = pd.read_csv(csv_path, parse_dates=["date"])
        logger.info(f"Loaded {len(self._data):,} rows from {csv_path}")
        return self._data

    def load_from_parquet(self, path: str) -> pd.DataFrame:
        """Load from a parquet file."""
        self._data = pd.read_parquet(path)
        logger.info(f"Loaded {len(self._data):,} rows from {path}")
        return self._data

    def train_val_test_split(
        self,
        val_split: float = 0.15,
        test_split: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-based train/val/test split (no leakage).

        Splits chronologically: oldest data → train, middle → val, newest → test.
        """
        df = self.data.sort_values("date")
        n = len(df)
        test_cutoff = int(n * (1 - test_split))
        val_cutoff = int(n * (1 - test_split - val_split))

        train = df.iloc[:val_cutoff].copy()
        val = df.iloc[val_cutoff:test_cutoff].copy()
        test = df.iloc[test_cutoff:].copy()

        logger.info(
            f"Split: train={len(train):,} ({train['date'].min().date()} to "
            f"{train['date'].max().date()}), val={len(val):,}, test={len(test):,}"
        )

        return train, val, test

    def get_product_series(self, product_id: str) -> pd.DataFrame:
        """Get time series for a single product."""
        return self.data[self.data["product_id"] == product_id].sort_values("date")

    def get_top_products(self, n: int = 50, by: str = "revenue") -> pd.DataFrame:
        """Get top N products by total sales or revenue."""
        if by == "sales":
            agg = self.data.groupby("product_id").agg(
                total_sales=("sales_units", "sum"),
                product_title=("product_title", "first"),
                category=("category", "first"),
                price=("price", "mean"),
                rating=("rating", "mean")
            ).sort_values("total_sales", ascending=False).head(n)
        else:
            agg = self.data.groupby("product_id").agg(
                total_revenue=("revenue", "sum"),
                product_title=("product_title", "first"),
                category=("category", "first"),
                price=("price", "mean"),
                rating=("rating", "mean")
            ).sort_values("total_revenue", ascending=False).head(n)
        return agg.reset_index()

    def get_category_summary(self) -> pd.DataFrame:
        """Get aggregate statistics by category."""
        return self.data.groupby("category").agg(
            product_count=("product_id", "nunique"),
            total_sales=("sales_units", "sum"),
            total_revenue=("revenue", "sum"),
            avg_price=("price", "mean"),
            avg_rating=("rating", "mean"),
            avg_discount=("discount_pct", "mean")
        ).round(2).sort_values("total_revenue", ascending=False)

    def get_daily_totals(self) -> pd.DataFrame:
        """Get daily aggregate sales across all products."""
        return self.data.groupby("date").agg(
            total_sales=("sales_units", "sum"),
            total_revenue=("revenue", "sum"),
            avg_price=("price", "mean"),
            avg_discount=("discount_pct", "mean"),
            product_count=("product_id", "nunique")
        ).reset_index().sort_values("date")
