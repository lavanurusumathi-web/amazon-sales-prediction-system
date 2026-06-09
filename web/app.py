"""
FastAPI Web Dashboard for Amazon Sales Prediction System

Provides:
- Sales dashboard with key metrics and visualizations
- Product browsing with search and filter
- Sales predictions with confidence intervals
- What-if analysis (price/discount scenarios)
- Feature importance viewer
- Category performance breakdown
"""
import os
import sys
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import json
from data.dataset import AmazonSalesDataset
from data.preprocessor import DataPreprocessor
from features.engineering import FeatureEngineer
from models.train import ModelTrainer, XGB_AVAILABLE
from models.predict import SalesPredictor
from models.evaluate import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Amazon Sales Prediction System", version="1.0.0")

# Setup templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(templates_dir))

# Global state (initialized on startup)
dataset: Optional[AmazonSalesDataset] = None
preprocessor: Optional[DataPreprocessor] = None
feature_engineer: Optional[FeatureEngineer] = None
trainer: Optional[ModelTrainer] = None
predictor: Optional[SalesPredictor] = None
evaluator: Optional[ModelEvaluator] = None
is_trained: bool = False
is_training: bool = False
training_progress: float = 0.0
training_history: List[Dict] = []
amazon_scraper: Any = None

executor = ThreadPoolExecutor(max_workers=1)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@app.on_event("startup")
async def startup():
    """Initialize the system, loading pre-trained models if available."""
    global dataset, preprocessor, feature_engineer, trainer, evaluator, predictor, is_trained, amazon_scraper

    data_dir = get_project_root() / "data"
    model_dir = get_project_root() / "models"

    logger.info("Initializing Amazon Sales Prediction System...")

    # Initialize components
    dataset = AmazonSalesDataset(data_dir=str(data_dir))
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    trainer = ModelTrainer(model_dir=str(model_dir))
    evaluator = ModelEvaluator()

    # Try to generate sample data if none exists
    try:
        df = dataset.generate(n_products=2000, n_days=365)
        logger.info(f"Loaded/generated dataset: {len(df):,} rows")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")

    # Try to load pre-trained models
    pretrain_dir = model_dir / "models_pretrained"
    if pretrain_dir.exists():
        try:
            logger.info("Loading pre-trained models...")
            if XGB_AVAILABLE:
                import xgboost as xgb
                xgb_path = pretrain_dir / "xgboost.json"
                if xgb_path.exists():
                    model = xgb.XGBRegressor()
                    model.load_model(str(xgb_path))
                    trainer.models["xgboost"] = model

            lgb_path = pretrain_dir / "lightgbm.joblib"
            if lgb_path.exists():
                trainer.models["lightgbm"] = joblib.load(str(lgb_path))

            rf_path = pretrain_dir / "random_forest.joblib"
            if rf_path.exists():
                trainer.models["random_forest"] = joblib.load(str(rf_path))

            ew_path = pretrain_dir / "ensemble_weights.joblib"
            if ew_path.exists():
                trainer.models["ensemble_weights"] = joblib.load(str(ew_path))

            # Build predictor
            weights = trainer.models.get("ensemble_weights", {})
            predictor = SalesPredictor(
                models={k: v for k, v in trainer.models.items() if v is not None and k != "ensemble_weights"},
                weights=weights,
                feature_engineer=feature_engineer,
                preprocessor=preprocessor,
            )
            is_trained = True
            logger.info(f"Pre-trained models loaded: {list(trainer.models.keys())}")
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            is_trained = False

    # Initialize persistent Selenium scraper (reuse across requests)
    try:
        logger.info("Starting Chrome scraper (headless)...")
        from data.scraper import AmazonScraper
        amazon_scraper = AmazonScraper()
        amazon_scraper._get_driver()
        logger.info("Chrome scraper ready")
    except Exception as e:
        logger.warning(f"Chrome scraper not available: {e}")
        amazon_scraper = None


# ============================================================
# API Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    global dataset, is_trained

    stats = {}
    category_data = pd.DataFrame()
    daily_trends = pd.DataFrame()
    top_products = pd.DataFrame()

    if dataset and dataset._data is not None:
        df = dataset.data

        # Basic stats
        stats = {
            "total_products": int(df["product_id"].nunique()),
            "total_sales": int(df["sales_units"].sum()),
            "total_revenue": float(df["revenue"].sum()),
            "avg_daily_sales": float(df.groupby("date")["sales_units"].sum().mean()),
            "avg_rating": float(df["rating"].mean()),
            "avg_price": float(df["price"].mean()),
            "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
            "n_categories": int(df["category"].nunique()),
        }

        # Category breakdown
        category_data = dataset.get_category_summary().head(10).reset_index().to_dict("records")

        # Daily trends (last 60 days)
        daily = dataset.get_daily_totals()
        daily = daily.tail(60).copy()
        daily["date"] = daily["date"].astype(str)
        daily_trends = daily.to_dict("records")

        # Top products
        top_products = dataset.get_top_products(20, "revenue").to_dict("records")
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
           "request": request,
           "stats": stats,
           "category_data": category_data,
           "daily_trends": daily_trends,
           "top_products": top_products,
           "is_trained": is_trained,
           "active_page": "dashboard"
    }
)
       
@app.get("/products", response_class=HTMLResponse)
async def products_page(
    request: Request,
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    sort: str = Query("revenue"),
    page: int = Query(1, ge=1)
):
    """Product listing page."""
    global dataset

    products = []
    categories = []
    total_pages = 1

    if dataset and dataset._data is not None:
        df = dataset.data
        categories = sorted(df["category"].unique().tolist())

        # Get latest data per product
        latest = df.sort_values("date").groupby("product_id").last().reset_index()

        # Filter
        if search:
            latest = latest[latest["product_title"].str.contains(search, case=False, na=False)]
        if category:
            latest = latest[latest["category"] == category]

        # Sort
        if sort == "sales":
            total_sales = df.groupby("product_id")["sales_units"].sum().reset_index()
            latest = latest.merge(total_sales, on="product_id", suffixes=("", "_total"))
            latest = latest.sort_values("sales_units_total", ascending=False)
        elif sort == "price":
            latest = latest.sort_values("price", ascending=False)
        elif sort == "rating":
            latest = latest.sort_values("rating", ascending=False)
        else:  # revenue
            total_rev = df.groupby("product_id")["revenue"].sum().reset_index()
            latest = latest.merge(total_rev, on="product_id", suffixes=("", "_total"))
            latest = latest.sort_values("revenue_total", ascending=False)

        # Pagination
        per_page = 24
        total = len(latest)
        total_pages = max(1, (total + per_page - 1) // per_page)
        start = (page - 1) * per_page
        end = start + per_page

        page_products = latest.iloc[start:end]

        products = []
        for _, p in page_products.iterrows():
            products.append({
                "id": p["product_id"],
                "title": p["product_title"],
                "category": p["category"],
                "subcategory": p["subcategory"],
                "brand": p["brand"],
                "price": float(p["price"]),
                "list_price": float(p.get("list_price", p["price"])),
                "rating": float(p["rating"]),
                "review_count": int(p.get("review_count", 0)),
                "sales": int(p.get("sales_units", 0)),
            })
    return templates.TemplateResponse(
        request=request,
        name="products.html",
        context={
            "request": request,
            "products": products,
            "categories": categories,
            "selected_category": category or "",
            "search_query": search or "",
            "sort_by": sort,
            "page": page,
            "total_pages": total_pages,
            "active_page": "products"
    }
)


@app.get("/predictions", response_class=HTMLResponse)
async def predictions_page(
    request: Request,
    product_id: Optional[str] = Query(None),
    days: int = Query(30, ge=7, le=365)
):
    """Sales predictions page."""
    global dataset, predictor, is_trained

    forecast = []
    product_info = {}
    what_if_scenarios = []
    products_list = []

    if dataset and dataset._data is not None:
        df = dataset.data
        products_list = (
            df.groupby("product_id")["product_title"]
            .first()
            .reset_index()
            .to_dict("records")
        )

        if product_id and predictor and is_trained:
            try:
                # Get product info
                product_data = df[df["product_id"] == product_id]
                if not product_data.empty:
                    p = product_data.iloc[-1]
                    product_info = {
                        "id": p["product_id"],
                        "title": p["product_title"],
                        "category": p["category"],
                        "price": float(p["price"]),
                        "rating": float(p["rating"]),
                    }

                # Get forecast
                forecast_df = predictor.predict_future(df, days_ahead=days, product_id=product_id)
                forecast = forecast_df.to_dict("records")
                # Pre-calculate totals for template (Jinja2 can't do this natively)
                forecast_total = sum(f.get("predicted_sales", 0) or 0 for f in forecast)

                # What-if scenarios
                scenarios = {
                    "Price +20%": {"price": product_info.get("price", 0) * 1.2},
                    "Price -20%": {"price": product_info.get("price", 0) * 0.8},
                    "Best Rating (5.0)": {"rating": 5.0},
                }
                scenario_df = predictor.what_if_analysis(df, product_id, scenarios)
                what_if_scenarios = scenario_df.to_dict("records")

            except Exception as e:
                logger.warning(f"Prediction failed for {product_id}: {e}")
    return templates.TemplateResponse(
        request=request,
        name="predictions.html",
        context={
            "request": request,
            "forecast": forecast,
            "product_info": product_info,
            "product_id": product_id or "",
            "days": days,
            "products_list": products_list,
            "is_trained": is_trained,
            "what_if_scenarios": what_if_scenarios,
            "active_page": "predictions"
        }
    )

def _run_training():
    """Run the full training pipeline in a background thread."""
    global is_trained, is_training, training_progress, predictor, trainer, evaluator

    try:
        df = dataset.data

        training_progress = 0.05

        # Use a subset of recent data for faster training
        df_tail = df.sort_values("date").tail(100000).copy()
        feat_df = feature_engineer.create_features(df_tail)

        training_progress = 0.20
        logger.info(f"Created {len(feature_engineer.get_feature_names())} features")

        processed = preprocessor.fit_transform(feat_df)
        training_progress = 0.30

        X, y = preprocessor.prepare_for_training(processed, target_col="sales_units")

        split_idx = int(len(X) * 0.8)
        val_idx = int(split_idx * 0.875)

        X_train, y_train = X.iloc[:val_idx], y.iloc[:val_idx]
        X_val, y_val = X.iloc[val_idx:split_idx], y.iloc[val_idx:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

        training_progress = 0.35
        logger.info("Training ensemble (no Optuna tuning)...")
        ensemble = trainer.train_ensemble(X_train, y_train, X_val, y_val, tune=False)

        training_progress = 0.80
        logger.info("Evaluating models...")
        for name, model in trainer.models.items():
            if model is None or name == "ensemble_weights":
                continue
            try:
                y_pred_test = model.predict(X_test)
                evaluator.evaluate(
                    y_test.values, y_pred_test,
                    model_name=name, dataset_name="test"
                )
            except Exception as e:
                logger.warning(f"Evaluation failed for {name}: {e}")

        training_progress = 0.90
        weights = trainer.models.get("ensemble_weights", {})
        predictor = SalesPredictor(
            models={k: v for k, v in trainer.models.items() if v is not None and k != "ensemble_weights"},
            weights=weights,
            feature_engineer=feature_engineer,
            preprocessor=preprocessor,
        )
        is_trained = True

        save_path = trainer.save_models()
        logger.info(f"Models saved to {save_path}")
        training_progress = 1.0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        is_training = False


@app.post("/api/train")
async def train_models():
    """Train ML models in a background thread."""
    global dataset, is_training

    if not dataset or dataset._data is None:
        raise HTTPException(400, "No dataset loaded. Generate data first.")
    if is_training:
        raise HTTPException(409, "Training already in progress")

    is_training = True
    executor.submit(_run_training)
    return JSONResponse({"success": True, "message": "Training started in background"})


@app.get("/api/training-status")
async def training_status():
    """Get training status, progress, and evaluation results."""
    global is_trained, is_training, training_progress, trainer, evaluator

    metrics = []
    top_features = []

    if is_trained and evaluator:
        try:
            metrics = evaluator.compare_models().to_dict("records")
        except:
            pass

    if is_trained and trainer and "xgboost" in trainer.models:
        imp = trainer.get_feature_importance("xgboost")
        if imp is not None:
            top_features = imp.head(20).to_dict("records")

    return JSONResponse({
        "trained": is_trained,
        "training": is_training,
        "progress": training_progress,
        "models": list(trainer.models.keys()) if is_trained and trainer else [],
        "metrics": metrics,
        "feature_importance": top_features,
    })


@app.get("/api/product/{product_id}")
async def get_product(product_id: str):
    """Get product details and history."""
    global dataset

    if not dataset or dataset._data is None:
        raise HTTPException(404, "No data loaded")

    df = dataset.data
    product_df = df[df["product_id"] == product_id]

    if product_df.empty:
        raise HTTPException(404, f"Product {product_id} not found")

    # Product info (latest values)
    latest = product_df.sort_values("date").iloc[-1]
    info = {
        "id": latest["product_id"],
        "title": latest["product_title"],
        "category": latest["category"],
        "subcategory": latest["subcategory"],
        "brand": latest["brand"],
        "price": float(latest["price"]),
        "rating": float(latest["rating"]),
        "review_count": int(latest.get("review_count", 0)),
        "total_sales": int(product_df["sales_units"].sum()),
        "total_revenue": float(product_df["revenue"].sum()),
    }

    # Sales history (last 90 days)
    history = product_df.tail(90)[["date", "sales_units", "price", "rating"]].to_dict("records")

    return JSONResponse({"product": info, "history": history})


@app.get("/api/predict/{product_id}")
async def predict_product(product_id: str, days: int = Query(30)):
    """Get sales predictions for a product."""
    global dataset, predictor, is_trained

    if not is_trained or not predictor:
        raise HTTPException(400, "Models not trained yet. Click 'Train Models' first.")

    try:
        forecast_df = predictor.predict_future(dataset.data, days_ahead=days, product_id=product_id)
        return JSONResponse({
            "product_id": product_id,
            "forecast": forecast_df.to_dict("records")
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics."""
    global dataset

    if not dataset or dataset._data is None:
        return JSONResponse({"error": "No data"})

    df = dataset.data

    stats = {
        "total_products": int(df["product_id"].nunique()),
        "total_sales": int(df["sales_units"].sum()),
        "total_revenue": float(df["revenue"].sum()),
        "avg_daily_sales": float(df.groupby("date")["sales_units"].sum().mean()),
        "avg_rating": float(df["rating"].mean()),
        "avg_price": float(df["price"].mean()),
        "categories": int(df["category"].nunique()),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
    }

    return JSONResponse(stats)


@app.get("/api/category-breakdown")
async def category_breakdown():
    """Get sales breakdown by category."""
    global dataset

    if not dataset or dataset._data is None:
        return JSONResponse({"error": "No data"})

    summary = dataset.get_category_summary()
    result = summary.reset_index().to_dict("records")
    # Ensure all values are JSON-safe
    for row in result:
        for k, v in list(row.items()):
            if hasattr(v, 'item'):  # numpy types
                row[k] = v.item()
    return JSONResponse(result)


@app.get("/api/daily-trends")
async def daily_trends(days: int = Query(90)):
    """Get daily sales trends."""
    global dataset

    if not dataset or dataset._data is None:
        return JSONResponse({"error": "No data"})

    daily = dataset.get_daily_totals().tail(days).copy()
    daily["date"] = daily["date"].astype(str)
    return JSONResponse(daily.to_dict("records"))


@app.get("/api/regenerate-data")
async def regenerate_data(n_products: int = Query(2000), n_days: int = Query(365)):
    """Regenerate the sample dataset."""
    global dataset

    if not dataset:
        raise HTTPException(500, "Dataset not initialized")

    df = dataset.generate(n_products=n_products, n_days=n_days, force=True)

    return JSONResponse({
        "success": True,
        "rows": len(df),
        "products": int(df["product_id"].nunique()),
        "message": f"Generated {len(df):,} rows for {df['product_id'].nunique():,} products"
    })


@app.get("/api/search-products")
async def search_products(q: str = Query(""), limit: int = Query(20)):
    """Search products by name."""
    global dataset

    if not dataset or dataset._data is None:
        return JSONResponse([])

    df = dataset.data
    latest = df.sort_values("date").groupby("product_id").last().reset_index()

    if q:
        mask = latest["product_title"].str.contains(q, case=False, na=False)
        latest = latest[mask]

    results = latest.head(limit)[
        ["product_id", "product_title", "category", "price", "rating"]
    ].to_dict("records")

    return JSONResponse(results)


# ============================================================
# Live Scraper Page
# ============================================================

@app.get("/scraper", response_class=HTMLResponse)
async def scraper_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="scraper.html",
        context={"active_page": "scraper"}
    )


@app.post("/api/scrape")
async def scrape_product(asin: str = Form(...)):
    global amazon_scraper
    if amazon_scraper is None:
        raise HTTPException(503, "Live scraper unavailable (Chrome/browser not available on this server)")
    try:
        result = amazon_scraper.get_product(asin)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/scrape/search")
async def scrape_search(q: str = Query(...), limit: int = Query(10)):
    global amazon_scraper
    if amazon_scraper is None:
        raise HTTPException(503, "Live scraper unavailable (Chrome/browser not available on this server)")
    try:
        results = amazon_scraper.search(q, max_results=limit)
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================
# Static Files (MUST be last — routes defined after mount get swallowed)
# ============================================================

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
