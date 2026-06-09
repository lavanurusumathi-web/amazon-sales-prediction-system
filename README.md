# 🎯 Amazon Sales Prediction System

A comprehensive machine learning system that predicts future sales of Amazon products using historical data, gradient boosting models (XGBoost, LightGBM), and time-series forecasting techniques.

## ✨ Features

- **📊 Interactive Dashboard** - Real-time visualizations of sales trends, category breakdowns, and top products
- **🤖 ML-Powered Predictions** - Ensemble of XGBoost, LightGBM, and Random Forest models with hyperparameter tuning
- **🎯 Sales Forecasting** - Predict future sales with confidence intervals for any product
- **🔮 What-If Analysis** - Simulate how price changes, discounts, or rating changes affect sales
- **📦 Product Catalog** - Browse and search products with detailed metrics
- **🕸️ Live Scraper** - Fetch real Amazon product data for predictions on current products
- **🏆 Feature Importance** - Understand which factors drive sales the most
- **⚡ Fast Performance** - Built with FastAPI for low-latency predictions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Navigate to the project directory
cd amazon-sales-predictor

# Install required packages
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
python run.py
```

This will:
1. Generate a realistic synthetic Amazon product dataset (2000 products × 365 days)
2. Launch the web dashboard at **http://127.0.0.1:8000**
3. Open your browser automatically

### 3. Train Models

1. Open the dashboard in your browser
2. Click the **"🚀 Train Models"** button in the top-right corner
3. Wait for training to complete (1-5 minutes depending on your hardware)
4. The system will train XGBoost, LightGBM, and Random Forest models

### 4. Make Predictions

1. Navigate to the **"🎯 Predictions"** page
2. Select a product from the dropdown
3. Choose your forecast horizon (7, 14, 30, 60, or 90 days)
4. View the interactive forecast chart with confidence intervals
5. Experiment with **What-If scenarios** to see how changes affect sales

## 📁 Project Structure

```
amazon-sales-predictor/
├── data/
│   ├── dataset.py         # Dataset generation & loading
│   ├── preprocessor.py    # Data cleaning & preprocessing
│   └── scraper.py         # Amazon product scraper
├── features/
│   └── engineering.py     # Feature engineering (lags, rolling stats)
├── models/
│   ├── train.py           # Model training & hyperparameter tuning
│   ├── predict.py         # Prediction pipeline
│   └── evaluate.py        # Model evaluation
├── web/
│   ├── app.py             # FastAPI web application
│   └── templates/         # HTML templates
│       ├── dashboard.html # Main dashboard
│       ├── products.html  # Product catalog
│       └── predictions.html # Prediction interface
├── run.py                 # Main entry point
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🧠 How It Works

### Data Generation

The system generates a realistic synthetic Amazon product dataset that captures:
- **Power-law sales distribution** - A few products sell a lot, most sell a little
- **Price elasticity** - Discounts boost sales; price increases reduce them
- **Seasonality** - Holiday effects (Black Friday, Christmas, Prime Day)
- **Day-of-week patterns** - Weekends drive more sales
- **Rating dynamics** - Ratings drift over time with new reviews
- **Category-specific trends** - Different categories have different base prices and velocities

### Feature Engineering

The system creates over 100 predictive features:

| Feature Group | Examples |
|--------------|----------|
| **Temporal** | Day of week, month, quarter, holiday proximity, weekend flag |
| **Lag** | Sales from 1, 3, 7, 14, 30, 60, 90 days ago |
| **Rolling** | 7/14/30-day moving average, std dev, min, max |
| **Price** | Discount %, price vs category avg, price change rate |
| **Rating** | Rating change, review growth rate |
| **Cross-product** | Category average sales, price, rating |
| **Interaction** | Price × rating, discount × reviews, sales per dollar |

### Models

The system trains an ensemble of:

1. **XGBoost** - Gradient boosting with tree-based learners
   - Hyperparameter tuning via Optuna (max_depth, learning_rate, subsample, etc.)
   - Early stopping to prevent overfitting

2. **LightGBM** - Light gradient boosting machine
   - Leaf-wise tree growth with depth constraints
   - Optimized for speed and memory efficiency

3. **Random Forest** - Ensemble of decision trees
   - Bagging with feature randomization
   - Robust baseline model

The ensemble combines models using optimized weights found via grid search on the validation set.

### Evaluation Metrics

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)
- **Bias** (Systematic over/under prediction)

## 🔧 Advanced Usage

### Generate Only Dataset

```bash
python run.py --generate-only --n-products 5000 --n-days 730
```

### Custom Host/Port

```bash
python run.py --host 0.0.0.0 --port 8080
```

### Using Real Amazon Data

The scraper module can fetch live Amazon product data:

```python
from data.scraper import AmazonScraper

scraper = AmazonScraper()

# Search for products
results = scraper.search("wireless headphones", max_results=10)

# Get detailed product info
product = scraper.get_product("B08N5WRWNW")
```

> **Note**: Respect Amazon's robots.txt and terms of service. The scraper uses polite delays between requests.

### Using a Kaggle Dataset

```python
from data.dataset import AmazonSalesDataset

ds = AmazonSalesDataset()
df = ds.load_from_csv("path/to/amazon_dataset.csv")
```

## 📊 Sample Dataset

The synthetic dataset includes:

- **2,000+ products** across 10 categories
- **365 days** of daily sales history
- Realistic patterns: seasonality, price elasticity, rating effects
- **10+ categories**: Electronics, Home & Kitchen, Books, Clothing, Sports, Beauty, Toys, Automotive, Health, Office

## 🐍 Python Version

Requires Python 3.9+

## 📦 Key Dependencies

- **pandas, numpy** - Data manipulation
- **xgboost, lightgbm** - Gradient boosting models
- **scikit-learn** - ML utilities and evaluation
- **optuna** - Hyperparameter optimization
- **fastapi, uvicorn** - Web framework
- **plotly, chart.js** - Visualizations
- **beautifulsoup4, selenium** - Web scraping

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
