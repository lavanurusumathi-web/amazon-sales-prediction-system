#!/usr/bin/env python3
"""
Amazon Sales Prediction System
Main entry point - launch the web dashboard.
"""
import os
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("amazon-sales-predictor")


def check_dependencies():
    """Check that required dependencies are installed."""
    required = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("xgboost", "xgboost"),
        ("sklearn", "scikit-learn"),
    ]

    missing = []
    for import_name, package_name in required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        logger.error("Missing required dependencies:")
        for pkg in missing:
            logger.error(f"  - {pkg}")
        logger.info(f"\nInstall with: pip install {' '.join(missing)}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Amazon Sales Prediction System"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't open browser automatically"
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Only generate the dataset and exit"
    )
    parser.add_argument(
        "--n-products", type=int, default=2000,
        help="Number of products to generate (default: 2000)"
    )
    parser.add_argument(
        "--n-days", type=int, default=365,
        help="Days of history per product (default: 365)"
    )

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    if args.generate_only:
        logger.info(f"Generating dataset with {args.n_products} products...")
        from data.dataset import AmazonSalesDataset
        ds = AmazonSalesDataset(data_dir="data")
        df = ds.generate(n_products=args.n_products, n_days=args.n_days, force=True)
        logger.info(f"Generated {len(df):,} rows for {df['product_id'].nunique():,} products")
        logger.info(f"Saved to data/amazon_sales_data.parquet")
        return

    # Launch web server
    import uvicorn

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║     Amazon Sales Prediction System          ║")
    logger.info("╠══════════════════════════════════════════════╣")
    logger.info(f"║  Dashboard:  http://{args.host}:{args.port}/   ║")
    logger.info("║                                              ║")
    logger.info("║  1. Generate dataset (auto on startup)       ║")
    logger.info("║  2. Click 'Train Models' on dashboard        ║")
    logger.info("║  3. Explore predictions under Predictions    ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # Open browser
    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"http://{args.host}:{args.port}/")

    # Start the server
    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
