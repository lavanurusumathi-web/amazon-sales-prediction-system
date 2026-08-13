from .dataset import AmazonSalesDataset, generate_sample_data
from .preprocessor import DataPreprocessor

try:
    from .scraper import AmazonScraper
except ImportError:
    AmazonScraper = None

__all__ = ["AmazonSalesDataset", "generate_sample_data", "DataPreprocessor", "AmazonScraper"]
