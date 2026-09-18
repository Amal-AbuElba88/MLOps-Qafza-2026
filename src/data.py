import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """Load raw input data for inference."""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise e


def validate_input_schema(df: pd.DataFrame, required_columns: list) -> bool:
    """Check if all required columns exist in the input dataframe."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Input data is missing columns: {missing_cols}")
    logger.info("Input schema validation passed successfully.")
    return True
