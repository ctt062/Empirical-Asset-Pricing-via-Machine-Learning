"""
Download and extract the Gu-Kelly-Xiu (2020) dataset.

This script downloads the datashare.zip file from Dacheng Xiu's website
and extracts the datashare.csv file containing monthly stock returns and
94 firm characteristics for ~30,000 US stocks from 1957-2016.

Dataset URL: https://dachxiu.chicagobooth.edu/download/datashare.zip

Author: Replication of Gu, Kelly, and Xiu (2020)
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import setup_logging, ensure_dir, get_project_root

# Import configuration
from config import DATA_DIR, DATA_RAW_DIR

# Configuration
DATA_URL = "https://dachxiu.chicagobooth.edu/download/datashare.zip"
ZIP_FILE = DATA_RAW_DIR / "datashare.zip"
CSV_FILE = DATA_RAW_DIR / "datashare.csv"

logger = setup_logging()


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """
    Download a file from a URL with progress bar.
    
    Parameters
    ----------
    url : str
        URL to download from
    destination : Path
        Destination file path
    chunk_size : int
        Size of chunks to download (default: 8192 bytes)
    """
    logger.info(f"Downloading from {url}")
    logger.info(f"Destination: {destination}")
    
    try:
        # Send GET request with stream=True
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Create parent directory if needed
        ensure_dir(destination.parent)
        
        # Download with progress bar
        with open(destination, 'wb') as file, tqdm(
            desc=destination.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = file.write(chunk)
                progress_bar.update(size)
        
        logger.info(f"Download completed: {destination}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        raise


def extract_zip(zip_path: Path, extract_to: Path, 
                target_file: Optional[str] = "datashare.csv") -> None:
    """
    Extract a specific file from a zip archive.
    
    Parameters
    ----------
    zip_path : Path
        Path to zip file
    extract_to : Path
        Directory to extract to
    target_file : str, optional
        Specific file to extract. If None, extracts all files.
    """
    logger.info(f"Extracting {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List contents
            file_list = zip_ref.namelist()
            logger.info(f"Zip contains {len(file_list)} file(s)")
            
            if target_file:
                # Extract specific file
                if target_file in file_list:
                    logger.info(f"Extracting {target_file}")
                    zip_ref.extract(target_file, extract_to)
                    logger.info(f"Extracted to {extract_to / target_file}")
                else:
                    logger.warning(f"Target file '{target_file}' not found in zip")
                    logger.info(f"Available files: {file_list}")
                    # Extract all if target not found
                    zip_ref.extractall(extract_to)
            else:
                # Extract all files
                zip_ref.extractall(extract_to)
                logger.info(f"Extracted all files to {extract_to}")
        
    except zipfile.BadZipFile as e:
        logger.error(f"Error extracting zip file: {e}")
        raise


def verify_csv(csv_path: Path) -> None:
    """
    Verify that the CSV file exists and is readable.
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV file
    """
    import pandas as pd
    
    logger.info(f"Verifying CSV file: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read first few rows to verify
    try:
        df_sample = pd.read_csv(csv_path, nrows=5)
        logger.info(f"CSV file is readable")
        logger.info(f"Shape (first 5 rows): {df_sample.shape}")
        logger.info(f"Columns: {df_sample.columns.tolist()[:10]}... (showing first 10)")
        
        # Check for expected columns
        expected_cols = ['permno', 'DATE', 'ret_exc']
        missing = [col for col in expected_cols if col not in df_sample.columns]
        if missing:
            logger.warning(f"Missing expected columns: {missing}")
        else:
            logger.info("All expected columns present")
        
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise


def main() -> None:
    """Main download and extraction pipeline."""
    logger.info("="*80)
    logger.info("Downloading Gu-Kelly-Xiu (2020) Dataset")
    logger.info("="*80)
    
    # Check if CSV already exists
    if CSV_FILE.exists():
        logger.info(f"CSV file already exists: {CSV_FILE}")
        response = input("Do you want to re-download? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Skipping download. Verifying existing file...")
            verify_csv(CSV_FILE)
            return
    
    # Create data directories
    ensure_dir(DATA_RAW_DIR)
    
    # Download zip file
    if not ZIP_FILE.exists():
        try:
            download_file(DATA_URL, ZIP_FILE)
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            sys.exit(1)
    else:
        logger.info(f"Zip file already exists: {ZIP_FILE}")
    
    # Extract CSV
    try:
        extract_zip(ZIP_FILE, DATA_RAW_DIR, target_file="datashare.csv")
    except Exception as e:
        logger.error(f"Failed to extract data: {e}")
        sys.exit(1)
    
    # Verify CSV
    try:
        verify_csv(CSV_FILE)
    except Exception as e:
        logger.error(f"CSV verification failed: {e}")
        sys.exit(1)
    
    # Optional: Remove zip file to save space
    logger.info(f"\nZip file size: {ZIP_FILE.stat().st_size / (1024**2):.1f} MB")
    response = input("Do you want to delete the zip file? (y/n): ").strip().lower()
    if response == 'y':
        ZIP_FILE.unlink()
        logger.info(f"Deleted {ZIP_FILE}")
    
    logger.info("="*80)
    logger.info("Data download and extraction completed successfully!")
    logger.info(f"Data location: {CSV_FILE}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
