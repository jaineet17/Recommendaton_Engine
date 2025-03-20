"""
Module for downloading the Amazon product reviews dataset.
"""

import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Define constants
DEFAULT_DATASET_URL = "https://example.com/amazon-product-reviews.zip"  # Placeholder URL
DEFAULT_SAVE_PATH = Path("data/raw")


def download_file(url: str, save_path: Path, chunk_size: int = 8192) -> Path:
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        save_path: Directory to save the file
        chunk_size: Size of chunks to download
        
    Returns:
        Path to the downloaded file
    """
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    
    local_filename = save_path / url.split('/')[-1]
    
    logger.info(f"Downloading {url} to {local_filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(local_filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            progress_bar.update(len(chunk))
            file.write(chunk)
    
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logger.warning("WARNING: Downloaded size doesn't match expected size!")
    
    return local_filename


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extract a zip file to a directory.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
    """
    logger.info(f"Extracting {zip_path} to {extract_to}")
    
    if not extract_to.exists():
        extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Extracting"):
            zip_ref.extract(file, extract_to)


def download_amazon_dataset(url: Optional[str] = None, 
                           save_path: Optional[Path] = None,
                           extract: bool = True) -> Path:
    """
    Download and optionally extract the Amazon product reviews dataset.
    
    Args:
        url: URL to download from (default: DEFAULT_DATASET_URL)
        save_path: Directory to save the file (default: data/raw)
        extract: Whether to extract the downloaded file
        
    Returns:
        Path to the downloaded/extracted dataset
    """
    url = url or DEFAULT_DATASET_URL
    save_path = save_path or DEFAULT_SAVE_PATH
    
    try:
        # Download the dataset
        zip_path = download_file(url, save_path)
        
        # Extract if requested
        if extract:
            extract_dir = save_path / "amazon_reviews"
            extract_zip(zip_path, extract_dir)
            return extract_dir
        
        return zip_path
    
    except Exception as e:
        logger.error(f"Error downloading Amazon dataset: {e}")
        raise


def clean_up(path: Path, keep_zip: bool = False) -> None:
    """
    Clean up temporary files after downloading and extracting.
    
    Args:
        path: Path to the file or directory to remove
        keep_zip: Whether to keep the zip file
    """
    logger.info(f"Cleaning up {path}")
    
    if path.is_file():
        if path.suffix == '.zip' and keep_zip:
            logger.info(f"Keeping zip file: {path}")
            return
        
        os.remove(path)
    elif path.is_dir():
        shutil.rmtree(path)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Download and extract the dataset
    logger.info("Starting dataset download")
    dataset_path = download_amazon_dataset()
    logger.info(f"Dataset downloaded and extracted to {dataset_path}") 