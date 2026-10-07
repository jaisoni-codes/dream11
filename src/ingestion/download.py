import os
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw"
CRICSHEET_URLS = [
    # Download a small subset of T20s to keep processing fast for the milestone
    "https://cricsheet.org/downloads/t20s_json.zip"
]

def download_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    for url in CRICSHEET_URLS:
        filename = url.split('/')[-1]
        filepath = os.path.join(RAW_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.info(f"Downloading {url} to {filepath}...")
            urllib.request.urlretrieve(url, filepath)
            logger.info(f"Downloaded {filename}.")
        else:
            logger.info(f"File {filename} already exists. Skipping download.")
            
        # Extract
        extract_dir = os.path.join(RAW_DIR, filename.replace('.zip', ''))
        if not os.path.exists(extract_dir):
            logger.info(f"Extracting {filename} to {extract_dir}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"Extracted {filename}.")
        else:
            logger.info(f"Extracted directory {extract_dir} already exists. Skipping extraction.")

if __name__ == "__main__":
    download_data()
