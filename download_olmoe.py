import os
import sys
from pathlib import Path
import logging
from typing import Optional
import shutil
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('download_log.txt')
    ]
)
logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, cache_dir: str = "./olmoe_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.model_dir = self.cache_dir / "models"
        self.data_dir = self.cache_dir / "data"
        self.tokenizer_dir = self.cache_dir / "tokenizer"
        
        for dir_path in [self.model_dir, self.data_dir, self.tokenizer_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Track download status
        self.status_file = self.cache_dir / "download_status.txt"
        self.downloaded = self._load_status()
    
    def _load_status(self) -> dict:
        """Load download status from file"""
        if not self.status_file.exists():
            return {}
        
        status = {}
        with open(self.status_file, 'r') as f:
            for line in f:
                component = line.strip()
                status[component] = True
        return status
    
    def _save_status(self, component: str):
        """Save download status to file"""
        with open(self.status_file, 'a') as f:
            f.write(f"{component}\n")
        self.downloaded[component] = True
    
    def download_olmoe(self) -> Optional[str]:
        """Download OLMoE-1B model"""
        if "olmoe" in self.downloaded:
            logger.info("OLMoE already downloaded, skipping...")
            return str(self.model_dir / "olmoe")
        
        try:
            logger.info("Downloading OLMoE-1B...")
            model_path = self.model_dir / "olmoe"
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(
                "allenai/OLMoE-1B-7B-0924",
                torch_dtype=torch.float16,
                cache_dir=str(model_path)
            )
            
            self._save_status("olmoe")
            logger.info(f"OLMoE downloaded successfully to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error downloading OLMoE: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def download_dolma_tokenizer(self) -> Optional[str]:
        """Download Dolma tokenizer"""
        if "tokenizer" in self.downloaded:
            logger.info("Tokenizer already downloaded, skipping...")
            return str(self.tokenizer_dir)
        
        try:
            logger.info("Downloading Dolma tokenizer...")
            tokenizer_path = self.tokenizer_dir / "dolma"
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "allenai/gpt-neox-olmo-dolma-v1_5",
                cache_dir=str(tokenizer_path)
            )
            
            self._save_status("tokenizer")
            logger.info(f"Tokenizer downloaded successfully to {tokenizer_path}")
            return str(tokenizer_path)
            
        except Exception as e:
            logger.error(f"Error downloading tokenizer: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def download_dolma_data(self) -> Optional[str]:
        """Download Dolma dataset"""
        if "dolma" in self.downloaded:
            logger.info("Dolma dataset already downloaded, skipping...")
            return str(self.data_dir / "dolma")
        
        try:
            logger.info("Downloading Dolma dataset...")
            data_path = self.data_dir / "dolma"
            
            # Download dataset with trust_remote_code=True
            dataset = load_dataset(
                "allenai/dolma",
                cache_dir=str(data_path),
                streaming=True,  # Use streaming to handle large dataset
                trust_remote_code=True  # Added this parameter
            )
            
            self._save_status("dolma")
            logger.info(f"Dolma dataset downloaded successfully to {data_path}")
            return str(data_path)
            
        except Exception as e:
            logger.error(f"Error downloading Dolma dataset: {str(e)}")
            logger.error(traceback.format_exc())
            return None

def main():
    # Create downloader instance
    downloader = ModelDownloader(cache_dir="./olmoe_cache")
    
    # Track overall success
    success = {
        "olmoe": False,
        "tokenizer": False,
        "dolma": False
    }
    
    # Download components
    olmoe_path = downloader.download_olmoe()
    if olmoe_path:
        success["olmoe"] = True
        logger.info(f"OLMoE saved to: {olmoe_path}")
    
    tokenizer_path = downloader.download_dolma_tokenizer()
    if tokenizer_path:
        success["tokenizer"] = True
        logger.info(f"Tokenizer saved to: {tokenizer_path}")
    
    dolma_path = downloader.download_dolma_data()
    if dolma_path:
        success["dolma"] = True
        logger.info(f"Dolma dataset saved to: {dolma_path}")
    
    # Print summary
    logger.info("\nDownload Summary:")
    for component, status in success.items():
        logger.info(f"{component}: {'Success' if status else 'Failed'}")
    
    return success

if __name__ == "__main__":
    main()