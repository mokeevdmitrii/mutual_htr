from ml_collections import ConfigDict

from .base_config import BaseDatasetConfig

import re

class IamConfig(BaseDatasetConfig):
    
    def __init__(self, config):
        super().__init__(config)
        
    def preprocess(self, text):
        """for train text"""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def postprocess(self, text):
        """for model output text"""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
        