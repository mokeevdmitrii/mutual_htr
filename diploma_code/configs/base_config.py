from ml_collections import ConfigDict

class BaseDatasetConfig:
    
    def __init__(self, config: ConfigDict):
        self.data = config.to_dict()
        
    def __getitem__(self, item):
        return self.data[item]    
   
    def preprocess(self, text):
        return text
        
    def postprocess(self, text):
        return text
        