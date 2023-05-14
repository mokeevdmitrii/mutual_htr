import torch

import pandas as pd
import numpy as np
import os
import cv2
import re

from torch.utils.data import Dataset

from ml_collections import ConfigDict


from diploma_code.char_encoder import CharEncoder
from diploma_code.transforms_functional import resize_if_greater, make_img_padding

class BaseLTRDataset(Dataset):

    def __init__(self, df: pd.DataFrame, config: ConfigDict, 
                 char_enc: CharEncoder, transforms=None):
        
        self.config = config
        self.char_enc = char_enc
        self.image_ids = df.index.values
        self.texts = df['text'].values
        self.paths = df['path'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = self.load_image(idx)
        gt_text = self.texts[idx]

        encoded = self.char_enc.encode(gt_text)
        
        img = self._transform_image(img)
        
        return {
            'id': img_id,
            'image': img,
            'gt_text': gt_text,
            'encoded': torch.tensor(encoded, dtype=torch.int32),
        }

    def load_image(self, idx):
        img = self._load_raw_image(idx) 
        return self._resize_image(img)
    
    def _load_raw_image(self, idx):
        img = cv2.imread(os.path.join(self.config['path'], self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _resize_image(self, img, shape=None):
        if shape is None:
            shape = (self.config['image_height'], self.config['image_width'])
        img, _ = resize_if_greater(img, shape[0], shape[1])
        img = make_img_padding(img, shape[0], shape[1])
        return img
    
    def _transform_image(self, img):
        if self.transforms:
            img = self.transforms(image=img)['image']

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return img


class LongLinesLTRDataset(BaseLTRDataset):
    
    def __init__(self, df: pd.DataFrame, config: ConfigDict, 
                 char_enc: CharEncoder, transforms=None, rng = np.random.default_rng()):
        
        super(LongLinesDataset, self).__init__(df, config, char_encoder, transforms)
        # proba
        # min_h_ratio
        # max_h_ratio
        self.long_lines = config['long_lines']
        self.rng = rng
        
    def __getitem__(self, idx):
        
        if rng.random() < self.long_lines.proba:
            idx_2 = rng.integers(low=0, high=len(self))

            item_1 = self._get_single_item(idx)
            item_2 = self._get_single_item(idx_2)
            
            
            # assume image is always CHW, no grayscale with HW
            C, _, _ = item_1['image'].shape
            
            space_W = rng.uniform(int(self.long_lines['min_h_ratio'] * self.config['image_height']), 
                        int(self.long_lines['max_h_ratio'] * self.config['image_height']))
            
            img1 = self._resize_image(item_1['image'], (self.config.image_height, item_1['image'].shape[-1]))
            img2 = self._resize_image(item_2['image'], (self.config.image_height, item_2['image'].shape[-1]))
            space = np.zeros((C, self.config['image_height'], space_W), dtype=np.uint8)
            
            img = np.concatenate([img1, space, img2], axis=-1)
            gt_text = item_1['gt_text'].rstrip() + ' ' + item_2['gt_text'].lstrip()
            encoded = self.char_enc.encode(gt_text)
            
            img_id = f"{item_1['id']}__CAT__{item_2['id']}"
            
            return {
                'id': img_id,
                'image': self._transform_image(img),
                'gt_text': gt_text,
                'encoded': encoded
            }
        
        else:
 
            return super().__getitem__(idx)
            
    def _get_single_item(self, idx):
        img_id = self.image_ids[idx]
        img = self.load_image(img_id)
        gt_text = self.texts[img_id]
        return {
            'id': img_id,
            'image': img,
            'gt_text': gt_text,
        }

