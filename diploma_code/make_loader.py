import torch
import torch.nn

import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

import diploma_code

from diploma_code.dataset import (
    BaseLTRDataset, LongLinesLTRDataset
)

from diploma_code.char_encoder import CharEncoder

from diploma_code.make_transforms import make_transforms

from diploma_code.utils import (
    dict_collate_fn
)

from ml_collections import ConfigDict


def make_char_encoder(data_config: ConfigDict):
    dataset_name = data_config.dataset
    dataset_config = data_config[dataset_name]
    cfg_constructor = dataset_config.config_constructor

    data_cool_cfg = eval(cfg_constructor)(dataset_config)
    return CharEncoder(data_cool_cfg)


def make_datasets(config: ConfigDict):
    dataset_config = config[config.dataset]
    
    df_path = os.path.join(config.root_path, config.dataset, 'marking.csv')
    
    df = pd.read_csv(df_path, index_col='sample_id')
    
    train_df = df[df['stage'] == 'train']
    valid_df = df[df['stage'] == 'valid']
    test_df = df[df['stage'] == 'test']
        
        
    char_enc = make_char_encoder(config)
    transforms = make_transforms(config.transforms)
    
    train_dataset_kwargs = {'transforms': transforms, **dataset_config.get('train_dataset_extra_args', {})}
    
    train_dataset_constructor = dataset_config.train_dataset_constructor
    
    train_dataset = eval(train_dataset_constructor)(train_df, dataset_config, 
                                                    char_enc, **train_dataset_kwargs)
    valid_dataset = BaseLTRDataset(valid_df, dataset_config, char_enc)
    test_dataset = BaseLTRDataset(test_df, dataset_config, char_enc)
    
    return {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    }

    
def make_dataloaders(config: ConfigDict):
    datasets = make_datasets(config.data)
    
    train_loader = DataLoader(
        datasets['train'],
        batch_size=config.training.batch_size,
        sampler=RandomSampler(datasets['train']),
        pin_memory=False,
        num_workers=config.training.loader_num_workers,
        collate_fn=dict_collate_fn
    )
    
    valid_loader = DataLoader(
        datasets['valid'],
        batch_size=config.evaluate.batch_size,
        sampler=SequentialSampler(datasets['valid']),
        pin_memory=False,
        num_workers=config.evaluate.loader_num_workers,
        collate_fn=dict_collate_fn
    )
    
    test_loader = DataLoader(
        datasets['test'],
        batch_size=config.evaluate.batch_size,
        sampler=SequentialSampler(datasets['test']),
        pin_memory=False,
        drop_last=False,
        num_workers=config.evaluate.loader_num_workers,
        collate_fn=dict_collate_fn
    )
    return {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    
    