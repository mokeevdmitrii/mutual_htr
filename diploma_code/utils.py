import wandb
import torch

import typing as tp

import random
import os
import numpy as np

from torch.nn.utils.rnn import pad_sequence


def log_metric_wandb(metric_name: str, mode: str, value: tp.Any, step: int):
    wandb.log({f'{metric_name}/{mode}': value}, step=step)


def batch_to_device(batch: tp.Dict[str, tp.Any], device: torch.device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def dict_collate_fn(batch):
    result = {}
    must_pad = {}
    for key, value in batch[0].items():
        result[key] = []
        must_pad[key] = isinstance(value, torch.Tensor)

    for i, sample in enumerate(batch):
        for key, value in sample.items():
            result[key].append(value)

    lengths = {}
    for key, values in result.items():
        if must_pad[key]:
            result[key] = pad_sequence(values, batch_first=True)
            lengths[f'{key}_length'] = torch.tensor(
                [value.shape[0] for value in values])
    result.update(lengths)
    return result

