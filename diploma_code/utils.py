import wandb
import torch

import typing as tp

def log_metric_wandb(metric_name: str, mode: str, value: tp.Any, step: int):
    wandb.log({f'{metric_name}/{mode}': value}, step=step)


def batch_to_device(batch: tp.Dict[str, tp.Any], device: torch.device):
    batch['input'] = batch['input'].to(device)
    batch['idx'] = batch['idx'].to(device)
    batch['w'] = batch['w'].to(device)
    batch['gt_text'] = batch['gt_text'].to(device)
    batch['tgt_len'] = batch['tgt_len'].to(device)
    return batch


