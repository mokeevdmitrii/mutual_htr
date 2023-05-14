import torch
import torch.nn.functional as F

import itertools

import typing as tp

from editdistance import eval as edit_distance

from diploma_code.char_encoder import (
    CharEncoder
)
from diploma_code.utils import (
    log_metric_wandb
)
import re

def my_ctc_loss(log_probs, targets, input_lengths, target_lengths):
    """
    log_probs: (L, B, C)
    targets: (B, L)
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='mean', zero_infinity=True)


def kl_div(log_inputs, log_targets, target_lengths, *args, **kwargs):
    """
    log_inputs: (L, B, C)
    log_targets: (L, B, C)
    input_lengths: unused
    """
    pointwise = F.kl_div(log_inputs, log_targets, reduction="none", log_target=True)
    loss = 0
    for b in range(log_inputs.size(1)):
        # add mean over length for each batch
        loss = loss + torch.sum(torch.mean(pointwise[:target_lengths[b], b, :], dim=0))
    # average over batch
    return loss / log_targets.size(1)

def my_dml_loss(log_probs: tp.Tuple[torch.Tensor, torch.Tensor],
             targets, input_lengths, target_lengths):

    lp_1, lp_2 = log_probs
    il_1, il_2 = input_lengths
    ctc_1 = my_ctc_loss(lp_1, targets, il_1, target_lengths)
    ctc_2 = my_ctc_loss(lp_2, targets, il_2, target_lengths)
    kl = (kl_div(lp_1, lp_2.detach(), il_2) + kl_div(lp_2, lp_1.detach(), il_1)) / 2
    return {
        'loss_1': ctc_1,
        'loss_2': ctc_2,
        'kl': kl
    }

@torch.no_grad()
def decode_ocr_probs(log_probs: torch.Tensor, char_encoder: CharEncoder):
    """
    log_probs: (L, B, C)
    """
    # shape: (L, B)
    best = torch.argmax(log_probs, dim=-1)
    res = []
    for b in range(log_probs.shape[1]):
        curr_best = [k.item() for k, g in itertools.groupby(best[:, b])]
        out_str = char_encoder.decode(curr_best)
        res.append(re.sub('\s+', ' ', out_str))
    return res


def get_edit_distance(preds: str, targets: str, uncased: bool = False):
    """
    inputs: str
    targets: str
    """
    preds = preds.strip(" ")
    targets = targets.strip(" ")

    if uncased:
        errors = edit_distance(preds.lower(), targets.lower())
    else:
        errors = edit_distance(preds, targets)

    return {
        'err_count': errors,
        'gt_len': len(targets)
    }


class EpochValueProcessor:
    def __init__(self,
                 name: str,
                 mode: str,
                 report_per_batch: bool = False,
                 report_final: bool = True,
                 log_metric=log_metric_wandb):
        self.value = 0.
        self.items = 0
        self.mode = mode

        self.name = name
        self.report_per_batch = report_per_batch
        self.report_final = report_final

        self.log_metric=log_metric

    def __call__(self, value: torch.Tensor, batch_size: int, step: int):
        if self.report_per_batch:
            self.log_metric(self.name, self.mode, value.item(), step)

        self.value += value.item() * batch_size
        self.items += batch_size

        return value

    def finalize(self, step):
        value = self.value / self.items

        if self.report_final:
            self.log_metric(f'epoch_{self.name}', self.mode, value, step)

        self.value = 0.
        self.items = 0
        return value


class EpochDMLProcessor:
    def __init__(self,
                 name: str,
                 mode: str,
                 report_per_batch: bool = False,
                 report_final: bool = True,
                 log_metric=log_metric_wandb):

        self.loss_1 = EpochValueProcessor(f'{name}_1', mode, report_per_batch, report_final, log_metric)
        self.loss_2 = EpochValueProcessor(f'{name}_2', mode, report_per_batch, report_final, log_metric)
        self.kl = EpochValueProcessor(f'{name}_kl', mode, report_per_batch, report_final, log_metric)

    def __call__(self, loss, batch_size, step):

        l1,l2,kl = loss['loss_1'], loss['loss_2'], loss['kl']

        l1 = self.loss_1(l1, batch_size, step)
        l2 = self.loss_2(l2, batch_size, step)
        kl = self.kl(kl, batch_size, step)

        return l1 + l2 + kl

    def finalize(self, step):

        return l1.finalize(step) + l2.finalize(step) + kl.finalize(step)



class CERProcessor:

    def __init__(self, char_encoder: CharEncoder, name, mode, report_per_batch=False,
                 report_final=True, log_metric=log_metric_wandb):

        self.value = EpochValueProcessor(name, mode, report_per_batch, report_final, log_metric)
        self.char_encoder = char_encoder

    def __call__(self, log_probs, targets, step):
        str_preds = decode_ocr_probs(log_probs, self.char_encoder)
        str_targets = targets

        if len(str_preds) != len(str_targets):
            raise ValueError(f"Strange: len pred != len target: {len(str_preds)} != {len(str_targets)}. Targets shape: {targets.shape}, log probs shape: {log_probs.shape}")

        e, t = 0, 0
        for pred, target in zip(str_preds, str_targets):
            err_dict = get_edit_distance(pred, target)
            err, total = err_dict['err_count'], err_dict['gt_len']
            e += err
            t += total

        return self.value(100 * torch.tensor([e]) / t, t, step)

    def finalize(self, step):
        return self.value.finalize(step)

