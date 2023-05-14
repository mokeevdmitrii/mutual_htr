# DEPRECATED, TO BE REWRITTEN / DELETED

import torch

import typing as tp

from .data_loader.data_common import CharEncoder
from .data_loader.datasets import (
    BaseLTRDataset, LongLinesLTRDataset, MyConcatDataset,
    MergingSampler, TransformLTRWrapperDataset
)
from .data_loader.iam import load_iam_data_dict, make_iam_split
from .data_loader.mjsynth import load_mjsynth_chars, load_mjsynth_samples
from .data_loader.loader import stacking_collate_fn

from .make_transforms import (
    make_iam_train_augment, make_mjsynth_train_augment, make_iam_test_augment, make_final_augment
)


def make_char_encoder(data_config):
    char_encoder = CharEncoder()

    iam_conf = data_config.iam
    mjsynth_conf = data_config.mjsynth

    iam_data_dict = load_iam_data_dict(iam_conf.path)
    char_encoder.update_with_chars(iam_data_dict['chars'])
    mjsynth_chars = load_mjsynth_chars(mjsynth_conf.path)
    char_encoder.update_with_chars(mjsynth_chars)

    return char_encoder


def make_dataloader(data_config, mode: str, batch_size: int, num_workers: int):

    if mode != 'train' and mode != 'valid' and mode != 'test':
        raise ValueError(f"invalid dataloader mode: {mode}, expected train|valid|test")

    iam_conf = data_config.iam
    mjsynth_conf = data_config.mjsynth

    iam_data_dict = load_iam_data_dict(iam_conf.path)
    mjsynth_chars = load_mjsynth_chars(mjsynth_conf.path)

    char_encoder = CharEncoder()

    char_encoder.update_with_chars(iam_data_dict['chars'])
    char_encoder.update_with_chars(mjsynth_chars)

    iam_split = make_iam_split(iam_data_dict["samples"], iam_conf.path, mode)

    if mode == 'train':
        mjsynth_transform = make_mjsynth_train_augment(mjsynth_conf.transforms)

        mjsynth_split = load_mjsynth_samples(mjsynth_conf.path, mode)
        mjsynth_dataset = LongLinesLTRDataset(mjsynth_conf.transforms.long_lines,
                                              mjsynth_split, mjsynth_conf.path,
                                              char_encoder, mjsynth_transform)

        iam_transform = make_iam_train_augment(iam_conf.transforms)
    else:
        iam_transform = make_iam_test_augment(iam_conf.transforms)

    iam_dataset = BaseLTRDataset(iam_split, iam_conf.path, char_encoder, iam_transform)

    if mode == 'train':
        dss = [iam_dataset, mjsynth_dataset]
        wss = [iam_conf.weight, mjsynth_conf.weight]
        dataset = MyConcatDataset(dss)
    else:
        dataset = iam_dataset

    dataset = TransformLTRWrapperDataset(dataset, make_final_augment(data_config))

    # create dataloader with cool sampler
    if mode == 'train':
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=MergingSampler(dss, wss),
                                             num_workers=num_workers, collate_fn=stacking_collate_fn)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, collate_fn=stacking_collate_fn)

    return loader
