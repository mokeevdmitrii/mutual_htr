import os
import torch
import torchvision
import cv2
import math

import numpy as np
import typing as tp

from ml_collections import ConfigDict

from .data_common import Sample, CharEncoder


class BaseLTRDataset(torch.utils.data.Dataset):
    def __init__(self, samples: tp.Sequence[Sample], root_path: str,
                 encoder: CharEncoder, transform: tp.Callable = None):

        self.samples = samples
        self.encoder = encoder
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, gt_text = os.path.join(self.root_path, self.samples[idx].file_path), self.samples[idx].gt_text
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"file path {file_path} has no image")
        img = img[..., None]
        img = torchvision.transforms.functional.to_tensor(img)
        if self.transform is not None:
            img = self.transform(img)
        w = img.shape[-1]
        txt = self.encoder.encode(gt_text)
        return {
            'input': img,
            'idx': idx,
            'w': w,
            'gt_text': txt,
            'tgt_len': torch.LongTensor([len(txt)])
        }



class LongLinesLTRDataset(BaseLTRDataset):
    def __init__(self, long_lines: ConfigDict, samples: tp.Sequence[Sample], root_path: str,
                 encoder: CharEncoder, transform: tp.Callable = None,
                 gen: np.random.Generator = np.random.default_rng()):

        super().__init__(samples, root_path, encoder, transform)

        self._ll_proba = long_lines.prob
        self._ll_min_space_to_h = long_lines.min_space_to_h
        self._ll_max_space_to_h = long_lines.max_space_to_h
        self._ll_space_value = long_lines.space_value
        self._gen = gen

    def __getitem__(self, idx):

        d = super().__getitem__(idx)
        use_ll = self._gen.random() < self._ll_proba

        if use_ll:
            rand_idx = self._gen.integers(low=0,high=len(self))
            another_d = super().__getitem__(rand_idx)

            img1, img2 = d['input'], another_d['input']
            txt1, txt2 = d['gt_text'], another_d['gt_text']

            _, h1, w1 = img1.shape
            _, h2, w2 = img2.shape

            assert h1 == h2, f"LTR shapes: {h1} != {h2}, heights do not match"

            space = torch.ones((1, h1, int(h1 * self._gen.uniform(
                self._ll_min_space_to_h, self._ll_max_space_to_h))),
                            dtype=d['input'].dtype) * self._ll_space_value

            result_img = torch.cat((img1, space, img2), dim=-1)
            result_txt = torch.cat((txt1, self.encoder.encode(" "), txt2), dim=0)
            result_tgt_len = result_txt.shape[0]
            result_w = result_img.shape[-1]

            return {
                'input': result_img,
                'idx': idx,
                'w': result_w,
                'gt_text': result_txt,
                'tgt_len': result_tgt_len
            }
        else:
            return d


class MyConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: tp.Sequence[torch.utils.data.Dataset]):
        self.dataset = torch.utils.data.ConcatDataset(datasets)

    def __len__(self, x):
        return len(self.dataset)

    def __getitem__(self, idx):
        d = self.dataset[idx]
        if isinstance(d, dict):
            if 'idx' in d:
                d['idx'] = idx
        return d


class RandomSampler:
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: tp.Sized
    replacement: bool

    def __init__(self, data_source: tp.Sized, replacement: bool = False,
                 num_samples: tp.Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> tp.Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples


class MergingSampler:
    """
    Sampler that merges several dataset given weights (sampling rates).
    Number of samples from each dataset is limited by the 'smallest' dataset
    in terms of len / weight ratio.

    Supposed to be used with MyConcatDataset.
    Dataset length must not change in runtime.
    """
    def __init__(self, datasets: tp.Sequence[torch.utils.data.Dataset], weights: tp.Sequence[tp.Union[int, float]], gen=None):

        assert len(datasets) == len(weights), f"datasets must have same len as weights, {len(datasets)} != {len(weights)}"

        lens = [len(d) for d in datasets]
        weighted_lens = [l / w if w != 0 else np.inf for l, w in zip(lens, weights)]
        min_idx = min(enumerate(weighted_lens), key=lambda x: x[1])[0]

        min_len, min_w = lens[min_idx], weights[min_idx]
        if min_w == 0:
            assert False, f"invalid weights {weights}"

        result_lens = [min(l, int(math.ceil(min_len / min_w * w))) for l, w in zip(lens, weights)]

        self.lens_cumsum = np.cumsum(result_lens)
        self.dataset_lens_begin = np.cumsum(lens) - np.array(lens)

        self.generator = gen

        self.samplers = [RandomSampler(d, replacement=False, num_samples=l, generator=self.generator)
                         for d, l in zip(datasets, result_lens)]

    def __len__(self):
        return self.lens_cumsum[-1]

    def __iter__(self):
        iters = [iter(s) for s in self.samplers]
        for idx in torch.randperm(len(self), generator=self.generator):
            for iters_idx, l in enumerate(self.lens_cumsum):
                if l > idx:
                    x = next(iters[iters_idx])
                    yield self.dataset_lens_begin[iters_idx] + x
                    break


class TransformLTRWrapperDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: torch.utils.data.Dataset, transform: tp.Callable):

        self.transform = transform
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Any]:
        d = self.dataset[idx]
        d['input'] = self.transform(d['input'])
        return d


