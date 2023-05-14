import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as TF

import numpy as np
import typing as tp


class VerticalRandomMasking:

    def __init__(self, min_w, max_w, tile_prob, mask_prob, rng=torch.Generator()):
        self.min_w = min_w
        self.max_w = max_w
        self.tile_prob = tile_prob
        self.rng = torch.Generator()
        self.mask_prob = mask_prob

    def __call__(self, x):
        C, H, W = x.shape
        if torch.rand(size=(1, ), generator=self.rng) > self.mask_prob:
            return x

        tw = torch.randint(low=self.min_w, high=self.max_w + 1, size=(1, ), generator=self.rng)
        begin = 0
        ignore = False
        while begin < W:
            end = min(W, begin + tw)
            if not ignore and torch.rand(size=(1,), generator=self.rng) < self.tile_prob:
                x[..., begin:end] = torch.rand((C, H, end - begin), generator=self.rng)
                ignore = True
            elif ignore:
                ignore = False
            begin = end
        return x


class HorizontalResizeOnly:

    def __init__(self, h, interpolation=TF.InterpolationMode.BILINEAR):
        self.h = h
        self.interpolation = interpolation

    def __call__(self, img):
        _, H, W = img.shape
        dtype = img.dtype

        w = int(W * self.h / H)
        out = TF.resize(img, (self.h, w), self.interpolation, max_size=None, antialias=True)

        if out.dtype != img.dtype:
            out = out.to(dtype)
        return out


class GaussianNoise:
    def __init__(self, sigma: tp.Tuple[float, float], generator=None):
        self.sigma = sigma
        self.generator = generator

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        dtype = image.dtype

        sigma = self.sigma[0] + torch.rand((1,), generator=self.generator).item() * (self.sigma[1] - self.sigma[0])

        out = image + torch.randn_like(image) * sigma
        if out.dtype != dtype:
            out = out.to(dtype)

        return out

class RandomHorizontalStretch:
    def __init__(self, factor: tp.Tuple[float, float], interpolation=TF.InterpolationMode.BILINEAR, generator=None):
        self.min_f = factor[0]
        self.max_f = factor[1]
        self.generator = generator
        self.interpolation = interpolation

    def __call__(self, img):
        _, H, W = img.shape
        factor = self.min_f + torch.rand((1,), generator=self.generator).item() * (self.max_f - self.min_f)
        desired_w = int(W * factor)

        out = TF.resize(img, (H, desired_w), self.interpolation, max_size=None, antialias=None)
        return out

class RandomChoiceN:
    def __init__(self, transforms, K: int, rng = np.random.default_rng()):
        assert K >= 0 and K <= len(transforms)
        self.transforms = transforms
        self.K = K
        self.N = len(transforms)
        self.rng = rng

    def __call__(self, img):
        idxs = self.rng.choice(self.N, (self.K, ), replace=False)
        idxs = np.sort(idxs)
        dtype = img.dtype

        out = img
        for idx in idxs:
            out = self.transforms[idx](out)
        if out.dtype != dtype:
            out = out.to(dtype)
        return out

