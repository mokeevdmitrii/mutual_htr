import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as TF

import numpy as np
import typing as tp

class HorizontalChunker:
    def __init__(self, h, w, ctx_w=None):

        assert h < w, 'chunk height must be smaller than width'
        self.h = h
        self.w = w
        if ctx_w is None:
            self.ctx_w = self.h
        else:
            self.ctx_w = ctx_w

    def split(self, img):
        """
        overlapping chunks of size (self.h,
                                    self.h of padding / prev chunk +
                                    self.w + at least self.h of pad / next chunk)
        """

        # assert img.min() >= -0.001 and img.max() <= 1.001, f'invalid img range [{img.min()}, {img.max()}], apply after [0,1] normalization'

        C, H, W = img.shape

        if H != self.h:
            raise ValueError(f"Invalid image height: H ({H}) != self.h ({self.h})")

        extra_pad = self.w - W % self.w
        chunk_size = self.w + 2 * self.ctx_w
        padded = F.pad(img, (self.ctx_w, self.ctx_w + extra_pad), mode="constant", value=0)
        chunks = padded.unfold(dimension=-1, size=chunk_size, step=self.w).clone()
        chunks = chunks.permute(2, 0, 1, 3) # single image to BCHW
        N = len(chunks)
        dtype = chunks.dtype
        for i, chunk in enumerate(chunks):
            assert len(chunk.shape) == 3
            if i != 0:
                chunks[i, :, :, :self.ctx_w] *= 0.5
            if i != N - 1:
                chunks[i, :, :, -self.ctx_w:] *= 0.5
        if chunks.dtype != dtype:
            chunks = chunks.to(dtype)
        return chunks

    def _make_ranges(self, idxs):
        """
        maps arbitrary sequence of kind
        [a1, ..., a1, a2, ..., a2, ...] to
        [(0, a1_range_end), (a2_range_start, a2_range_end), ...]
        group by indexes
        """
        if len(idxs) == 0:
            return []
        begin = 0
        res = []
        for i, x in enumerate(idxs[1:]):
            # [begin, end) is an equal range
            if x != idxs[begin]:
                res.append((begin, i + 1))
                begin = i + 1
        res.append((begin, len(idxs)))
        return res


    def _assert_divisible(self, sz):
        "used in merge_single_chunk"
        rem = (self.w + 2 * self.ctx_w) % sz
        assert rem == 0, f'not natural shrinkage factor: {(self.w + 2 * self.ctx_w)} / {sz}'
        factor = (self.w + 2 * self.h) // sz
        assert self.w % factor == 0, f'bad choice of w: {self.w} != {factor} * k for some k'
        assert self.ctx_w % factor == 0, f'bad choice of ctx_w: {self.ctx_w} != {factor} * k for some k'


    def merge_single_chunk(self, chunks, w):
        """
        select and join valid portions of chunk
        chunks are (B, L, C)
        width is the desired size of total chunk_length

        returns: merged chunk of shape
        (total_L, C)
        """

        ch_L = chunks.size(1)
        # always round down everywhere
        ch_start = (ch_L * self.ctx_w) // (self.w + 2 * self.ctx_w)
        ranges = []
        left_w = w
        for i, ch in enumerate(chunks):
            if left_w == 0 and i != len(chunks) - 1:
                raise ValueError(f'invalid chunk split: ranges={ranges}, ch_start={(ch_L * self.ctx_w) // (self.w + 2 * self.ctx_w)}, ch_L={w}, b={chunks.size(0)}')
            ch_len = min(ch_L - 2 * ch_start, left_w)
            ranges.append((ch_start, ch_start + ch_len))
            left_w = left_w - ch_len
        res = torch.cat([chunk[ranges[i][0]:ranges[i][1], :] for i, chunk in enumerate(chunks)], dim=0)
        return res


    def merge(self, chunks, idxs, ws):
        """
        chunks are (B_1, L, C) joining them by length
        returns:
           chunks: (max(lens), B, C) merged tensor
           chunk_lens: list with lens of chunks of len B_1
        """
        ranges = self._make_ranges(idxs)
        result_chunks = []
        chunk_lens = []
        for i, r in enumerate(ranges):
            r_chunks = chunks[r[0]:r[1],...]
            result_chunks.append(self.merge_single_chunk(r_chunks, ws[i].item()))
            chunk_lens.append(result_chunks[-1].size(0))
        return torch.nn.utils.rnn.pad_sequence(result_chunks, batch_first=False), torch.LongTensor(chunk_lens)


class ChunkTransform:
    def __init__(self, chunker: HorizontalChunker):
        self.chunker = chunker

    def __call__(self, x):
        return self.chunker.split(x)

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

