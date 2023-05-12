import torch

import typing as tp

class Sample:
    def __init__(self, gt_text, file_path):
        self.gt_text = gt_text
        self.file_path = file_path

    def __repr__(self):
        return f"gt text: {self.gt_text}, filePath: {self.file_path}"


class CharEncoder:

    def __init__(self, chars: tp.Set[str] = set(), blank=0):
        self._encode = {}
        self._decode = {}
        self._max = -1
        self._blank = blank

        self._decode[blank] = ""
        self.update_with_chars(chars)
        # always add space, used for long lines
        self.update_with_chars(set([" "]))

    @property
    def blank(self):
        return self._blank

    def update_with_chars(self, chars: tp.Set[str]) -> None:
        sorted_chars = sorted(list(chars))
        for k in sorted_chars:
            if k not in self._encode:
                if self._max + 1 == self._blank:
                    self._max += 1
                self._encode[k] = self._max + 1
                self._decode[self._max + 1] = k
                self._max += 1

    def __len__(self):
        return 0 if self._max < 0 else self._max

    def encode(self, text: str) -> torch.LongTensor:
        return torch.LongTensor(list(map(lambda x: self._encode[x], text)))

    def decode(self, tnsr: tp.Sequence[int]) -> str:
        return "".join(list(map(lambda x: self._decode[x], tnsr)))

