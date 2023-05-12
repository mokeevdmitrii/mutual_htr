import os
import typing as tp

from .data_common import Sample


def load_iam_data_dict(path) -> tp.Dict[str, tp.Any]:
    f = open(os.path.join(path, 'lines.txt'))
    chars = set()
    samples = []
    for line in f:
        if not line or line[0]=='#':
            continue
        # line text goes right after 8 spaces
        line_split = line.strip().split(' ', 8)
        assert len(line_split) == 9
        filename_split = line_split[0].split('-')
        file_name = 'lines/' + filename_split[0] + '/' +\
                   filename_split[0] + '-' + filename_split[1] + '/' + line_split[0] + '.png'

        gt_text = line_split[8].strip(" ")

        chars = chars.union(set(list(gt_text)))
        samples.append(Sample(gt_text, file_name))

    return {
        "samples": samples,
        "chars": chars
    }


def make_iam_split(samples: tp.List[Sample], path: str, split_type="train"):
    if split_type not in ["train", "valid", "test"]:
        raise ValueError("invalid split type, use 'train', 'valid' or 'test', please.")

    TYPE_MAP = {'train': 'train', 'valid': 'validation', 'test': 'test'}
    split = TYPE_MAP[split_type]

    folders = [x.strip("\n") for x in open(os.path.join(path, f"splits/{split}.uttlist")).readlines()]
    split_samples = []

    for i, s in enumerate(samples):
        file = s.file_path.split("/")[-1].split('.')[0].strip(" ")
        folder = "-".join(file.split("-")[:-1])
        if folder in folders:
            split_samples.append(s)

    return split_samples

