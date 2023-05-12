import os

import typing as tp

from .data_common import Sample


def load_mjsynth_chars(data_path: str) -> tp.Set[str]:
    """
    Loads all chars used in MJSynth dataset
    """
    chars = set()
    annotation_file = os.path.join(data_path, 'annotation.txt')

    with open(annotation_file, 'r') as f:
        for idx, line in enumerate(f):
            base_name = os.path.basename(line.strip().split(" ",1)[0])
            label = "_".join(base_name.split("_")[1:-1])
            chars |= set(label)
    return chars

def mjsynth_get_annotation(data_path: str, mode: str):
    return os.path.join(data_path, f"annotation_{mode}.txt")


def mjsynth_get_blacklist(data_path: str):
    blacklist = set()
    with open(os.path.join(data_path, 'blacklist.txt')) as f:
        for line in f:
            blacklist.add(line.strip())
    return blacklist


def count_lines_in_file(name: str) -> int:
    with open(name, 'r') as f:
        return sum(1 for _ in f)

def load_mjsynth_samples(data_path: str, mode: str, domain_size=None):
    samples = []

    MODE_MATCH = {"train": "train", "valid": "val", "test": "test"}
    if mode not in MODE_MATCH:
        raise ValueError(f"Invalid mode {mode}, allowed: {list(MODE_MATCH.keys())}")

    annotation_path = mjsynth_get_annotation(data_path, MODE_MATCH[mode])

    num_files_max = count_lines_in_file(annotation_path)
    if domain_size is not None:
        num_files_max = min(num_files_max, domain_size)

    b = mjsynth_get_blacklist(data_path)
    bad_c = 0
    with open(annotation_path, 'r') as f:
        for num, line in enumerate(f):
            if num - bad_c == num_files_max:
                break
            rel_path, num = line.split(' ', 1)
            if rel_path in b:
                bad_c += 1
                continue
            gt_text = "_".join(os.path.basename(rel_path).split("_")[1:-1])
            samples.append(Sample(gt_text, rel_path))

    return samples

