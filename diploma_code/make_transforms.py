import torchvision as tv

from ml_collections import ConfigDict

from .data_loader.transforms import (
    HorizontalResizeOnly, VerticalRandomMasking, HorizontalChunker, ChunkTransform, GaussianNoise, RandomHorizontalStretch, RandomChoiceN
)

def make_iam_train_augment(config: ConfigDict):

    transforms = tv.transforms.Compose([
        tv.transforms.RandomInvert(),
        HorizontalResizeOnly(config.image_height),
        RandomChoiceN([
            GaussianNoise(config.noise_gauss),
            tv.transforms.GaussianBlur(kernel_size=config.gauss_kernel_size, sigma=config.gauss_sigma),
            tv.transforms.RandomPerspective(distortion_scale=config.distortion_scale, p=1.0),
            RandomHorizontalStretch(factor=config.stretch_factor),
            VerticalRandomMasking(
                int(config.image_height * config.vertical_mask_min_height_ratio),
                int(config.image_height * config.vertical_mask_max_height_ratio),
                config.vertical_mask_tile_prob,
                1.0
            ),
        ], config.number_of_transforms)
    ])


    return transforms


def make_iam_test_augment(config: ConfigDict):

    transforms = tv.transforms.Compose([
        HorizontalResizeOnly(config.image_height),
    ])

    return transforms


def make_mjsynth_train_augment(config: ConfigDict):
    transforms = tv.transforms.Compose([
        HorizontalResizeOnly(config.image_height),
    ])

    return transforms


def make_mjsynth_test_augment(config: ConfigDict):
    transforms = tv.transforms.Compose([
        HorizontalResizeOnly(config.image_height),
    ])

    return transforms


def make_chunker(config: ConfigDict):
    return HorizontalChunker(config.input_height, config.chunk_size)


def make_final_augment(data_config):
    return ChunkTransform(make_chunker(data_config))
