from ml_collections import config_dict

import os
import torch

IMAGE_HEIGHT = config_dict.FieldReference(40)
CHUNK_SIZE = config_dict.FieldReference(320)

DEFAULT_RUN_NAME = "__NAME__"


def default_attn_ctc_model_config():
    hidden_features = config_dict.FieldReference(256)

    model = config_dict.ConfigDict()

    backbone = model.backbone = config_dict.ConfigDict()
    backbone.out_features = hidden_features

    encoder = model.encoder = config_dict.ConfigDict()
    encoder.type = "attn"
    encoder.features = hidden_features

    attn = encoder.attn = config_dict.ConfigDict()
    attn.num_heads = 4
    attn.num_layers = 4

    decoder = model.decoder = config_dict.ConfigDict()
    decoder.type = "ctc"

    ctc = decoder.ctc = config_dict.ConfigDict()
    ctc.in_features = hidden_features
    ctc.out_features = 81 # counted later
    ctc.image_height = IMAGE_HEIGHT
    ctc.chunk_size = CHUNK_SIZE

    return model



def default_diploma_config():

    image_height = IMAGE_HEIGHT

    CONFIG = config_dict.ConfigDict()

    model = CONFIG.model = config_dict.ConfigDict()
    model.type = "single" # single / duo
    model.first = default_attn_ctc_model_config()
    model.second = default_attn_ctc_model_config()

    data = CONFIG.data = config_dict.ConfigDict()

    data.input_height = image_height
    data.chunk_size = CHUNK_SIZE

    data.root_path = "/home/jupyter/mnt/datasets/diploma"

    iam = data.iam = config_dict.ConfigDict()
    iam.path = os.path.join(data.get_ref('root_path').get(), "./")
    iam.weight = 1
    iam.train_len = 6482

    iam.transforms = config_dict.ConfigDict()

    iam.transforms.vertical_mask_tile_prob = 0.15
    iam.transforms.vertical_mask_min_height_ratio = 0.25
    iam.transforms.vertical_mask_max_height_ratio = 0.5

    iam.transforms.gauss_sigma = (1.0,2.5)
    iam.transforms.gauss_kernel_size = (5,5)
    iam.transforms.noise_gauss = (0.02, 0.1)
    iam.transforms.stretch_factor = (0.8, 1.2)

    iam.transforms.image_height = image_height
    iam.transforms.inverse_prob = 0.5

    iam.transforms.distortion_scale = 0.1

    iam.transforms.number_of_transforms = 3


    mjsynth = data.mjsynth = config_dict.ConfigDict()
    mjsynth.path = os.path.join(data.get_ref('root_path').get(), "mjsynth/")
    mjsynth.weight = 3

    mjsynth.transforms = config_dict.ConfigDict()
    mjsynth.transforms.image_height = image_height
    mjsynth.transforms.long_lines = config_dict.ConfigDict()
    mjsynth.transforms.long_lines.prob = 0.2
    mjsynth.transforms.long_lines.min_space_to_h = 0.5
    mjsynth.transforms.long_lines.max_space_to_h = 1.0
    mjsynth.transforms.long_lines.space_value = 0.0


    training = CONFIG.training = config_dict.ConfigDict()

    CONFIG.eval = False

    training.batch_size = 8
    training.num_epochs = 150
    training.eval_epochs_interval = 5
    training.eval_test_interval = 15
    training.snapshot_epochs_interval = 5
    training.checkpoints_folder = "./checkpoints"
    training.loader_num_workers = 6
    training.load_from_checkpoint = False
    training.grad_clip_value = 1


    evaluate = CONFIG.evaluate = config_dict.ConfigDict()
    evaluate.batch_size = 8
    evaluate.loader_num_workers = 6
    evaluate.load_from_checkpoint = True


    optim = CONFIG.optimizer = config_dict.ConfigDict()
    optim.constructor = "torch.optim.AdamW"
    optim.weight_decay = 1e-4
    optim.params = config_dict.ConfigDict()
    optim.params.lr = 3e-4
    optim.params.betas = (0.9, 0.999)

    lr_scheduler = CONFIG.lr_scheduler = config_dict.ConfigDict()
    lr_scheduler.constructor = "StepLRWithWarmup"
    lr_scheduler.meta_params = config_dict.ConfigDict()
    lr_scheduler.meta_params.warmup_epochs = 5
    lr_scheduler.meta_params.step_epochs = 15

    lr_scheduler.params = config_dict.ConfigDict()
    lr_scheduler.params.initial_lr = optim.params.get_ref('lr')
    lr_scheduler.params.warmup_lr = lr_scheduler.params.get_ref('initial_lr')
    # always 5 epochs
    lr_scheduler.params.warmup_steps = iam.get_ref('train_len') * (iam.get_ref('weight') + mjsynth.get_ref('weight')) // iam.get_ref('weight') // training.get_ref('batch_size') * lr_scheduler.meta_params.get_ref('warmup_epochs')
    # always num_epochs // 5 epochs
    lr_scheduler.params.step = iam.get_ref('train_len') * (iam.get_ref('weight') + mjsynth.get_ref('weight')) // iam.get_ref('weight') // training.get_ref('batch_size') * lr_scheduler.meta_params.get_ref('step_epochs')
    lr_scheduler.params.decay = 0.7


    wandb = CONFIG.wandb = config_dict.ConfigDict()
    wandb.project_name = "diploma_dml"
    wandb.run_name = DEFAULT_RUN_NAME
    wandb.resume = False

    CONFIG.device = "cuda:0"

    return CONFIG

def get_config():
    return default_diploma_config()

