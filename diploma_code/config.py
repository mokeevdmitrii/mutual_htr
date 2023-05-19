from ml_collections import config_dict

IMAGE_HEIGHT = config_dict.FieldReference(64)
IMAGE_WIDTH = config_dict.FieldReference(1024)
HIDDEN_FEATURES = config_dict.FieldReference(256)
TIME_FEATURE_COUNT = config_dict.FieldReference(256)
NUM_CLASSES = config_dict.FieldReference(81)

DEFAULT_RUN_NAME = "__NAME__"

def global_vars_config():
    config = config_dict.ConfigDict()
    config.image_height = IMAGE_HEIGHT
    config.image_width = IMAGE_WIDTH
    config.hidden_features = HIDDEN_FEATURES
    config.time_feature_count = TIME_FEATURE_COUNT
    config.num_classes = NUM_CLASSES
    return config


def default_attn_ctc_model_config(constants):
    
    hidden_features = HIDDEN_FEATURES

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


def default_model_v2_config(global_vars: config_dict.ConfigDict):
    
    model = config_dict.ConfigDict()
        
    backbone = model.backbone = config_dict.ConfigDict()
    
    backbone.constructor = "Resnet34Backbone"
    
    backbone.Resnet34Backbone = config_dict.ConfigDict()
    backbone.Resnet34Backbone.num_layers = 3
    backbone.Resnet34Backbone.pretrained = True
    backbone.Resnet34Backbone.max_pool_stride_1 = True
    
    encoder = model.encoder = config_dict.ConfigDict()
    encoder.constructor = "BiLSTMEncoder"
    
    encoder.BiLSTMEncoder = config_dict.ConfigDict()
    encoder.BiLSTMEncoder.input_size = global_vars.get_ref('hidden_features')
    encoder.BiLSTMEncoder.hidden_size = global_vars.get_ref('hidden_features')
    encoder.BiLSTMEncoder.num_layers = 3
    encoder.BiLSTMEncoder.dropout = 0.1
    
    encoder.TransformerEncoder = config_dict.ConfigDict()
    encoder.TransformerEncoder.in_features = global_vars.get_ref('hidden_features')
    encoder.TransformerEncoder.num_layers = 4
    encoder.TransformerEncoder.num_heads = 4
    encoder.TransformerEncoder.pe_max_len = 1500
    
    decoder = model.decoder = config_dict.ConfigDict()
    decoder.constructor = "CTCDecoderModel"
    
    decoder.CTCDecoderModel = config_dict.ConfigDict()
    decoder.CTCDecoderModel.num_classes = global_vars.get_ref('num_classes')
    decoder.CTCDecoderModel.time_feature_count = global_vars.get_ref('time_feature_count')

    return model



def default_diploma_config():

    CONFIG = config_dict.ConfigDict()
    
    global_vars = CONFIG.global_vars = global_vars_config()

    model = CONFIG.model = config_dict.ConfigDict()
    model.type = "single" # single / duo
    
    model.first = default_model_v2_config(global_vars)
    model.second = default_model_v2_config(global_vars)

    data = CONFIG.data = config_dict.ConfigDict()

    data.image_height = IMAGE_HEIGHT
    data.image_width = IMAGE_WIDTH
    data.root_path = "/home/jupyter/mnt/datasets/diploma"
    
    data.dataset = 'iam'

    iam = data.iam = config_dict.ConfigDict()
    
    iam.train_dataset_constructor = "BaseLTRDataset"
    iam.config_constructor = "diploma_code.configs.IamConfig"
    iam.chars = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    iam.blank = 'β'
    iam.length = 9652
    iam.image_height = IMAGE_HEIGHT
    iam.image_width = IMAGE_WIDTH
    
    transforms = data.transforms = config_dict.ConfigDict()
    
    basic_albums = transforms.basic_albums = config_dict.ConfigDict()
    
    basic_albums.CLAHE = config_dict.ConfigDict()
    basic_albums.CLAHE.enabled = True
    basic_albums.CLAHE.params = config_dict.ConfigDict()
    basic_albums.CLAHE.params.clip_limit = 4.0
    basic_albums.CLAHE.params.tile_grid_size = (8, 8)
    basic_albums.CLAHE.params.p = 0.25
    basic_albums.CLAHE.params.always_apply = False
    
    basic_albums.Rotate = config_dict.ConfigDict()
    basic_albums.Rotate.enabled = True
    basic_albums.Rotate.params = config_dict.ConfigDict()
    basic_albums.Rotate.params.limit = 2
    basic_albums.Rotate.params.interpolation = 1
    basic_albums.Rotate.params.border_mode = 0
    basic_albums.Rotate.params.p = 0.5
    
    basic_albums.ImageCompression = config_dict.ConfigDict()
    basic_albums.ImageCompression.enabled = True
    basic_albums.ImageCompression.params = config_dict.ConfigDict()
    basic_albums.ImageCompression.params.quality_lower = 75
    basic_albums.ImageCompression.params.p = 0.5
    
    blot = transforms.blot = config_dict.ConfigDict()
    blot.enabled = True
    blot.p = 0.5
    blot.params = config_dict.ConfigDict()
    blot.params.rect_config = config_dict.ConfigDict()
    
    blot.params.rect_config.x = (None, None)
    blot.params.rect_config.y = (None, None)
    blot.params.rect_config.h = (25, 50)
    blot.params.rect_config.w = (10 * 2, 30 * 2)
    
    blot.params.params = config_dict.ConfigDict()
    blot.params.params.incline = (10, 50)
    blot.params.params.intensivity = (0.75, 0.75)
    blot.params.params.transparency = (0.05, 0.4)
    blot.params.params.count = (1, 10)


#     iam.image_height = IMAGE_HEIGHT
#     iam.image_width = IMAGE_WIDTH

#     iam.transforms = config_dict.ConfigDict()

#     iam.transforms.vertical_mask_tile_prob = 0.15
#     iam.transforms.vertical_mask_min_height_ratio = 0.25
#     iam.transforms.vertical_mask_max_height_ratio = 0.5

#     iam.transforms.gauss_sigma = (1.0,2.5)
#     iam.transforms.gauss_kernel_size = (5,5)
#     iam.transforms.noise_gauss = (0.02, 0.1)
#     iam.transforms.stretch_factor = (0.8, 1.2)

#     iam.transforms.image_height = IMAGE_HEIGHT
#     iam.transforms.inverse_prob = 0.5

#     iam.transforms.distortion_scale = 0.1

#     iam.transforms.number_of_transforms = 3


#     mjsynth = data.mjsynth = config_dict.ConfigDict()
#     mjsynth.path = os.path.join(data.get_ref('root_path').get(), "mjsynth/")
#     mjsynth.weight = 3

#     mjsynth.transforms = config_dict.ConfigDict()
#     mjsynth.transforms.image_height = IMAGE_HEIGHT
#     mjsynth.transforms.long_lines = config_dict.ConfigDict()
#     mjsynth.transforms.long_lines.prob = 0.2
#     mjsynth.transforms.long_lines.min_space_to_h = 0.5
#     mjsynth.transforms.long_lines.max_space_to_h = 1.0
#     mjsynth.transforms.long_lines.space_value = 0.0

    training = CONFIG.training = config_dict.ConfigDict()

    CONFIG.eval = False

    training.batch_size = 8
    training.num_epochs = 300
    training.eval_epochs_interval = 5
    training.eval_test_interval = 15
    training.snapshot_epochs_interval = 5
    training.checkpoints_folder = "./checkpoints"
    training.loader_num_workers = 6
    training.load_from_checkpoint = False
    training.grad_clip_value = 1


    evaluate = CONFIG.evaluate = config_dict.ConfigDict()
    evaluate.batch_size = 16
    evaluate.loader_num_workers = 6
    evaluate.load_from_checkpoint = True


    optim = CONFIG.optimizer = config_dict.ConfigDict()
    optim.constructor = "torch.optim.AdamW"
    optim.weight_decay = 1e-2
    optim.params = config_dict.ConfigDict()
    optim.params.lr = 2e-4
    optim.params.betas = (0.9, 0.999)

    lr_scheduler = CONFIG.lr_scheduler = config_dict.ConfigDict()
    lr_scheduler.constructor = "torch.optim.lr_scheduler.OneCycleLR"
    lr_scheduler.params = config_dict.ConfigDict()
    lr_scheduler.params.max_lr = 0.001
    lr_scheduler.params.pct_start = 0.1
    lr_scheduler.params.anneal_strategy = 'cos'
    lr_scheduler.params.final_div_factor = 10 ** 5
    lr_scheduler.params.epochs = training.get_ref('num_epochs')
    lr_scheduler.params.steps_per_epoch = (data.get_ref(data.dataset).get().get_ref('length') + training.get_ref('batch_size') - 1) // training.get_ref('batch_size')


    wandb = CONFIG.wandb = config_dict.ConfigDict()
    wandb.project_name = "diploma_dml"
    wandb.run_name = DEFAULT_RUN_NAME
    wandb.resume = False

    CONFIG.device = "cuda:0"

    return CONFIG

def get_config():
    return default_diploma_config()

