import torch

from ml_collections import ConfigDict

from diploma_code.model_v2 import PositionalEncoding

def pytorch_configure_optim_groups(model, weight_decay=0):
    """
    I love and hate pytorch
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, PositionalEncoding, torch.nn.MultiheadAttention, torch.nn.LSTM)
    blacklist_weight_modules = (torch.nn.BatchNorm2d, torch.nn.LayerNorm, )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif 'weight' in pn and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif 'weight' in pn and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)


    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups


def pytorch_make_optimizer(model: torch.nn.Module, optimizer_config: ConfigDict):
    if 'weight_decay' in optimizer_config:
        optim_groups = pytorch_configure_optim_groups(model, optimizer_config.weight_decay)
    else:
        optim_groups = [{"params": model.parameters()}]

    return eval(optimizer_config.constructor)(optim_groups, **optimizer_config.params)


class StepLRWithWarmup:

    def __init__(self, initial_lr, warmup_steps, warmup_lr, step, decay):
        self.warmup_steps = int(warmup_steps)
        self.initial_lr = initial_lr
        self.warmup_lr = warmup_lr
        self.step = int(step)
        self.decay = decay

    def __call__(self, epoch: int):
        if epoch <= self.warmup_steps:
            desired_lr = self.warmup_lr * epoch / self.warmup_steps
        else:
            desired_lr = self.warmup_lr * (self.decay ** (epoch // self.step))
        return desired_lr / self.initial_lr


def make_lr_scheduler(optimizer, scheduler_config):
    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                             lr_lambda=eval(scheduler_config.constructor)(
                                                 **scheduler_config.params
                                             ))

def make_lr_scheduler_torch(optimizer, scheduler_config):
    return eval(scheduler_config.constructor)(optimizer=optimizer, **scheduler_config.params)

