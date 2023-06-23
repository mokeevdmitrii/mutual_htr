from diploma_code.model_v2 import (
    Resnet34Backbone, BiLSTMEncoder, PositionalEncoding, TransformerEncoder, CTCDecoderModel,
    ParallelModel, make_single_model_v2, make_model_v2
)
from diploma_code.char_encoder import (
    CharEncoder
)
from diploma_code.dataset import (
    BaseLTRDataset, LongLinesLTRDataset
)
from diploma_code.optimizing import (
    pytorch_make_optimizer, StepLRWithWarmup, ConstantLambdaLR, ReverseSqrtWithLinearWarmup,
    make_lr_scheduler, make_lr_scheduler_torch, make_lr_scheduler_lambda
)
from diploma_code.utils import (
    log_metric_wandb, batch_to_device, dict_collate_fn, seed_everything
)
from diploma_code.evaluation import (
    my_ctc_loss, my_dml_loss, my_mcl_loss, my_dml_loss_tr, 
    trkl_div, decode_ocr_probs, get_edit_distance,
    EpochValueProcessor, EpochDMLProcessor, EpochDivProcessor, CERProcessor
)
from diploma_code.make_loader import (
    make_char_encoder, make_dataloaders
)

from ml_collections import ConfigDict

import typing as tp
import torch
import os
import wandb
import numpy as np

from tqdm import tqdm



class LTRTrainer:

    def __init__(self, config: ConfigDict):

        self.config = config
        self.model = make_model_v2(config.model)
        self.checkpoints_folder = config.training.checkpoints_folder
        
        self.load_checkpoints_folder = config.load_checkpoints_folder
        
        self.evaluate = config.eval
        
        self.model_type = self.config.model.type

        self.had_nan = False
        self.had_grad_nan = False
        
        self.device = torch.device(config.device)
        self.model.to(self.device)

        if not self.evaluate:
            self.configure_train_state()

        do_load_from_checkpoint = config.evaluate.load_from_checkpoint if self.evaluate else config.training.load_from_checkpoint

        if do_load_from_checkpoint:
            self.load_from_checkpoint()

        self.char_encoder = make_char_encoder(config.data)
        
    @staticmethod
    def merge_model_checkpoints(folder1: str, folder2: str, res_folder: str):
        first_model = torch.load(LTRTrainer.get_model_path(folder1))['model']
        second_model = torch.load(LTRTrainer.get_model_path(folder2))['model']
        
        res_dict = {'model': {}}
        for name, param in first_model.items():
            res_dict['model'][f'models.0.{name}'] = param
        for name, param in second_model.items():
            res_dict['model'][f'models.1.{name}'] = param
        
        os.makedirs(res_folder, exist_ok=True)
        torch.save(res_dict, LTRTrainer.get_model_path(res_folder))
        
    @staticmethod
    def get_model_path(folder):
        return os.path.join(folder, "model.pth")

    @property
    def checkpoint_model_path(self):
        return LTRTrainer.get_model_path(self.checkpoints_folder)
    
    @property
    def checkpoint_training_state_path(self):
        return os.path.join(self.checkpoints_folder, "train_state.pth")
    
    @property
    def load_checkpoint_model_path(self):
        return LTRTrainer.get_model_path(self.load_checkpoints_folder)
    
    @property
    def load_checkpoint_training_state_path(self):
        return os.path.join(self.load_checkpoints_folder, "train_state.pth")
    
    @property
    def checkpoint_nan_model_path(self):
        return os.path.join(self.nan_folder, 'nan_model_state.pth')

    @staticmethod
    def validate_mode(mode: str) -> None:
        if mode != 'train' and mode != 'valid' and mode != 'test':
            raise ValueError(f"Invalid mode: {mode}, expected one of 'train', 'valid', 'test'")

    def load_from_checkpoint(self):

        checkpoint = torch.load(self.load_checkpoint_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        if not self.evaluate:

            self.configure_train_state()
            
            if self.config.training.load_train_state:
                train_checkpoint = torch.load(self.load_checkpoint_training_state_path)

                self.backbone_optimizer.load_state_dict(train_checkpoint['backbone_optimizer'])
                self.optimizer.load_state_dict(train_checkpoint['optimizer'])
                self.backbone_lr_scheduler.load_state_dict(train_checkpoint['backbone_lr_scheduler'])
                self.lr_scheduler.load_state_dict(train_checkpoint['lr_scheduler'])
                self.epoch = train_checkpoint['epoch']
                self.step = train_checkpoint['step']

    def save_checkpoint(self):
        if not os.path.exists(self.checkpoints_folder):
            os.makedirs(self.checkpoints_folder)
        torch.save({
            'model': self.model.state_dict()
        }, self.checkpoint_model_path)

        torch.save({
            'backbone_optimizer': self.backbone_optimizer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'backbone_lr_scheduler': self.backbone_lr_scheduler.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'step': self.step
        }, self.checkpoint_training_state_path)
        
    def save_nan_checkpoint(self):
        if not os.path.exists(self.checkpoints_folder):
                os.makedirs(self.checkpoints_folder)
        torch.save({
            'model': self.model.state_dict()
        }, self.checkpoint_nan_model_path)
        torch.save({
            batch, self.checkpoint_nan_batch_path
        })

    def log_metric(self, metric_name, loader_name, value):
        log_metric_wandb(metric_name, loader_name, value, self.step)

        
    def configure_train_state(self):
        self.optimizer = pytorch_make_optimizer(self.model, self.config.optimizer, lambda x: 'backbone' not in x)
        self.backbone_optimizer = pytorch_make_optimizer(self.model, self.config.backbone_optimizer, lambda x: 'backbone' in x)
        
        self.lr_scheduler = make_lr_scheduler(self.optimizer, self.config.lr_scheduler)
        self.backbone_lr_scheduler = make_lr_scheduler(self.backbone_optimizer, self.config.backbone_lr_scheduler)
        
        self.epoch = 1
        self.step = 0


    def configure_loader(self, mode: str) -> torch.utils.data.DataLoader:
        LTRTrainer.validate_mode(mode)

        attr_name = f"{mode}_loader"
        try:
            return self.__getattribute__(attr_name)
        except AttributeError:
            loaders = make_dataloaders(self.config)
            self.train_loader = loaders['train']
            self.valid_loader = loaders['valid']
            self.test_loader = loaders['test']
            
            return self.__getattribute__(attr_name)


    def make_criterion_processor(self, mode, report_per_batch, model_type=None, name='loss', report_final=True):
        if model_type is None:
            model_type = self.config.model.type
        if model_type == "single":
            return EpochValueProcessor(name, mode, report_per_batch=report_per_batch, report_final=report_final)
        elif model_type == "duo":
            return EpochDMLProcessor(name, mode, self.config.dml, report_per_batch=report_per_batch, report_final=report_final)
        elif model_type == "duo_tr":
            return EpochDMLProcessor(name, mode, self.config.dml, report_per_batch=report_per_batch, report_final=report_final, truncated=True)
        elif model_type == "duo_div":
            return EpochDivProcessor(name, mode, report_per_batch=report_per_batch, report_final=report_final)
        else:
            raise ValueError(f"Invalid model mode: {model_type}")

    def make_cer_processor(self, mode, name='cer', report_per_batch=False, report_final=True):
        return CERProcessor(self.char_encoder, name, mode, report_per_batch=report_per_batch, report_final=report_final)

    def calc_loss(self, log_probs, gt_text, inp_len, tgt_len, model_type=None):
        if model_type is None:
            model_type = self.config.model.type
        if model_type == "single":
            return my_ctc_loss(log_probs, gt_text, inp_len, tgt_len)
        elif model_type == "duo":
            return my_dml_loss(log_probs, gt_text, inp_len, tgt_len)
        elif model_type == "duo_tr":
            return my_dml_loss_tr(log_probs, gt_text, inp_len, tgt_len)
        elif model_type == "duo_div":
            return my_mcl_loss(log_probs, gt_text, inp_len, tgt_len)
        else:
            raise ValueError(f"Invalid model mode: {model_type}")

    def validate_inf_or_nan_loss(self, loss, batch, log_preds, inp_len):
        val = loss.item()
        if (np.isnan(val) or np.isinf(val) or self.had_grad_nan) and not self.had_nan:
            self.had_nan = True
            if not os.path.exists(self.checkpoints_folder):
                os.makedirs(self.checkpoints_folder)
            with open(os.path.join(self.checkpoints_folder, "nan.txt"), "w+") as f:
                for k, v in batch.items():
                    f.write(f"'{k}' shape: {v.__getattr__('shape', None)}")
                    f.write(f"'{k}': {v}")
                f.write(f"loss: {val}\n")
                f.write(f"inp_len: {inp_len}\n")
            
            self.save_nan_checkpoint()
            
        if np.isnan(val) or np.isinf(val):
            # zero out huge losses
            loss.fill_(0)


    def train(self, run_name):

        loader = self.configure_loader(mode='train')

        wandb.init(project=self.config.wandb.project_name, name=run_name, resume=self.config.wandb.resume)

        criterion_processor = self.make_criterion_processor(mode='train', report_per_batch=True, report_final=True)

        def grad_hook(grad):
            out = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            out = torch.clamp(out, -self.config.training.grad_clip_value, self.config.training.grad_clip_value)
            return out

        for p in self.model.parameters():
            p.register_hook(grad_hook)

        while self.epoch <= self.config.training.num_epochs:

            self.model.train()
            for batch in tqdm(loader):
                self.step += 1

                batch = batch_to_device(batch, self.device)
                gt_text, tgt_len = batch['gt_text'], batch['encoded_length']

                logits = self.model(batch['image'])
                
                if self.model_type == 'single':
                    inp_len = torch.IntTensor([logits.size(1)] * batch['encoded'].shape[0]).to(self.device)   
                    log_probs = logits.log_softmax(2).permute(1, 0, 2).to(self.device)
                elif str(self.model_type).startswith('duo'):
                    inp_len = tuple(torch.IntTensor([l.size(1)] * batch['encoded'].shape[0]).to(self.device) for l in logits)
                    log_probs = tuple(l.log_softmax(2).permute(1, 0, 2).to(self.device) for l in logits)
                    
                                        
                loss = self.calc_loss(log_probs, batch['encoded'], inp_len, tgt_len)

                result_loss = criterion_processor(loss, len(gt_text), self.step)

                self.validate_inf_or_nan_loss(result_loss, batch, log_probs, inp_len)

                self.optimizer.zero_grad()
                self.backbone_optimizer.zero_grad()

                result_loss.backward()
                
                self.log_grads()

                self.optimizer.step()
                self.backbone_optimizer.step()
                
                last_lrs = self.lr_scheduler.get_last_lr()
                for i, lr in enumerate(last_lrs):
                    self.log_metric(f"lr_{i}", "train", lr)
                    
                last_backbone_lrs = self.backbone_lr_scheduler.get_last_lr()
                for i, lr in enumerate(last_backbone_lrs):
                    self.log_metric(f"backbone_lr_{i}", "train", lr)

                self.lr_scheduler.step()
                self.backbone_lr_scheduler.step()

            criterion_processor.finalize(self.step)

            self.model.eval()

            if self.epoch % self.config.training.eval_epochs_interval == 0:
                self.validate(mode='valid')

            if self.epoch % self.config.training.eval_test_interval == 0:
                self.validate(mode='test')

            if self.epoch % self.config.training.snapshot_epochs_interval == 0:
                self.save_checkpoint()

            self.epoch += 1

        self.save_checkpoint()
        wandb.finish()
        
    def log_grads(self):
        grad_dict = {
            "model": [],
            "backbone": []
        }
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if 'backbone' in name:
                grad_dict['backbone'].append(param.grad.detach().flatten())
            else:
                grad_dict['model'].append(param.grad.detach().flatten())
        
        for k, grad_list in grad_dict.items():
            if len(grad_list) > 0:
                norm = torch.cat(grad_list).norm()
                self.log_metric(f'{k}_grad_norm', 'train', norm.item())


    def validate(self, mode):
        if str(self.model_type).startswith('duo'):
            return self.validate_duo(mode)
        
        val_loader = self.configure_loader(mode=mode)

        criterion_processor = self.make_criterion_processor(mode=mode, report_per_batch=False, report_final=True)
        cer_processor = self.make_cer_processor(mode=mode)
        
        self.validate_impl(self.model, val_loader, criterion_processor, cer_processor)
    
    
    def validate_duo(self, mode):
        
        val_loader = self.configure_loader(mode=mode)
        
        loss_1 = self.make_criterion_processor(mode=mode, model_type='single', report_per_batch=False, report_final=True, name='loss_1')
        loss_2 = self.make_criterion_processor(mode=mode, model_type='single', report_per_batch=False, report_final=True, name='loss_2')
        
        cer_1 = self.make_cer_processor(mode=mode, name='cer_1')
        cer_2 = self.make_cer_processor(mode=mode, name='cer_2')
        
        self.validate_impl(self.model.models[0], val_loader, loss_1, cer_1)
        self.validate_impl(self.model.models[1], val_loader, loss_2, cer_2)
    
    def validate_impl(self, model, val_loader, criterion_processor, cer_processor):

        prev_mode = model.training
        model.eval()

        def do_iter(batch, criterion_processor, cer_processor):
            batch = batch_to_device(batch, self.device)
            gt_text, tgt_len = batch['gt_text'], batch['encoded_length']

            logits = model(batch['image'])
            inp_len = torch.IntTensor([logits.size(1)] * batch['encoded'].shape[0])         
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            log_probs, inp_len = log_probs.to(self.device), inp_len.to(self.device)

            loss = self.calc_loss(log_probs, batch['encoded'], inp_len, tgt_len, model_type='single')

            _ = criterion_processor(loss, len(gt_text), self.step)
            _ = cer_processor(log_probs, gt_text, self.step)


        with torch.no_grad():
            for batch in tqdm(val_loader):
                do_iter(batch, criterion_processor, cer_processor)
            _ = criterion_processor.finalize(self.step)
            _ = cer_processor.finalize(self.step)

        model.train(prev_mode)

    def evaluate(self):
        self.validate(mode='test')

