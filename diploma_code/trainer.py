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
    pytorch_make_optimizer, StepLRWithWarmup, make_lr_scheduler, make_lr_scheduler_torch
)
from diploma_code.utils import (
    log_metric_wandb, batch_to_device, dict_collate_fn, seed_everything
)
from diploma_code.evaluation import (
    my_ctc_loss, my_dml_loss, decode_ocr_probs, get_edit_distance,
    EpochValueProcessor, EpochDMLProcessor, CERProcessor
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
        self.evaluate = config.eval

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

    @property
    def checkpoint_model_path(self):
        return os.path.join(self.checkpoints_folder, "model.pth")

    @property
    def checkpoint_training_state_path(self):
        return os.path.join(self.checkpoints_folder, "train_state.pth")
    
    @property
    def checkpoint_nan_model_path(self):
        return os.path.join(self.nan_folder, 'nan_model_state.pth')

    @staticmethod
    def validate_mode(mode: str) -> None:
        if mode != 'train' and mode != 'valid' and mode != 'test':
            raise ValueError(f"Invalid mode: {mode}, expected one of 'train', 'valid', 'test'")

    def load_from_checkpoint(self):

        checkpoint = torch.load(self.checkpoint_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        if not self.evaluate:
            self.configure_train_state()

            train_checkpoint = torch.load(self.checkpoint_training_state_path)

            self.optimizer.load_state_dict(train_checkpoint['optimizer'])
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
            'optimizer': self.optimizer.state_dict(),
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
        self.optimizer = pytorch_make_optimizer(self.model, self.config.optimizer)
        self.lr_scheduler = make_lr_scheduler_torch(self.optimizer, self.config.lr_scheduler)
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



    def make_criterion_processor(self, mode, report_per_batch, report_final=True):
        if self.config.model.type == "single":
            return EpochValueProcessor('loss', mode, report_per_batch=report_per_batch, report_final=report_final)
        elif self.config.model.type == "duo":
            return EpochDMLProcessor("loss", mode, report_per_batch=report_per_batch, report_final=report_final)
        else:
            raise ValueError(f"Invalid model mode: {mode}")

    def make_cer_processor(self, mode, report_per_batch=False, report_final=True):
        return CERProcessor(self.char_encoder, 'cer', mode, report_per_batch=report_per_batch, report_final=report_final)

    def calc_loss(self, log_probs, gt_text, inp_len, tgt_len):
        if self.config.model.type == "single":
            return my_ctc_loss(log_probs, gt_text, inp_len, tgt_len)
        elif self.config.model.type == "duo":
            return my_dml_loss(log_probs, gt_text, inp_len, tgt_len)
        else:
            raise ValueError(f"Invalid model mode: {self.config.model.type}")

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
                inp_len = torch.IntTensor([logits.size(1)] * batch['encoded'].shape[0])         
                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                log_probs, inp_len = log_probs.to(self.device), inp_len.to(self.device)
                                        
                loss = self.calc_loss(log_probs, batch['encoded'], inp_len, tgt_len)

                result_loss = criterion_processor(loss, len(gt_text), self.step)

                self.validate_inf_or_nan_loss(result_loss, batch, log_probs, inp_len)

                result_loss.backward()

                grads = [
                    param.grad.detach().flatten()
                    for param in self.model.parameters()
                    if param.grad is not None
                ]
                if len(grads) > 0:
                    norm = torch.cat(grads).norm()
                    self.log_metric("grad_norm", "train", norm.item())

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.log_metric("lr", "train", self.lr_scheduler.get_last_lr()[0])

                self.lr_scheduler.step()

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


    def validate(self, mode):
        val_loader = self.configure_loader(mode=mode)

        criterion_processor = self.make_criterion_processor(mode=mode, report_per_batch=False, report_final=True)
        cer_processor = self.make_cer_processor(mode=mode)

        prev_mode = self.model.training
        self.model.eval()

        def do_iter(batch, criterion_processor, cer_processor):
            batch = batch_to_device(batch, self.device)
            gt_text, tgt_len = batch['gt_text'], batch['tgt_len']

            logits = self.model(batch['image'])
            inp_len = torch.IntTensor([logits.size(1)] * batch['encoded'].shape[0])         
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            log_probs, inp_len = log_probs.to(self.device), inp_len.to(self.device)

            loss = self.calc_loss(log_probs, batch['encoded'], inp_len, tgt_len)

            _ = criterion_processor(loss, len(gt_text), self.step)
            _ = cer_processor(log_probs, gt_text, self.step)


        with torch.no_grad():
            for batch in tqdm(val_loader):
                do_iter(batch, criterion_processor, cer_processor)
            _ = criterion_processor.finalize(self.step)
            _ = cer_processor.finalize(self.step)

        self.model.train(prev_mode)


    def evaluate(self):
        self.validate(mode='test')

