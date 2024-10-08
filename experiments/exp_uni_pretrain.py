import logging
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl

from experiments.data_monash import DatasetGP_Monash, read_monash_pretrain_data
from experiments.model_task_data_config import dict_models
from experiments.utils_exp import mean_squared_error


class PL_Wrapper(pl.LightningModule):
    def __init__(self, model, args, local_loger):
        super().__init__()
        self.args = args
        self.model = model
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.warmup_steps = args.warmup_steps
        self.clip_grad = args.clip
        self.save_hyperparameters()
        self.local_loger = local_loger
        num_params_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        self.local_loger.info(
            f'num_params_trainable={num_params_trainable}')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay,
                                      betas=(self.beta1, self.beta2))
        pre_total_steps = self.num_steps()
        scheduler = {
            # eta_min = 10% of the maximal learning rate
            'scheduler': CosineAnnealingLR(optimizer, T_max=pre_total_steps, eta_min=1.0e-5),
            'name': 'cosine_annealing_lr',
            'interval': 'step',
            'frequency': 1}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(
                self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr
        # update params
        optimizer.step(closure=optimizer_closure)

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        self.local_loger.info(f"dataset_size: {dataset_size}")
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size * \
            self.trainer.max_epochs // (
                self.trainer.accumulate_grad_batches * num_devices)
        self.local_loger.info(f"total steps: {num_steps}")
        return num_steps

    def training_step(self, batch, batch_idx):
        results = self.model.forecast(batch)
        pred = results["pred"]
        observed_data = batch["data_out"]
        mse = mean_squared_error(observed_data, pred, batch["mask_out"])
        if results.get("loss") is not None:
            loss = results["loss"]
        else:
            loss = mse
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/mse", mse, on_step=True, on_epoch=True)
        return loss


class PLDataModule_Monash(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_dl_workers = args.num_dl_workers
        self.collate_fn = dict_models[args.base_model][args.ml_task]["collate_fn"]
        self.data_train = read_monash_pretrain_data(args)

    def setup(self, stage):
        self.dataset_train = DatasetGP_Monash(self.data_train, self.args)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            collate_fn=lambda batch: self.collate_fn(
                batch, self.args),
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_dl_workers)


class Exp_Uni_Pretrain:
    ''' Base experiment class '''

    def __init__(self, args: Namespace):
        self.args = args
        self.epochs_max = args.epochs_max
        self.proj_path = Path(args.proj_path)

        self.tags = [self.args.ml_task,
                     self.args.base_model,
                     "nembd"+str(self.args.dim_attn_internal),
                     "nhead"+str(self.args.nhead),
                     "nlyrs"+str(self.args.attn_layers),
                     str(args.vt_norm),
                     str(args.patch_module),
                     str(args.leader_node or args.lyr_time_embed or args.dummy_patch),
                     args.version,
                     args.test_info]

        self.args.exp_name = '_'.join(
            self.tags + [("r"+str(args.random_state))])

        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)

        logging.basicConfig(filename=self.proj_path / 'log' / (self.args.exp_name+'.log'),
                            filemode='w',
                            level=logging.INFO,
                            force=True)

        self.local_loger = logging.getLogger()

    def run(self) -> None:

        core_model = dict_models[self.args.base_model][self.args.ml_task]["class_obj"](
            self.args)

        num_params = sum(p.numel() for p in core_model.parameters())
        print(f"Number of parameters: {num_params}")

        model = PL_Wrapper(core_model, self.args, self.local_loger)

        data_module = PLDataModule_Monash(self.args)

        self.local_loger.info(f"Batch size: {self.args.batch_size}")

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.proj_path/'results/pl_checkpoint',
            filename=self.args.exp_name + "_{epoch:02d}",
            save_top_k=-1)

        lr_monitor = LearningRateMonitor(logging_interval='step')

        if self.args.dev_mode == 'resume':
            wandb_logger = WandbLogger(
                project="flextsf", id="fhr0cw9c", resume="must")
        else:
            wandb_logger = WandbLogger(log_model="all", name=self.args.exp_name,
                                       project="flextsf", tags=self.tags, group="_".join(self.tags),)

        if self.args.dev_mode == 'debug':
            trainer = pl.Trainer(
                accelerator='cpu', logger=wandb_logger,
                min_epochs=self.args.epochs_min,
                max_epochs=self.args.epochs_max,
                gradient_clip_val=self.args.clip,
                callbacks=[checkpoint_callback, lr_monitor])
        elif self.args.dev_mode == 'spec_gpu':
            device = int(self.args.device.split(':')[1]) + 1
            trainer = pl.Trainer(
                devices=device, accelerator='gpu',
                strategy='ddp', logger=wandb_logger,
                min_epochs=self.args.epochs_min,
                max_epochs=self.args.epochs_max,
                gradient_clip_val=self.args.clip,
                callbacks=[checkpoint_callback, lr_monitor])
        else:
            trainer = pl.Trainer(
                devices=-1, accelerator='gpu',
                strategy='ddp', logger=wandb_logger,
                min_epochs=self.args.epochs_min,
                max_epochs=self.args.epochs_max,
                gradient_clip_val=self.args.clip,
                callbacks=[checkpoint_callback, lr_monitor])

        if self.args.dev_mode == 'resume':
            print("Checkpoint path: ", self.args.ckpt_path)
            trainer.fit(model, data_module, ckpt_path=self.args.ckpt_path)
        else:
            trainer.fit(model, data_module)

    def finish(self):
        pass
