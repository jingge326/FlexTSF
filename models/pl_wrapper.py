import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiments.utils_exp import mean_squared_error


class PL_Wrapper(pl.LightningModule):
    def __init__(self, model, args):
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

    def training_step(self, batch, batch_idx):
        pred = self.model.compute(batch)["pred"]
        observed_data = batch["data_out"]
        loss = mean_squared_error(observed_data, pred, batch["mask_out"])
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     return None
    #     # results, _ = self.model.run_validation(batch)
    #     # self.log('val_loss', results['loss'], on_step=True, on_epoch=True)
    #     # self.log('val_mse', results['mse'], on_step=True, on_epoch=True)
    #     # return results['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay,
                                      betas=(self.beta1, self.beta2))
        scheduler = {
            # eta_min = 10% of the maximal learning rate
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.args.pre_total_steps, eta_min=1.0e-5),
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
