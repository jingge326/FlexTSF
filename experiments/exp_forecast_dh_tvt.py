from copy import deepcopy
import copy
from datetime import datetime
import json
import logging
import math
import os
import traceback
from pathlib import Path
import random
import time
import numpy as np
import wandb
import torch
from argparse import Namespace

from experiments.utils_exp import mean_squared_error, mean_absolute_error
from experiments.model_task_data_config import get_general_tvt, dict_models, dataset_info
from models.model_utils import modify_state_dict_keys, reconstruct_pretrained_flextsf


class Exp_Forecast_Dh:
    def __init__(self, args: Namespace):
        self.args = args
        self.proj_path = Path(args.proj_path)
        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        self.device = torch.device(args.device)
        train_setting = args.train_setting
        if train_setting == "few":
            train_setting = "few" + "_" + str(int(args.few_shot_config))
        self.exp_tags = [args.base_model,
                         args.ml_task,
                         args.model_type,
                         train_setting,
                         str(args.vt_norm),
                         str(args.patch_module),
                         str(args.leader_node or args.lyr_time_embed or args.dummy_patch),
                         args.version,
                         args.test_info]
        if self.args.train_setting != "full":
            self.exp_tags.insert(-3, f"ppl{self.args.patch_len_pretrain}")

        if self.args.log_tool == 'wandb':
            # initialize weight and bias
            os.environ["WANDB__SERVICE_WAIT"] = "1800"
            wandb.init(
                project="flextsf",
                config=copy.deepcopy(dict(self.args._get_kwargs())),
                group="_".join(self.exp_tags),
                tags=self.exp_tags,
                name="r"+str(self.args.random_state))

    def get_model(self):
        try:
            class_obj = dict_models[self.args.base_model][self.args.ml_task]["class_obj"]
        except KeyError:
            raise NotImplementedError(
                f'No model available for {self.args.base_model} and {self.args.ml_task}')

        model = class_obj(self.args)

        if self.args.model_type != "initialize":
            model = reconstruct_pretrained_flextsf(
                model, load_para_path=self.proj_path/self.args.pre_model)

        return model

    def get_data(self):
        dl_train, dl_val, dl_test, others = get_general_tvt(
            self.args, self.logger)

        self.scaler = others.get("scaler", None)

        return dl_train, dl_val, dl_test

    def training_step(self, batch):
        results = self.model.forecast(batch)
        if results.get("loss") is not None:
            loss = results["loss"]
        else:
            pred = results["pred"]
            observed_data = batch["data_out"]
            loss = mean_squared_error(observed_data, pred, batch["mask_out"])

        return loss

    def validation_step(self, epoch):
        metrics = self.compute_results_all_batches(self.dlval)
        self.logger.info(f"val_mse={metrics['mse']:.5f}")
        self.logger.info(f"val_mae={metrics['mae']:.5f}")
        self.logger.info(f"val_forward_time={metrics['forward_time']:.5f}")
        if self.args.log_tool == "wandb":
            wandb.log(
                {"val_mse/"+self.args.data_name: metrics['mse'], "epoch_id": epoch})
            wandb.log(
                {"val_mae/"+self.args.data_name: metrics['mae'], "epoch_id": epoch})
        return metrics['loss']

    def test_step(self):
        metrics = self.compute_results_all_batches(
            self.dltest, record_forecasts=True)
        self.logger.info(f"test_mse={metrics['mse']:.5f}")
        self.logger.info(f"test_mae={metrics['mae']:.5f}")
        self.logger.info(f"test_forward_time={metrics['forward_time']:.5f}")
        if self.args.log_tool == "wandb":
            wandb.log(
                {"test_mse/"+self.args.data_name: metrics['mse'], "run_id": 1})
            wandb.log(
                {"test_mae/"+self.args.data_name: metrics['mae'], "run_id": 1})
        return metrics['loss']

    def compute_results_all_batches(self, dl, record_forecasts=False):
        metrics = {}
        metrics["loss"] = 0
        metrics["mse"] = 0
        metrics["mae"] = 0
        metrics["forward_time"] = 0

        n_test_batches = 0
        records = []
        for batch in dl:
            results = self.model.run_validation(batch)
            pred = results['pred']
            truth = batch['data_out']
            if results.get("loss") is None:
                results["loss"] = mean_squared_error(
                    truth, pred, batch["mask_out"])
            pred = pred.detach().cpu().numpy()
            truth = truth.detach().cpu().numpy()
            mask = batch["mask_out"].detach().cpu().numpy()

            if results.get("mse") is None:
                results['mse'] = mean_squared_error(
                    orig=truth, pred=pred, mask=mask)
            if results.get("mae") is None:
                results['mae'] = mean_absolute_error(
                    orig=truth, pred=pred, mask=mask)

            for key in metrics.keys():
                if results.get(key) is not None:
                    var = results[key]
                    if isinstance(var, torch.Tensor):
                        var = var.detach()
                    metrics[key] += var

            if record_forecasts:
                record = {}
                for bk, bv in batch.items():
                    if isinstance(bv, torch.Tensor):
                        record[bk] = bv.detach().cpu().numpy()
                    else:
                        record[bk] = bv
                # transform the pred to the original scale
                if self.scaler is not None:
                    record["pred"] = self.scaler.inverse_transform(
                        pred) * record["mask_out"]
                    record["truth_in"] = self.scaler.inverse_transform(
                        record["data_in"])
                    record["truth_out"] = self.scaler.inverse_transform(
                        record["data_out"])
                else:
                    record["pred"] = pred * record["mask_out"]
                    record["truth_in"] = record["data_in"]
                    record["truth_out"] = record["data_out"]
                records.append(record)

            n_test_batches += 1

        if n_test_batches > 0:
            for key, _ in metrics.items():
                metrics[key] = metrics[key] / n_test_batches

        # Save forecasts to a file
        if record_forecasts:
            np.save(self.proj_path / 'results/forecasts' /
                    (self.args.exp_name+'.npy'), records)

        return metrics

    def run_tvt_exp(self):
        num_params_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(
            f'num_params_trainable={num_params_trainable}')

        self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                      lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.scheduler = None
        if self.args.lr_scheduler_step > 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optim, self.args.lr_scheduler_step, self.args.lr_decay)

        # Training loop parameters
        best_loss = float('inf')
        waiting = 0
        durations = []
        best_model = deepcopy(self.model.state_dict())

        for epoch in range(1, self.args.epochs_max):
            iteration = 1
            self.model.train()
            start_time = time.time()

            for batch in self.dltrain:
                # Single training step
                self.optim.zero_grad()
                train_loss = self.training_step(batch)
                train_loss.backward()
                if self.args.clip_gradient:
                    # Optional gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip)
                self.optim.step()

                self.logger.info(
                    f'[epoch={epoch:04d}|iter={iteration:04d}] train_loss={train_loss:.5f}')
                if self.args.log_tool == 'wandb':
                    wandb.log({"train_loss/"+self.args.data_name: train_loss})
                iteration += 1

            epoch_duration = time.time() - start_time
            durations.append(epoch_duration)
            self.logger.info(
                f'[epoch={epoch:04d}] epoch_duration={epoch_duration:5f}')

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_loss = self.validation_step(epoch)
            self.logger.info(
                f'[epoch={epoch:04d}] val_loss={val_loss:.5f}')
            if self.args.log_tool == 'wandb':
                wandb.log(
                    {"val_loss/"+self.args.data_name: val_loss, "epoch_id": epoch})

            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

            # Early stopping procedure
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.model.state_dict())
                waiting = 0
            else:
                waiting += 1

            if waiting >= self.args.patience:
                break

            if self.args.log_tool == 'wandb':
                wandb.log(
                    {"lr/"+self.args.data_name: self.optim.param_groups[0]['lr'], "epoch_id": epoch})

        # Load best model
        self.model.load_state_dict(best_model)
        # Held-out test set step
        with torch.no_grad():
            test_loss = self.test_step()

        self.logger.info(
            f'epoch_duration_mean={np.mean(durations):.5f}')
        self.logger.info(f'test_loss={test_loss:.5f}')

        if self.args.log_tool == 'wandb':
            wandb.log({"test_loss/"+self.args.data_name: test_loss, "run_id": 1})

        self.record_experiment(self.args, self.model)
        torch.save(self.model.state_dict(), self.proj_path /
                   'results/model_para' / (self.args.exp_name+'.pt'))

        logging.shutdown()

    def run_zeroshot_exp(self):
        self.model.eval()
        path_ckpts = self.proj_path/"results/pl_checkpoint/"
        if self.args.test_info_pt == "":
            self.args.test_info_pt = self.args.test_info

        with torch.no_grad():
            # Collect and filter checkpoint files based on specified conditions
            list_ckpts = []
            for f in os.listdir(path_ckpts):
                f_terms = f.split("_")
                if (f"{self.args.version}" in f_terms and
                    f"r{self.args.pre_random_seed}" in f_terms and
                    self.args.base_model in f_terms and
                        self.args.test_info_pt in f_terms):
                    list_ckpts.append(os.path.join(path_ckpts, f))

            # Sort checkpoints by modification time
            list_ckpts.sort(key=os.path.getmtime)

            # Iterate through filtered checkpoints
            done = False
            for path_ckpt in list_ckpts:
                # Extract epoch number
                epoch_num = int(path_ckpt.split("epoch=")[-1].split(".")[0])

                if epoch_num != self.args.zeroshot_epoch:
                    continue
                done = True

                # Log checkpoint path and epoch number
                self.logger.info(f"\n{path_ckpt}\nEpoch {epoch_num}")

                # Load and modify state_dict
                state_dict = modify_state_dict_keys(
                    torch.load(path_ckpt)['state_dict']
                )
                self.model.load_state_dict(state_dict, strict=True)

                # Compute results
                metrics = self.compute_results_all_batches(
                    self.dltest, record_forecasts=True)

                # Log metrics
                self.logger.info(
                    f"Epoch {epoch_num} test_mse={metrics['mse']:.5f}")
                self.logger.info(
                    f"Epoch {epoch_num} test_mae={metrics['mae']:.5f}")

                # Log to wandb if required
                if self.args.log_tool == "wandb":
                    wandb.log({
                        f"test_mse/{self.args.data_name}": metrics['mse'],
                        f"test_mae/{self.args.data_name}": metrics['mae'],
                        "epoch_num": epoch_num
                    })

            if not done:
                raise ValueError(
                    f"Epoch {self.args.zeroshot_epoch} not found in checkpoints")

    def run(self) -> None:

        if self.args.data_name != "":
            datasets = [self.args.data_name]
        else:
            if self.args.data_group == "regular":
                datasets = ["etth2", "ettm2", "exchange_rate",
                            "illness", "weather", "HARPhone"]
            elif self.args.data_group == "irregular":
                datasets = ["metr_la", "SpokenArabicDigits",
                            "CharacterTrajectories", "HARw4IMU", "eICU", "PhysioNet2012"]
            else:
                datasets = ["etth2", "ettm2", "exchange_rate", "illness", "weather", "HARPhone",
                            "metr_la", "SpokenArabicDigits", "CharacterTrajectories", "HARw4IMU", "eICU", "PhysioNet2012"]

        for data_name in datasets:
            try:
                tags_data_model = self.exp_tags.copy()
                tags_data_model.insert(1, data_name)
                self.args.exp_name = '_'.join(
                    tags_data_model + [("r"+str(self.args.random_state))])

                logging.basicConfig(filename=self.proj_path/'log'/(self.args.exp_name+'.log'),
                                    filemode='w',
                                    level=logging.INFO,
                                    force=True)
                self.logger = logging.getLogger()

                self.logger.info(f'Device: {self.device}')

                self.args.data_name = data_name

                dc = dataset_info[data_name]["configs"]
                self.args.var_num = dc["var_num"]
                self.logger.info(f"var_num={self.args.var_num}")

                self.args.ltf_pred_len = int(
                    dc["seq_len"] * self.args.forecast_ratio)
                self.args.ltf_input_len = dc["seq_len"] - \
                    self.args.ltf_pred_len
                self.logger.info(
                    f"ltf_input_len={self.args.ltf_input_len}, ltf_pred_len={self.args.ltf_pred_len}")

                if dataset_info[data_name].get("hyperparameters") is not None:
                    hp = dataset_info[data_name]["hyperparameters"]
                    # Configure hyperparameters
                    if hp.get(self.args.base_model) is not None:
                        for key, value in hp[self.args.base_model].items():
                            setattr(self.args, key, value)
                            self.logger.info(f"{key}={value}")

                if self.args.model_type == "reconstruct" or self.args.train_setting == "zero":
                    self.args.batch_size = int(
                        32/math.ceil(math.pow((dc["var_num"]), 1.5) * dc["seq_len"] / 8000))
                    self.logger.info(
                        f"Batch size for {data_name}: {self.args.batch_size}")

                self.model = self.get_model().to(self.device)
                self.dltrain, self.dlval, self.dltest = self.get_data()
                # Record the number of samples of each dataset
                self.logger.info(
                    f"train_num={len(self.dltrain.dataset)}, val_num={len(self.dlval.dataset)}, test_num={len(self.dltest.dataset)}")

                num_params = sum(p.numel() for p in self.model.parameters())
                self.logger.info(f'num_params={num_params}')

                if self.args.train_setting == 'zero':
                    self.run_zeroshot_exp()
                else:
                    self.run_tvt_exp()
            except Exception as e:
                self.logger.error(traceback.format_exc())
                self.logger.error(str(e))
                continue

    def record_experiment(self, args, model):
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")
        file_result = self.proj_path / \
            'results/model_hyper/{}.txt'.format(dt_string)
        with open(file_result, 'w') as fr:
            args_dict = vars(args)
            # Convert any PosixPath objects to strings
            args_dict = {key: str(value) for key, value in args_dict.items()}
            json.dump(args_dict, fr, indent=0)
            print(model, file=fr)
