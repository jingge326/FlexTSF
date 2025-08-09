import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.utils_exp import compute_log_normal_pdf
from models.flextsf_components import Patcher, TransformerDecoder, ValueNorm, TimeNorm, ValueNormMinMax


class FlexTSF_General_Forecast(TransformerDecoder):
    def __init__(self, args):
        super().__init__(args)
        self.patcher = Patcher(args)
        if self.args.patch_seg == "random":
            self.patch_info = {i: {"cnt": 0, "loss_accu": 0.0, "weight": 0.0}
                               for i in range(args.smallest_patch_len, args.largest_patch_len+1)}
            self.pweight = None

        if self.args.value_norm == True:
            if self.args.v_norm_type == "standard":
                self.value_norm = ValueNorm(
                    num_features=1, affine=True, subtract_last=False)
            else:
                self.value_norm = ValueNormMinMax(
                    num_features=1, affine=True)
        if self.args.time_norm == True:
            self.time_norm = TimeNorm(affine=self.args.time_norm_affine)

        if args.leader_node == True:
            dim_in = 2
            if self.args.value_norm == True:
                dim_in += 2
            if self.args.time_norm == True:
                dim_in += 2
            self.leaderlyr = nn.Sequential(
                nn.Linear(dim_in, 128),
                nn.Tanh(),
                nn.Linear(128, args.dim_attn_internal))

        self.state_mapper_in = nn.Linear(
            args.dim_patch_ts, args.dim_attn_internal)
        self.state_mapper_out = nn.Linear(
            args.dim_attn_internal, args.dim_patch_ts)

    def forecast(self, batch, k_iwae=1, patch_len=None):
        # The beginning of time_in is 0
        assert (batch["time_in"][..., 0] == 0).all()

        if self.args.patch_module != "ivp":
            k_iwae = 1

        if self.args.value_norm == True:
            # Normalize values
            data_in, ins_mean, ins_std = self.value_norm(
                batch['data_in'], mask=batch['mask_in'], mode='norm')
        else:
            data_in = batch['data_in']

        if self.args.time_norm == True:
            # Normalize times
            time_in, t_unit_inst = self.time_norm(
                batch['time_in'], mask=batch["mask_in"].any(dim=-1))
            # also normalize time_out using the same t_unit_inst
            time_out = self.time_norm.normalize(batch["time_out"])
        else:
            time_in = batch["time_in"]
            time_out = batch["time_out"]

        if self.args.leader_node == True:
            if len(batch["gmean"].shape) == 1:
                gmean = batch["gmean"].repeat(
                    batch["data_in"].size(0), 1).view(-1, 1)
                gstd = batch["gstd"].repeat(
                    batch["data_in"].size(0), 1).view(-1, 1)
            else:
                gmean = batch["gmean"]
                gstd = batch["gstd"]
            stat = torch.cat([gmean, gstd], dim=-1)

            if self.args.value_norm == True:
                ins_mean = ins_mean.view(-1, 1)
                ins_std = ins_std.view(-1, 1)
                stat = torch.cat([stat, ins_mean, ins_std], dim=-1)

            if self.args.time_norm == True:
                # static features from the time normalization
                t_unit_inst = t_unit_inst.repeat_interleave(
                    batch['data_in'].size(-1)).view(-1, 1)
                if len(batch["time_unit"].shape) == 0:
                    t_unit_global = batch["time_unit"].repeat_interleave(
                        t_unit_inst.size(0)).view(-1, 1)
                else:
                    t_unit_global = batch["time_unit"].view(-1, 1)
                # concatenate all the static features
                stat = torch.cat([stat, t_unit_inst, t_unit_global], dim=-1)

            batch['stat'] = stat.unsqueeze(1)

        # Added for self.args.ar_gen_way == "simul"
        time_in = time_in + time_out
        time_in = time_in.repeat_interleave(batch['data_in'].size(-1), dim=0)

        ###### Patching ######
        # Set the patch length
        if self.args.patch_seg == "random":
            if patch_len is None:
                # randomly choose the patch length
                patch_len = random.choice(tuple(self.patch_info))
        else:
            patch_len = batch["patch_len"]

        # Channel independent. Time dimension at the end
        data_in = data_in.permute(0, 2, 1)
        mask_in = batch["mask_in"].permute(0, 2, 1)
        mask_out = batch["mask_out"].permute(0, 2, 1)

        stride = patch_len - int(
            patch_len * self.args.patch_overlap_rate)

        # Pad the last dim to make sure the last patch is complete
        data_in = F.pad(
            data_in, (0, stride), mode="replicate")
        mask_in = F.pad(
            mask_in, (0, stride), mode="constant", value=0)
        mask_out = F.pad(
            mask_out, (0, stride), mode="constant", value=0)
        time_in = F.pad(time_in, (0, stride), mode="replicate")

        # Splitting the sequence to get patch segments
        data_in = data_in.unfold(
            dimension=-1, size=patch_len, step=stride)
        mask_in = mask_in.unfold(
            dimension=-1, size=patch_len, step=stride)
        mask_out = mask_out.unfold(
            dimension=-1, size=patch_len, step=stride)
        time_in = time_in.unfold(
            dimension=-1, size=patch_len, step=stride)

        # Reshape the data
        batch_size, num_vars, num_patches_in, _ = mask_in.shape
        _, _, num_patches_out, _ = mask_out.shape
        data_in = data_in.view(-1, num_patches_in, patch_len)
        mask_in = mask_in.view(-1, num_patches_in, patch_len)

        # contain empty patches in between and at the end of the sequence
        exist_patch_in = mask_in.any(dim=-1)
        exist_patch_out = mask_out.any(dim=-1).view(-1, num_patches_out)

        if self.args.ar_in_pads == True:
            exist_patch_in = exist_patch_in + exist_patch_out

        # Align the timestamps of each patch
        oid_in = time_in[:, :, 0]
        time_in = time_in - oid_in.unsqueeze(-1)
        t = oid_in

        input_states, others = self.patcher(
            data_in, time_in, mask_in, k_iwae, True)
        input_states = self.state_mapper_in(input_states)

        if self.args.patch_module == "ivp":
            loss_input = others["patching_loss"]

        # Adjust the shape of the data to match the number of samples in the generation part
        mask_in = mask_in.unsqueeze(0).repeat(k_iwae, 1, 1, 1)
        t = t.repeat(k_iwae, 1, 1)
        exist_patch_in = exist_patch_in.repeat(k_iwae, 1)

        dim0 = k_iwae * batch_size * num_vars

        # Add the leader node
        if self.args.leader_node == True:
            batch["stat"] = batch["stat"].unsqueeze(0).repeat(k_iwae, 1, 1, 1)
            leader = self.leaderlyr(batch['stat'])
            input_states = torch.cat([leader, input_states], dim=-2)
            t = torch.cat([torch.ones_like(t[..., 0:1]) * -1, t], dim=-1)
            exist_patch_in = torch.cat([torch.ones_like(
                exist_patch_in[:, 0:1]).bool(), exist_patch_in], dim=-1)

        input_states = input_states.view(dim0, *input_states.shape[-2:])
        t = t.view(dim0, *t.shape[-1:])

        hidden_states = self.forward(input_states, t, exist_patch_in)

        if self.args.leader_node == True:
            hidden_states = hidden_states[:, 1:, :]

        hidden_states = self.state_mapper_out(hidden_states)

        hidden_states = hidden_states.view(
            k_iwae, batch_size * num_vars, *hidden_states.shape[1:])

        output_value = self.patcher.generate(
            hidden_states, time_in.repeat(k_iwae, 1, 1, 1)) * exist_patch_out.unsqueeze(-1)

        gen_seq = output_value.view(
            k_iwae, batch_size, num_vars, -1, patch_len)

        if self.args.patch_overlap_rate == 0:
            pred = gen_seq.view(
                k_iwae, batch_size, num_vars, -1)[:, :, :, :batch['data_out'].size(1)].permute(0, 1, 3, 2)
        else:
            gen_seq1 = gen_seq.permute(0, 1, 2, 4, 3).contiguous()
            gen_seq2 = gen_seq1.view(
                k_iwae * batch_size * num_vars, patch_len, num_patches_out)

            pred = F.fold(
                gen_seq2,
                output_size=(1, patch_len+stride*(num_patches_out-1)),
                kernel_size=(1, patch_len),
                stride=(1, stride)
            )

            overlap_mask = torch.ones_like(gen_seq2)
            overlap_counter = F.fold(
                overlap_mask,
                output_size=(1, patch_len+stride*(num_patches_out-1)),
                kernel_size=(1, patch_len),
                stride=(1, stride)
            )

            pred = pred / overlap_counter

            pred = pred[..., :batch['data_out'].shape[1]].squeeze(-2).view(
                k_iwae, batch_size, num_vars, -1).permute(0, 1, 3, 2)

        if self.args.value_norm == True:
            pred, _, _ = self.value_norm(
                pred, mask=batch['mask_out'].unsqueeze(0), mode='denorm')

        results = {"pred": pred.mean(dim=0)}

        likelihood = compute_log_normal_pdf(
            batch['data_out'].unsqueeze(0), batch['mask_out'].unsqueeze(0), pred, self.args)
        # sum out the traj dim
        loss_forecast = -torch.logsumexp(likelihood, 0)
        # mean out the batch dim
        loss_forecast = torch.mean(loss_forecast, dim=0)

        if self.args.patch_module == "ivp":
            loss = loss_forecast + loss_input * self.args.ratio_il
        else:
            loss = loss_forecast

        assert not (torch.isnan(loss).any() or torch.isinf(loss).any())

        results["loss"] = loss

        if self.args.patch_seg == "random":
            self.patch_info[patch_len]["cnt"] += 1
            self.patch_info[patch_len]["loss_accu"] += loss

        return results

    def prepare_validation(self, logger):
        if self.args.patch_seg == "random":
            loss_all = []
            sampled_keys = []

            # First pass: calculate loss for sampled patches; for unsampled ones, assign zero weight.
            for key, pinfo in self.patch_info.items():
                if pinfo["cnt"] > 0:
                    pinfo["loss"] = pinfo["loss_accu"] / pinfo["cnt"]
                    loss_all.append(pinfo["loss"])
                    sampled_keys.append(key)
                    device = pinfo["loss"].device
                else:
                    # Unsampled patches get zero weight.
                    pinfo["weight"] = torch.tensor(0.0)

            # Outlier pruning: remove the patch length with a high z-score if it exceeds the threshold.
            if len(loss_all) > 1:
                loss_tensor = torch.stack(loss_all)
                loss_mean = loss_tensor.mean()
                loss_std = loss_tensor.std()
                z_score = (loss_tensor - loss_mean) / loss_std
                threshold = 1.3
                max_z_index = torch.argmax(z_score).item()
                max_z_value = z_score[max_z_index].item()
                if max_z_value > threshold:
                    key_to_remove = sampled_keys[max_z_index]
                    logger.info(
                        f"Patch length {key_to_remove} has a high z-score ({max_z_value:.2f}). Removing it from patch_info.")
                    del self.patch_info[key_to_remove]

            # Second pass: compute weights for the remaining patches.
            weight_sum = 0.0
            for key, pinfo in self.patch_info.items():
                if pinfo.get("cnt") > 0:
                    pinfo["weight"] = 1 / (pinfo["loss"] ** 2)
                    weight_sum += pinfo["weight"]
                else:
                    pinfo["weight"] = pinfo["weight"].to(device)
                # Unsampled patches remain with weight = 0.0

            # Normalize weights for patches that were sampled.
            if weight_sum > 0:
                for key, pinfo in self.patch_info.items():
                    if pinfo.get("cnt", 0) > 0:
                        pinfo["weight"] = pinfo["weight"] / weight_sum

            logger.info(
                f"pinfos lengths: {[k for k in self.patch_info.keys()]}")
            logger.info(
                f"pinfos weights: {[pinfo['weight'].item() for pinfo in self.patch_info.values()]}")

            # Reset accumulators for the next cycle.
            for key, pinfo in self.patch_info.items():
                pinfo["loss_accu"] = 0.0
                pinfo["cnt"] = 0

    def run_validation(self, batch):

        if self.args.patch_seg == "random":
            if self.args.patch_len_vali == "best":
                # Choose the patch length (key of self.patch_info) with the lowest loss
                patch_len_best = max(
                    self.patch_info, key=lambda x: self.patch_info[x]["weight"])
                return self.forecast(batch, self.args.k_iwae, patch_len=patch_len_best)

            else:
                results_list = []
                weights = []
                for k, v in self.patch_info.items():
                    results = self.forecast(
                        batch, self.args.k_iwae, patch_len=k)
                    results_list.append(results)
                    weights.append(v["weight"])

                # Integrate the results of all patch lengths by averaging
                if self.args.patch_len_vali == "average":
                    # Average the results of all patch lengths
                    results = {}
                    results["pred"] = torch.stack(
                        [re["pred"] for re in results_list], dim=0).mean(dim=0)
                    results["loss"] = torch.stack(
                        [re["loss"] for re in results_list], dim=0).mean(dim=0)
                    return results

                # Integrate the results of all patch lengths by weighted averaging
                elif self.args.patch_len_vali == "weighted_average":
                    # Weighted average of the results of all patch lengths
                    weights = torch.stack(weights)
                    results = {}
                    results["pred"] = torch.stack(
                        [re["pred"] for re in results_list], dim=0).mul(weights.view(-1, 1, 1, 1)).sum(dim=0)
                    results["loss"] = torch.stack(
                        [re["loss"] for re in results_list], dim=0).mul(weights).sum(dim=0)
                    return results
        else:
            return self.forecast(batch, self.args.k_iwae)
