import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.utils_exp import compute_log_normal_pdf
from models.flextsf_components import Patcher, TransformerDecoder, ValueNorm, TimeNorm


class FlexTSF_General_Forecast(TransformerDecoder):
    def __init__(self, args):
        super().__init__(args)
        self.patcher = Patcher(args)
        if self.args.vt_norm == True:
            self.value_norm = ValueNorm(
                num_features=1, affine=True, subtract_last=False)
            self.time_norm = TimeNorm(affine=self.args.time_norm_affine)

        if args.leader_node == True:
            self.leaderlyr = nn.Sequential(
                nn.Linear(6, 128),
                nn.Tanh(),
                nn.Linear(128, args.dim_patch_ts))

    def forecast(self, batch, k_iwae=1):
        # The beginning of time_in is 0 and the first element of time_out is bigger than the last element of time_in
        assert (batch["time_in"][..., 0] == 0).all() and (
            batch["time_out"][..., 0] >= batch["time_in"].max(dim=-1)[0]).all()

        ###### VT-Norm ######
        if self.args.vt_norm == True:
            # Normalize values
            data_in, ins_mean, ins_std = self.value_norm(
                batch['data_in'], mask=batch['mask_in'], mode='norm')
            # Normalize times
            exist_time_in = batch["mask_in"].any(dim=-1)
            time_in, t_unit_inst = self.time_norm(
                batch['time_in'], mask=exist_time_in)
            # also normalize time_out using the same t_unit_inst
            time_out = self.time_norm.normalize(batch["time_out"])

            if self.args.leader_node == True:
                # static features from the value normalization
                ins_mean = ins_mean.view(-1, 1)
                ins_std = ins_std.view(-1, 1)
                if len(batch["gmean"].shape) == 1:
                    gmean = batch["gmean"].repeat(
                        batch["data_in"].size(0), 1).view(-1, 1)
                    gstd = batch["gstd"].repeat(
                        batch["data_in"].size(0), 1).view(-1, 1)
                else:
                    gmean = batch["gmean"]
                    gstd = batch["gstd"]
                # static features from the time normalization
                t_unit_inst = t_unit_inst.repeat_interleave(
                    batch['data_in'].size(-1)).view(-1, 1)
                if len(batch["time_unit"].shape) == 0:
                    t_unit_global = batch["time_unit"].repeat_interleave(
                        t_unit_inst.size(0)).view(-1, 1)
                else:
                    t_unit_global = batch["time_unit"].view(-1, 1)
                # concatenate all the static features
                stat = torch.cat([gmean, gstd, ins_mean, ins_std, t_unit_inst, t_unit_global],
                                 dim=-1).unsqueeze(1)
                batch['stat'] = stat

        else:
            data_in = batch['data_in']
            time_in = batch["time_in"]
            time_out = batch["time_out"]

        time_in = time_in.repeat_interleave(batch['data_in'].size(-1), dim=0)
        time_out = time_out.repeat_interleave(
            batch['data_out'].size(-1), dim=0)

        ###### Patching ######
        # Channel independent. Time dimension at the end
        data_in = data_in.permute(0, 2, 1)
        mask_in = batch["mask_in"].permute(0, 2, 1)
        exist_in = batch["exist_in"]
        gen_seq = torch.zeros_like(batch['data_out']).permute(0, 2, 1)
        mask_out = batch["mask_out"].permute(0, 2, 1)

        stride = batch["patch_len"] - int(
            batch["patch_len"] * self.args.patch_overlap_rate)

        # Pad the last dim to make sure the last patch is complete
        data_in = F.pad(
            data_in, (0, stride), mode="replicate")
        gen_seq = F.pad(
            gen_seq, (0, stride), mode="replicate")
        mask_in = F.pad(
            mask_in, (0, stride), mode="constant", value=0)
        exist_in = F.pad(
            exist_in, (0, stride), mode="constant", value=0)
        mask_out = F.pad(
            mask_out, (0, stride), mode="constant", value=0)
        time_in = F.pad(time_in, (0, stride), mode="replicate")
        time_out = F.pad(time_out, (0, stride), mode="replicate")

        # Splitting the sequence to get patch segments
        data_in = data_in.unfold(
            dimension=-1, size=batch["patch_len"], step=stride)
        gen_seq = gen_seq.unfold(
            dimension=-1, size=batch["patch_len"], step=stride)
        mask_in = mask_in.unfold(
            dimension=-1, size=batch["patch_len"], step=stride)
        exist_in = exist_in.unfold(
            dimension=-1, size=batch["patch_len"], step=stride)
        mask_out = mask_out.unfold(
            dimension=-1, size=batch["patch_len"], step=stride)
        time_in = time_in.unfold(
            dimension=-1, size=batch["patch_len"], step=stride)
        time_out = time_out.unfold(
            dimension=-1, size=batch["patch_len"], step=stride)

        # Reshape the data
        batch_size, num_vars, num_patches_in, patch_len = mask_in.shape
        _, _, num_patches_out, _ = mask_out.shape
        data_in = data_in.view(-1, num_patches_in, patch_len)
        mask_in = mask_in.view(-1, num_patches_in, patch_len)
        exist_in = exist_in.repeat_interleave(num_vars, dim=0).view(
            -1, num_patches_in, patch_len)
        mask_out_tmp = mask_out.view(-1, num_patches_out, patch_len)
        # Indicate which patch is empty.
        # Different time series samples or variables may have different lengths,
        # resulting in some empty patches in a batch of samples.
        # contain empty patches at the end of the sequence
        exist_seq_in = exist_in.any(dim=-1)
        # contain empty patches in between and at the end of the sequence
        exist_patch_in = mask_in.any(dim=-1)

        # Align the timestamps of each patch
        oid_in = time_in[:, :, 0]
        oid_out = time_out[:, :, 0]
        time_in = time_in - oid_in.unsqueeze(-1)
        time_out = time_out - oid_out.unsqueeze(-1)
        t = oid_in

        input_states, others = self.patcher.encode(
            data_in, time_in, mask_in, k_iwae)

        kl_loss_input = others["kldiv_z0_all"]

        # Adjust the shape of the data to match the number of samples in the generation part
        mask_in = mask_in.unsqueeze(0).repeat(k_iwae, 1, 1, 1)
        gen_seq = gen_seq.unsqueeze(0).repeat(k_iwae, 1, 1, 1, 1)
        t = t.repeat(k_iwae, 1, 1)
        oid_out = oid_out.repeat(k_iwae, 1, 1)
        time_out = time_out.repeat(k_iwae, 1, 1, 1)
        exist_patch_in = exist_patch_in.repeat(k_iwae, 1)
        exist_seq_in = exist_seq_in.unsqueeze(0).repeat(k_iwae, 1, 1)

        # Add the leader node
        if self.args.vt_norm == True and self.args.leader_node == True:
            batch["stat"] = batch["stat"].unsqueeze(0).repeat(k_iwae, 1, 1, 1)
            leader = self.leaderlyr(batch['stat'])
            input_states = torch.cat([leader, input_states], dim=-2)
            t = torch.cat([torch.ones_like(t[..., 0:1]) * -1, t], dim=-1)

        # A batch of time series may have different lengths, resulting in different numbers of non-empty patches.
        # exist_edge_patch_in indicates the location of the last non-empty patch.
        exist_last = torch.cat(
            [exist_seq_in[:, :, 1:], torch.zeros_like(exist_seq_in[:, :, 0:1])], dim=-1)
        exist_edge_patch_in = torch.logical_xor(exist_last, exist_seq_in)

        if (self.args.leader_node == True) and (self.args.dummy_patch == True):
            # Add two extra slides at the beginning, because we have the leader and padding
            exist_edge_patch_in = torch.cat([torch.zeros_like(
                exist_edge_patch_in[:, :, 0:2]).bool(), exist_edge_patch_in], dim=-1).unsqueeze(-1)
            exist_patch_in = torch.cat([torch.ones_like(
                exist_patch_in[:, 0:1]).bool(), exist_patch_in, torch.ones_like(exist_patch_in[:, -1:]).bool()], dim=-1)  # Add two extra slides at the beginning and end
            # make sure the position of dummy patch is True
            exist_patch_in = (
                exist_patch_in + exist_edge_patch_in.view(exist_patch_in.shape)) > 0
        elif (self.args.leader_node == False) and (self.args.dummy_patch == True):
            exist_edge_patch_in = torch.cat([torch.zeros_like(
                exist_edge_patch_in[:, :, 0:1]).bool(), exist_edge_patch_in], dim=-1).unsqueeze(-1)
            exist_patch_in = torch.cat([exist_patch_in, torch.ones_like(
                exist_patch_in[:, -1:]).bool()], dim=-1)
            exist_patch_in = (
                exist_patch_in + exist_edge_patch_in.view(exist_patch_in.shape)) > 0
        elif (self.args.leader_node == False) and (self.args.dummy_patch == False):
            exist_edge_patch_in = exist_edge_patch_in.unsqueeze(-1)
        else:
            raise ValueError(
                "Invalid combination of leader_node and dummy_patch")

        input_states_last = (input_states * exist_edge_patch_in[:, :, 1:, :]).sum(
            dim=-2, keepdim=True)

        if self.args.dummy_type == "clone":
            padded_states = input_states_last
        elif self.args.dummy_type == "detach":
            padded_states = input_states_last.detach()
        elif self.args.dummy_type == "zero":
            padded_states = torch.zeros_like(input_states_last)
        else:
            raise ValueError("Invalid dummy_type")

        edge_gen_process = torch.cat([torch.zeros_like(
            input_states[:, :, 0:1, :]), torch.ones_like(input_states[:, :, 0:1, :])], dim=-2).bool()

        kldiv_list = []
        prev_pos = 0
        # Auto-regressive generation
        for i in range(0, num_patches_out):
            if self.args.dummy_patch == True:

                hidden_states = torch.cat(
                    [input_states, padded_states], dim=-2)

                t = torch.cat([t, torch.zeros_like(t[..., 0:1])], dim=-1)
                # The last dim is the same, so just taking the first element is fine
                t = t + exist_edge_patch_in[..., 0] * oid_out[..., i:i+1]
            else:
                hidden_states = input_states

            dim0 = k_iwae * batch_size * num_vars

            hidden_states = hidden_states.view(dim0, *hidden_states.shape[-2:])
            t = t.view(dim0, *t.shape[-1:])

            hidden_states = self.forward(
                hidden_states, t, exist_patch_in, prev_pos)

            hidden_states = hidden_states.view(
                k_iwae, batch_size * num_vars, *hidden_states.shape[1:])
            t = t.view(k_iwae, batch_size * num_vars, *t.shape[1:])

            last_states = (hidden_states *
                           exist_edge_patch_in).sum(dim=-2, keepdim=True)

            output_value, _ = self.patcher.decode(
                last_states, time_out[..., i:i+1, :])

            # Mean out i_kwae samples
            output_value_in = output_value.mean(dim=0)

            # time_out is repeated by i_kwae, so we can just use the first one
            temp_states_evolved, others = self.patcher.encode(
                output_value_in, -time_out[0, :, i:i+1, :], torch.ones_like(mask_in[0, :, 0:1, :]), k_iwae)

            gen_seq[:, :, :, i, :] = output_value.view(
                k_iwae, batch_size, num_vars, patch_len)

            t = oid_out[:, :, i:i+1]
            prev_pos += hidden_states.shape[-2] - 1

            # In generation part, just using the last states is fine
            exist_edge_patch_in = edge_gen_process

            # Don't need to calculate the KL divergence and input_states if it's the last step
            if i < gen_seq.size(-2) - 1:
                input_states = temp_states_evolved

                if self.args.dummy_type == "clone":
                    padded_states = input_states
                elif self.args.dummy_type == "detach":
                    padded_states = input_states.detach()
                elif self.args.dummy_type == "zero":
                    padded_states = torch.zeros_like(input_states)
                else:
                    raise ValueError("Invalid dummy_type")

                if self.args.patch_module == "ivp":
                    kldiv_list.append(others["kldiv_z0_all"])

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

        if self.args.vt_norm == True:
            pred, _, _ = self.value_norm(
                pred, mask=batch['mask_out'].unsqueeze(0), mode='denorm')

        results = {"pred": pred.mean(dim=0)}

        if self.args.patch_module == "ivp":
            # Calculate the KL divergence loss
            # kl_mask_out version
            if len(kldiv_list) > 0:
                kldiv_both = torch.cat(
                    [kl_loss_input, torch.cat(kldiv_list, dim=1)], dim=1)
            else:
                kldiv_both = kl_loss_input

            # Reshape the data to reconstruct the variable dimension
            kldiv_both = kldiv_both.view(
                batch_size, num_vars, *kldiv_both.shape[1:])
            if self.args.kl_alles == True:
                kldiv_loss = kldiv_both.mean([1, 2, 3, 4])
            else:
                # ones_like: Don't expose the structure of output data to the model
                mask_kl = torch.cat(
                    [mask_in[0, ...], torch.ones_like(mask_out_tmp[..., :-1, :])], dim=-2)
                # Repeat mask_kl to match the shape of kldiv_both
                mask_kl = mask_kl.unsqueeze(-1).repeat(
                    *([1] * len(mask_kl.shape)), self.args.dim_patch_ts)
                mask_kl = mask_kl.view(
                    batch_size, num_vars, *mask_kl.shape[1:])
                kldiv_loss = (
                    kldiv_both * mask_kl).sum([1, 2, 3, 4])/(mask_kl.sum([1, 2, 3, 4]) + 1e-8)
        else:
            # kldiv_loss is zero
            kldiv_loss = torch.zeros(batch_size).to(pred.device)

        likelihood = compute_log_normal_pdf(
            batch['data_out'].unsqueeze(0), batch['mask_out'].unsqueeze(0), pred, self.args)
        # sum out the traj dim
        loss = -torch.logsumexp(likelihood -
                                self.args.kl_coef * kldiv_loss, 0)

        # mean over the batch
        loss = loss.mean()
        results["loss"] = loss

        return results

    def run_validation(self, batch):
        return self.forecast(batch, self.args.k_iwae)
