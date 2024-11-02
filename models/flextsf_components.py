import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

import models.model_utils as model_utils
from models.ivp_solvers.flow import CouplingFlow, ResNetFlow
from models.ivp_solvers.gru import GRUFlow
from models.ivp_solvers.ode import ODEModel


class TimeNorm(nn.Module):
    def __init__(self, affine=True):
        super(TimeNorm, self).__init__()
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, times, mask=None):
        if mask is not None:
            times = self.__normalizewM(times, mask)
        else:
            times = self.__normalize(times)
        return times, self.unit

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(1))
        self.affine_bias = nn.Parameter(torch.zeros(1))

    def __normalize(self, times):
        # The beginning of the timestamps has to be zero
        assert torch.all(times[:, 0] == 0)
        # Calculate the smallest time interval based available timestamps
        intervals = (times[:, 1:] - times[:, :-1])

        # create a mask for non-zero interval values
        mask_int = intervals > 0
        intervals_tmp = torch.where(mask_int, intervals, torch.max(
            intervals, dim=-1).values.unsqueeze(-1))

        self.unit = torch.min(intervals_tmp, dim=-1).values
        # normalize the time interval using the smallest interval
        intervals = intervals / self.unit.unsqueeze(-1)
        # Get the new timestamps
        times = torch.concat([times[:, 0].unsqueeze(-1),
                              torch.cumsum(intervals, dim=-1)], dim=-1)

        if self.affine:
            times = times * self.affine_weight
            times = times + self.affine_bias

        return times

    def __normalizewM(self, times, mask):
        # The beginning of the timestamps has to be zero
        assert torch.all(times[:, 0] == 0)
        # Calculate the smallest time interval based available timestamps
        intervals = times[:, 1:] - times[:, :-1]

        # create a mask for non-zero interval values
        mask_int = intervals > 0
        # replacing the excluded values with some value that will get ignored in following steps
        intervals_tmp = torch.where(mask_int, intervals, torch.max(
            intervals, dim=-1).values.unsqueeze(-1))

        self.unit = torch.min(intervals_tmp, dim=-1).values

        # normalize the time interval using the smallest interval
        intervals = intervals / self.unit.unsqueeze(-1)
        # Get the new timestamps
        times = torch.concat([times[:, 0].unsqueeze(-1),
                              torch.cumsum(intervals, dim=-1)], dim=-1)

        if self.affine:
            times = times * self.affine_weight
            times = times + self.affine_bias

        return times * mask

    def normalize(self, times):
        # This part doesn't need a mask, because the final loss is calculated
        # using the available observations
        time_zero = torch.zeros_like(times[:, 0:1])
        time_tmp = torch.cat([time_zero, times], dim=-1)
        intervals = (time_tmp[:, 1:] - time_tmp[:, :-1])
        intervals = intervals / self.unit.unsqueeze(-1)
        # Get the new timestamps
        times = torch.cumsum(intervals, dim=-1)
        return times


class ValueNorm(nn.Module):
    def __init__(self, num_features: int, eps=1e-8, affine=True, subtract_last=False):
        super(ValueNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mask, mode: str):
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == 'denorm':
            x = self._denormalize(x, mask)
        else:
            raise NotImplementedError
        return x, self.mean, self.stdev

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.sum(x * mask, dim=dim2reduce, keepdim=True) / \
                (torch.sum(mask, dim=dim2reduce, keepdim=True) + self.eps)
        self.stdev = torch.sqrt((torch.sum((x - self.mean) ** 2 * mask, dim=dim2reduce,
                                keepdim=True) / (torch.sum(mask, dim=dim2reduce, keepdim=True) + self.eps)) + self.eps)

    def _normalize(self, x, mask):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / (self.stdev + self.eps)

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x * mask

    def _denormalize(self, x, mask):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x * mask


# Refering to the learnable time embedding of the mTAN model
# https://github.com/reml-lab/mTAN/tree/main
class TimeEmbedding(nn.Module):
    def __init__(self, embed_time):
        super().__init__()
        self.periodic = nn.Linear(1, embed_time-1)
        self.linear = nn.Linear(1, 1)

    def forward(self, tt):
        tt = torch.log(tt + 1e-5)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)


# Time embedding adopted from the mTAN model
class InputEmbedding(nn.Module):
    def __init__(self, dim_data_in, dim_time_emb, dim_out):
        super().__init__()
        self.value_embedding = nn.Linear(dim_data_in, dim_out)
        self.time_embedding = TimeEmbedding(dim_time_emb)
        self.add_time_lyr = nn.Linear(dim_time_emb, dim_out)

    def forward(self, x, time):
        x = self.value_embedding(
            x) + self.add_time_lyr(self.time_embedding(time))
        return x


class SolverWrapper(nn.Module):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def forward(self, x, t):
        assert len(x.shape) - len(t.shape) == 1
        t = t.unsqueeze(-1)
        if t.shape[-3] == 1:
            t = t.repeat_interleave(x.shape[-3], dim=-3)
        if len(x.shape) == 4 and x.shape[0] != t.shape[0]:
            t = t.repeat_interleave(x.shape[0], dim=0)
        y = self.solver(x, t)
        return y


def build_ivp_solver(args):
    ivp_solver = None
    hidden_dims = [args.dim_ivp_hidden] * args.hidden_layers
    if args.ivp_solver == 'ode':
        ivp_solver = SolverWrapper(ODEModel(args.dim_patch_ts, args.odenet, hidden_dims, args.activation,
                                            args.final_activation, args.ode_solver, args.solver_step, args.atol, args.rtol))
    else:
        if args.ivp_solver == 'couplingflow':
            flow = CouplingFlow
        elif args.ivp_solver == 'resnetflow':
            flow = ResNetFlow
        elif args.ivp_solver == 'gruflow':
            flow = GRUFlow
        else:
            raise NotImplementedError

        ivp_solver = SolverWrapper(flow(
            args.dim_patch_ts, args.flow_layers, hidden_dims, args.time_net, args.time_hidden_dim))
    return ivp_solver


class IVP_Patcher(nn.Module):
    def __init__(self, args, process):
        super(IVP_Patcher, self).__init__()
        self.args = args
        self.process = process
        if process == "in":
            self.z_mapper_in = nn.Linear(
                1, args.dim_patch_ts, bias=False)
            self.ivp_solver_in = build_ivp_solver(args)
            self.z2mu_mapper = Z_to_mu(args.dim_patch_ts)
            self.z2std_mapper = Z_to_std(args.dim_patch_ts)
            self.register_buffer('mu', torch.tensor([args.prior_mu]))
            self.register_buffer('std', torch.tensor([args.prior_std]))
        else:
            self.z_mapper_out = nn.Linear(
                args.dim_patch_ts, 1, bias=False)
            self.ivp_solver_out = build_ivp_solver(args)

    def forward(self, x, t, m=None, k_iwae=1, padding=False):
        others = {}
        if self.process == "in":
            temp_states_initial = self.z_mapper_in(x.unsqueeze(-1))
            temp_states_evolved = self.ivp_solver_in(temp_states_initial, -t)

            if padding is False:
                z0_mean = self.z2mu_mapper(temp_states_evolved)
                z0_std = self.z2std_mapper(temp_states_evolved) + 1e-8
                # KL Divergence Loss
                fp_distr = Normal(z0_mean, z0_std)
                kldiv_z0_all = kl_divergence(
                    fp_distr, torch.distributions.Normal(self.mu, self.std))
                assert not (torch.isinf(kldiv_z0_all).any() |
                            torch.isnan(kldiv_z0_all).any())

                # Sample from the inferred distribution
                z0_mean_iwae = z0_mean.repeat(
                    k_iwae, *([1] * len(z0_mean.shape)))
                z0_std_iwae = z0_std.repeat(k_iwae, *([1] * len(z0_std.shape)))
                initial_state = model_utils.sample_standard_gaussian(
                    z0_mean_iwae, z0_std_iwae)

                # Calculate the mean of the evolved states using the mask
                mask_in_tmp = m.view(1, *m.shape, 1)

                # Integrate inference results
                if self.args.combine_methods == "average":
                    input_states = torch.sum(
                        initial_state * mask_in_tmp, dim=-2) / (mask_in_tmp.sum(dim=-2) + 1e-8)
                elif self.args.combine_methods == "kl_weighted":
                    kl_r = kldiv_z0_all
                    kl_w = kl_r / (torch.sum(kl_r * mask_in_tmp,
                                             dim=-2, keepdim=True) + 1e-8)
                    kl_w = kl_w * mask_in_tmp
                    input_states = torch.sum(initial_state * kl_w, dim=-2)
                else:
                    raise NotImplementedError
                others["kldiv_z0_all"] = kldiv_z0_all
            else:
                # If the patch is a dummy patch
                input_states = temp_states_evolved.repeat(
                    k_iwae, *([1] * len(temp_states_evolved.shape))).mean(dim=-2)

            out = input_states
        else:
            temp_states_evolved = self.ivp_solver_out(x.unsqueeze(-2), t)
            out = self.z_mapper_out(temp_states_evolved).squeeze(-1)

        return out, others


class Patcher(nn.Module):
    def __init__(self, args):
        super(Patcher, self).__init__()
        self.args = args
        if args.patch_module == "ivp":
            self.patch_in = IVP_Patcher(args, "in")
            self.patch_out = IVP_Patcher(args, "out")
        elif args.patch_module == "none":
            self.patch_in = nn.Linear(1, args.dim_patch_ts)
            self.patch_out = nn.Linear(args.dim_patch_ts, 1)
        else:
            raise NotImplementedError

    def encode(self, x, t, m, k_iwae, padding=False):
        if self.args.patch_module == "none":
            x = self.patch_in(x)
            x = x.unsqueeze(0).repeat(k_iwae, *([1] * len(x.shape)))
            others = {}
        else:
            x, others = self.patch_in(x, t, m, k_iwae, padding)
        return x, others

    def decode(self, x, t):
        if self.args.patch_module == "none":
            x = self.patch_out(x)
            x = (x, {})
        else:
            x = self.patch_out(x, t)
        return x


class Z_to_mu(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),)

    def forward(self, data):
        return self.net(data)


class Z_to_std(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Softplus(),)

    def forward(self, data):
        return self.net(data)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Apply the RMSNorm normalization to the input tensor.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * \
            ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AttentionBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        """
        Initialize a AttentionBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args: Model configuration parameters.

        Attributes:
            nhead (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
            wq: Linear transformation for queries.
            wk: Linear transformation for keys.
            wv: Linear transformation for values.
            wo: Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.nhead = args.nhead
        self.dim = args.dim_attn_internal
        self.head_dim = args.dim_attn_internal // args.nhead
        self.feed_forward = FeedForward(
            dim=args.dim_attn_internal,
            hidden_dim=4 * args.dim_attn_internal,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=None
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(
            args.dim_attn_internal, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim_attn_internal, eps=args.norm_eps)

        self.args = args
        self.wq = nn.Linear(
            args.dim_patch_ts,
            args.nhead * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim_patch_ts,
            self.nhead * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim_patch_ts,
            self.nhead * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.nhead * self.head_dim,
            args.dim_patch_ts,
            bias=False,
        )

    def forward(
        self,
        hidden_states_in: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        x = self.attention_norm(hidden_states_in)

        # Start the scaled dot-product attention with causal masking
        # Cache calculated keys and values
        if start_pos == 0:
            self.cache_k = torch.zeros(
                (
                    x.shape[0],
                    self.args.max_seq_len,
                    self.nhead,
                    self.head_dim,
                )
            ).to(x.device)
            self.cache_v = torch.zeros(
                (
                    x.shape[0],
                    self.args.max_seq_len,
                    self.nhead,
                    self.head_dim,
                )
            ).to(x.device)

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.nhead, self.head_dim)
        xk = xk.view(bsz, seqlen, self.nhead, self.head_dim)
        xv = xv.view(bsz, seqlen, self.nhead, self.head_dim)

        # Apply the layer-wise time embedding
        # Refering to the rotary position embedding
        if self.args.lyr_time_embed == True:
            xq_ = torch.view_as_complex(
                xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(
                xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(-2)
            xq_new = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_new = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            xq = xq_new.type_as(xq)
            xk = xk_new.type_as(xk)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        # Ignore the last token for caching, which is the dummy patch
        self.cache_k[:bsz, start_pos: start_pos + seqlen - 1] = xk[:bsz, :-1]
        self.cache_v[:bsz, start_pos: start_pos + seqlen - 1] = xv[:bsz, :-1]
        keys = torch.cat(
            [self.cache_k[:bsz, : start_pos + seqlen - 1], xk[:bsz, -1:]], dim=1)
        values = torch.cat(
            [self.cache_v[:bsz, : start_pos + seqlen - 1], xv[:bsz, -1:]], dim=1)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # (bs, n_local_heads, cache_len + seqlen, head_dim)
        keys = keys.transpose(1, 2)
        # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / \
            math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask.unsqueeze(1)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        hidden_states = torch.matmul(scores, values)
        hidden_states = hidden_states.transpose(
            1, 2).contiguous().view(bsz, seqlen, -1)
        hidden_states = self.wo(hidden_states)

        # Apply the residual connection after attention
        hidden_states = hidden_states_in + hidden_states

        # Apply feedforward layer and residual connection
        hidden_states_out = hidden_states + \
            self.feed_forward.forward(self.ffn_norm(hidden_states))

        return hidden_states_out


class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_layers = args.attn_layers
        if args.lyr_time_embed == False:
            self.states_in_lyr = InputEmbedding(
                args.dim_patch_ts, args.embed_time, args.dim_attn_internal)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(AttentionBlock(layer_id, args))

        self.norm = RMSNorm(args.dim_attn_internal, eps=args.norm_eps)

    def forward(self, hidden_states: torch.Tensor, timestamps: torch.Tensor, exist_input: torch.Tensor, start_pos=0):

        seqlen = hidden_states.shape[1]

        # Apply the layer-wise time embedding
        # Refering to the rotary position embedding
        if self.args.lyr_time_embed == True:
            head_dim = self.args.dim_attn_internal // self.args.nhead
            freqs = 1.0 / (self.args.freqs_theta ** (torch.arange(0, head_dim, 2)
                                                     [: (head_dim // 2)].float() / head_dim))
            freqs = freqs.to(timestamps.device)
            freqs = einsum('...i,...j->...ij', timestamps, freqs).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        else:
            hidden_states = self.states_in_lyr(
                hidden_states, timestamps.unsqueeze(-1))
            freqs_cis = None

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float(
                '-inf'), device=hidden_states.device)
            mask = torch.triu(mask, diagonal=1)

            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=hidden_states.device),
                mask
            ]).type_as(hidden_states)

            if mask.shape[0] == exist_input.shape[1]:
                identity = torch.eye(
                    mask.shape[0], dtype=torch.bool, device=hidden_states.device).unsqueeze(0)
                exist_input = exist_input.unsqueeze(-2)  # (N, 1, seqlen)
                # Turn the diagonal elements to True to avoid NaN in the softmax
                # Broadcasting applies here. (N, seqlen, seqlen)
                exist_input = torch.logical_or(identity, exist_input)
                # exist_input_transformed: don't take empty patches into account in the attention
                exist_input_transformed = torch.where(
                    exist_input, 0.0, float('-inf'))
                mask = mask.unsqueeze(0).expand(exist_input.shape[0], -1, -1)
                mask = mask + exist_input_transformed
            else:
                mask = mask.unsqueeze(0).expand(exist_input.shape[0], -1, -1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, start_pos, freqs_cis, mask)
        hidden_states = self.norm(hidden_states)

        return hidden_states
