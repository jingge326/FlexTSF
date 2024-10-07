import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Independent


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def gaussian_log_likelihood(mu_2d, data_2d, prior_std, indices=None):
    n_data_points = mu_2d.size()[-1]

    if n_data_points > 0:
        gaussian = Independent(
            Normal(loc=mu_2d, scale=prior_std.repeat(n_data_points)), 1)
        log_prob = gaussian.log_prob(data_2d)
        log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros([1]).to(data_2d).squeeze()
    return log_prob


def compute_binary_CE_loss(label_predictions, mortality_label):
    # print('Computing binary classification loss: compute_CE_loss')

    mortality_label = mortality_label.reshape(-1)

    if len(label_predictions.size()) == 1:
        label_predictions = label_predictions.unsqueeze(0)

    n_traj = label_predictions.size(0)
    label_predictions = label_predictions.reshape(n_traj, -1)

    idx_not_nan = ~torch.isnan(mortality_label)
    if len(idx_not_nan) == 0.0:
        print("All are labels are NaNs!")
        ce_loss = torch.Tensor(0.0).to(mortality_label)
    label_predictions = label_predictions[:, idx_not_nan]

    mortality_label = mortality_label[idx_not_nan]

    if torch.sum(mortality_label == 0.0) == 0 or torch.sum(mortality_label == 1.0) == 0:
        print(
            "Warning: all examples in a batch belong to the same class -- please increase the batch size."
        )

    assert not torch.isnan(label_predictions).any()
    assert not torch.isnan(mortality_label).any()

    # For each trajectory, we get n_traj samples from z0 -- compute loss on all of them
    mortality_label = mortality_label.repeat(n_traj, 1)
    ce_loss = nn.BCEWithLogitsLoss()(label_predictions, mortality_label)

    # divide by number of patients in a batch
    ce_loss = ce_loss / n_traj
    return ce_loss


def compute_masked_likelihood(mu, data, mask, likelihood_func):
    # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
    n_traj, n_samples, n_timepoints, n_dims = data.size()

    res = []
    for i in range(n_traj):
        for k in range(n_samples):
            for j in range(n_dims):
                data_masked = torch.masked_select(
                    data[i, k, :, j], mask[i, k, :, j].bool())
                mu_masked = torch.masked_select(
                    mu[i, k, :, j], mask[i, k, :, j].bool())
                log_prob = likelihood_func(
                    mu_masked, data_masked, indices=(i, k, j))
                res.append(log_prob)
    # shape: [n_samples*n_traj, 1]

    res = torch.stack(res, 0).to(data)
    res = res.reshape((n_traj, n_samples, n_dims))
    # Take mean over the number of dimensions
    res = torch.mean(res, -1)  # !!!!!!!!!!! changed from sum to mean
    # res = res.transpose(0, 1)
    return res


def likelihood_data_mask(mu, data, prior_std, pred_m, true_m, mask_loss_ratio):
    # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
    n_traj, n_samples, _, n_dims = data.size()

    res = []
    for i in range(n_traj):
        for k in range(n_samples):
            for j in range(n_dims):

                data_masked = torch.masked_select(
                    data[i, k, :, j], true_m[i, k, :, j].bool())
                mu_masked = torch.masked_select(
                    mu[i, k, :, j], true_m[i, k, :, j].bool())

                log_prob_value = gaussian_log_likelihood(
                    mu_masked, data_masked, prior_std)

                # maximizing the likelihood with respect to the parameters is the same as minimizing the cross-entropy
                log_prob_mask = torch.neg(
                    F.binary_cross_entropy_with_logits(pred_m, true_m))

                log_prob = log_prob_value + mask_loss_ratio*log_prob_mask
                res.append(log_prob)

    res = torch.stack(res, 0).to(data)    # shape: [n_samples*n_traj, 1]
    res = res.reshape((n_traj, n_samples, n_dims))
    # Take mean over the number of feature dimensions
    res = torch.mean(res, -1)
    return res


def masked_gaussian_log_density(mu, data, prior_std, mask=None):
    # these cases are for plotting through plot_estim_density
    if len(mu.size()) == 3:
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)

    if len(data.size()) == 2:
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif len(data.size()) == 3:
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj, n_samples, n_timepoints, n_dims = mu.size()

    assert data.size()[-1] == n_dims

    # Shape after permutation: [n_samples, n_traj, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj * n_samples, n_timepoints * n_dims)
        n_traj, n_samples, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(
            n_traj * n_samples, n_timepoints * n_dims)

        res = gaussian_log_likelihood(mu_flat, data_flat, prior_std)
        res = res.reshape(n_traj, n_samples).transpose(0, 1)
    else:
        # Compute the likelihood per patient so that we don't priorize patients with more measurements
        def func(mu, data, indices):
            return gaussian_log_likelihood(mu, data, prior_std=prior_std, indices=indices)
        res = compute_masked_likelihood(mu, data, mask, func)
    return res


def mse(mu, data, indices=None):
    n_data_points = mu.size()[-1]
    if n_data_points > 0:
        mse = nn.MSELoss()(mu, data)
    else:
        mse = torch.zeros([1]).to(data).squeeze()
    return mse


def compute_mse(mu, data, mask=None):
    # these cases are for plotting through plot_estim_density
    if len(mu.size()) == 3:
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)

    if len(data.size()) == 2:
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif len(data.size()) == 3:
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj, n_samples, n_timepoints, n_dims = mu.size()
    assert data.size()[-1] == n_dims

    # Shape after permutation: [n_samples, n_traj, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj * n_samples, n_timepoints * n_dims)
        n_traj, n_samples, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(
            n_traj * n_samples, n_timepoints * n_dims)
        res = mse(mu_flat, data_flat)
    else:
        # Compute the likelihood per patient so that we don't priorize patients with more measurements
        res = compute_masked_likelihood(mu, data, mask, mse)
    return res


def compute_multiclass_CE_loss(label_predictions, true_label, mask):

    if len(label_predictions.size()) == 3:
        label_predictions = label_predictions.unsqueeze(0)

    n_traj, n_samples, n_tp, n_dims = label_predictions.size()

    # assert(not torch.isnan(label_predictions).any())
    # assert(not torch.isnan(true_label).any())

    # For each trajectory, we get n_traj samples from z0 -- compute loss on all of them
    true_label = true_label.repeat(n_traj, 1, 1)

    label_predictions = label_predictions.reshape(
        n_traj * n_samples * n_tp, n_dims
    )
    true_label = true_label.reshape(n_traj * n_samples * n_tp, n_dims)

    # choose time points with at least one measurement
    mask = torch.sum(mask, -1) > 0

    # repeat the mask for each label to mark that the label for this time point is present
    pred_mask = mask.repeat(n_dims, 1, 1).permute(1, 2, 0)

    label_mask = mask
    pred_mask = pred_mask.repeat(n_traj, 1, 1, 1)
    label_mask = label_mask.repeat(n_traj, 1, 1, 1)

    pred_mask = pred_mask.reshape(n_traj * n_samples * n_tp, n_dims)
    label_mask = label_mask.reshape(n_traj * n_samples * n_tp, 1)

    if (label_predictions.size(-1) > 1) and (true_label.size(-1) > 1):
        assert label_predictions.size(-1) == true_label.size(-1)
        # gen_mean are in one-hot encoding -- convert to indices
        _, true_label = true_label.max(-1)

    res = []
    for i in range(true_label.size(0)):
        pred_masked = torch.masked_select(
            label_predictions[i], pred_mask[i].bool())
        labels = torch.masked_select(true_label[i], label_mask[i].bool())

        pred_masked = pred_masked.reshape(-1, n_dims)

        if len(labels) == 0:
            continue

        ce_loss = nn.CrossEntropyLoss()(pred_masked, labels.long())
        res.append(ce_loss)

    ce_loss = torch.stack(res, 0).to(label_predictions)
    ce_loss = torch.mean(ce_loss)
    # # divide by number of patients in a batch
    # ce_loss = ce_loss / n_traj
    return ce_loss


def mean_squared_error(orig, pred, mask=None, mask_select=None):
    error = (orig - pred) ** 2
    if mask is not None:
        if mask_select is not None:
            mask = mask * mask_select
        error = error * mask
        mse = error.sum() / (mask.sum() + 1e-8)
    else:
        mse = error.mean()
    assert mse >= 0
    return mse


def mean_absolute_error(orig, pred, mask=None, mask_select=None):
    # if orig, pred or mask is Pytorch Tensor, convert it to numpy Array
    if isinstance(orig, torch.Tensor):
        orig = orig.cpu().detach().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()

    error = np.absolute(orig - pred)
    if mask is not None:
        if mask_select is not None:
            mask = mask * mask_select
        error = error * mask
        return error.sum() / mask.sum()
    else:
        return error.mean()


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    pdf = -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask
    return pdf


def compute_log_normal_pdf(observed_data, observed_mask, pred_x, args):
    obsrv_std = torch.zeros(pred_x.size()).to(
        observed_data.device) + args.obsrv_std
    noise_logvar = 2. * torch.log(obsrv_std).to(observed_data.device)
    pdf = log_normal_pdf(observed_data, pred_x, noise_logvar, observed_mask)
    logpx = pdf.sum(-1).sum(-1)
    logpx = logpx / (observed_mask.sum(-1).sum(-1) + 1e-8)
    return logpx


def log_lik_gaussian_simple(x, mu, logvar):
    """
    Return loglikelihood of x in gaussian specified by mu and logvar, taken from
    https://github.com/edebrouwer/gru_ode_bayes
    """
    return np.log(np.sqrt(2 * np.pi)) + (logvar / 2) + ((x - mu).pow(2) / (2 * logvar.exp()))


def GaussianNegLogLik(truth, gen_mean, gen_variance, mask=None):
    """ Computes Gaussian Negaitve Loglikelihood """
    assert truth.shape == gen_mean.shape == gen_variance.shape, f'truth {truth.shape} gen_mean {gen_mean.shape} gen_variance {gen_variance.shape}'

    # add epsilon to variance to avoid numerical instability
    epsilon = 1e-6 * torch.ones_like(truth)
    gen_variance = torch.maximum(gen_variance, epsilon)

    if mask == None:
        mask = torch.ones_like(truth)

    # sum over dimensions
    const = np.log(2 * np.pi)
    loglik_point_wise = 0.5 * \
        (torch.log(gen_variance) + torch.square(truth -
         gen_mean) / gen_variance + const) * mask
    loglik_loss = torch.sum(loglik_point_wise) / torch.sum(mask)

    return loglik_loss


def GaussianNegLogLik_Logvar(truth, gen_mean, gen_logvar, mask=None):
    # add epsilon to variance to avoid numerical instability
    # log(varicance) = -13.81551 => varicance = 1e-6
    epsilon = -13.81551 * torch.ones_like(gen_logvar)
    gen_logvar = torch.maximum(gen_logvar, epsilon)
    const = np.log(2 * np.pi)
    loglik_point_wise = .5 * \
        (const + gen_logvar + (truth - gen_mean)
         ** 2. / torch.exp(gen_logvar)) * mask
    loglik_loss = torch.sum(loglik_point_wise) / torch.sum(mask)

    return loglik_loss


# Some folders were listed in the .gitignore file because they contain either too many files
# or very large files. We need to create them before running the experiments.
def check_and_create_folders(proj_path):
    directories = [
        "log",
        "results",
        "results/pl_checkpoint",
        "results/forecasts",
        "results/model_para",
        "results/model_hyper"
    ]
    for directory in directories:
        directory = proj_path/directory
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
