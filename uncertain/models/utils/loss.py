import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

kl = nn.KLDivLoss(reduction="batchmean")
softmax = nn.Softmax(dim=1)

def pairwise_loss(latent_size, X, X_recon, decoder):
    return torch.mean(F.pairwise_distance(X.detach(), X_recon))

def orthogonal_loss(latent_size, X, X_recon, decoder):
    return torch.norm(torch.eye(latent_size).cuda() - torch.mm(decoder.weight.t(), decoder.weight))

def kl_loss(latent_size, X, X_recon, encoder, decoder):
    posterior_mean, posterior_logvar = encoder[0], encoder[1]
    prior_var = torch.tensor(1.0).cuda()
    prior_mean = torch.tensor(0.0).cuda()
    prior_logvar = torch.tensor(0.0).cuda()
    prior_var = prior_var.expand_as(posterior_logvar)
    posterior_var = posterior_logvar.exp()
    var_division = posterior_var / prior_var
    diff = posterior_mean - prior_mean
    diff_term  = diff * diff / prior_var
    logvar_division = prior_logvar - posterior_logvar
    return torch.mean(0.5 * (var_division + diff_term + logvar_division).sum(1))

def orthogonal_pairwise_loss(latent_size, X, X_recon, decoder):
    pw = pairwise_loss(latent_size, X, X_recon, decoder)
    ort = orthogonal_loss(latent_size, X, X_recon, decoder)
    # write_loss2(X_recon, pw, ort)
    return pw + ort

def kl_pairwise_loss(latent_size, X, X_recon,  encoder, decoder):
    kl = mmd_loss(encoder, t=0.1, kernel='diffusion')
    pw = pairwise_loss(latent_size, X, X_recon, decoder)
    # write_loss2(X_recon, pw, kl)
    return kl+pw

def write_loss2(X_recon, pw, ort):
    if X_recon.requires_grad == True:
        with open('bert-cola-pw-mmd-loss.csv', 'a', newline='\n') as f_object:
            f_object.write(f"{pw},{ort},{pw+ort}\n")

def mmd_loss(y_fake, t=0.1, kernel='diffusion'):
    '''
    computes the mmd loss with information diffusion kernel
    :param x: batch_size x latent dimension
    :param y:
    :param t:
    :return:
    '''
    n,d = y_fake.shape
    y_true = np.random.dirichlet(np.ones(100) * 1e-1, size=n)
    y_true = torch.from_numpy(y_true).cuda()

    y_fake = softmax(y_fake).double()

    x = y_true
    y = y_fake

    eps = 1e-6
    n,d = x.shape
    if kernel == 'tv':
        sum_xx = nd.zeros(1, ctx=ctx_model)
        for i in range(n):
            for j in range(i+1, n):
                sum_xx = sum_xx + nd.norm(x[i] - x[j], ord=1)
        sum_xx = sum_xx / (n * (n-1))

        sum_yy = nd.zeros(1, ctx=ctx_model)
        for i in range(y.shape[0]):
            for j in range(i+1, y.shape[0]):
                sum_yy = sum_yy + nd.norm(y[i] - y[j], ord=1)
        sum_yy = sum_yy / (y.shape[0] * (y.shape[0]-1))

        sum_xy = nd.zeros(1, ctx=ctx_model)
        for i in range(n):
            for j in range(y.shape[0]):
                sum_xy = sum_xy + nd.norm(x[i] - y[j], ord=1)
        sum_yy = sum_yy / (n * y.shape[0])
    else:
        qx = torch.sqrt(torch.clip(x, eps, 1))
        qy = torch.sqrt(torch.clip(y, eps, 1))
        xx = torch.mm(qx, qx.T)
        yy = torch.mm(qy, qy.T)
        xy = torch.mm(qx, qy.T)

        def diffusion_kernel(a, tmpt, dim):
            # return (4 * np.pi * tmpt)**(-dim / 2) * nd.exp(- nd.square(nd.arccos(a)) / tmpt)
            return torch.exp(-torch.square(torch.arccos(a)) / tmpt)

        off_diag = 1 - torch.eye(n).cuda()
        k_xx = diffusion_kernel(torch.clip(xx, 0, 1-eps), t, d-1)
        k_yy = diffusion_kernel(torch.clip(yy, 0, 1-eps), t, d-1)
        k_xy = diffusion_kernel(torch.clip(xy, 0, 1-eps), t, d-1)
        sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
        sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
        sum_xy = 2 * k_xy.sum() / (n * n)
    return sum_xx + sum_yy - sum_xy

def kl_orthogonal_pairwise_loss(latent_size, X, X_recon, encoder, decoder):
    kl_lo = kl_loss(latent_size, X, X_recon, encoder, decoder)
    pw_lo = pairwise_loss(latent_size, X, X_recon, decoder)
    ort_lo = orthogonal_loss(latent_size, X, X_recon, decoder)
    return kl_lo + pw_lo + ort_lo

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss. Adapted from https://bit.ly/2T6kfz7. If 0 < smoothing < 1,
    this smoothes the standard cross-entropy loss.
    """

    def __init__(self, n_classes, smoothing):
        super().__init__()
        _n_classes = n_classes
        self.confidence = 1. - smoothing
        smoothing_value = smoothing / (_n_classes - 1)
        one_hot = torch.full((_n_classes,), smoothing_value).cuda()
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, output, target):
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(F.log_softmax(output, 1), model_prob, reduction='sum')


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum()
        return b

