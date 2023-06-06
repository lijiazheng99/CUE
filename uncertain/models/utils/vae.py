from cmath import cos
from os import XATTR_SIZE_MAX
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from uncertain.dataloader import prepare_input, prepare_output

class VAE(PreTrainedModel):
    def __init__(self, config, hidden_size, latent_size, var_scale, eta_bn_prop, **kwargs):
        super().__init__(config, **kwargs)

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.prior_covar_weights = None
        self.var_scale = var_scale
        self.eta_bn_prop = eta_bn_prop

        # create the encoder
        self.encoder_layer = nn.Linear(self.hidden_size, self.latent_size, bias=False)

        self.encoder_dropout_layer = nn.Dropout(p=0.2)
        xavier_uniform_(self.encoder_layer.weight)
    
        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(self.latent_size, self.latent_size)
        self.logvar_layer = nn.Linear(self.latent_size, self.latent_size)

        self.mean_bn_layer = nn.BatchNorm1d(self.latent_size, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.ones(self.latent_size))
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(self.latent_size, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.ones(self.latent_size))
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)

        # create the decoder
        self.bate_layer = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        # self.bate_layer = nn.Linear(self.latent_size, self.hidden_size)
        xavier_uniform_(self.bate_layer.weight)

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(self.hidden_size, eps=0.001, momentum=0.001, affine=True)
        self.eta_bn_layer.weight.data.copy_(torch.ones(self.hidden_size))
        self.eta_bn_layer.weight.requires_grad = False

        self.cossim = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Initialize weights and apply final processing
        self.post_init()

    def encoder(self, input):
        en0_x = self.encoder_layer(input)

        encoder_output = F.softplus(en0_x)
        encoder_output_do = self.encoder_dropout_layer(encoder_output)

        posterior_mean = self.mean_layer(encoder_output_do)
        posterior_logvar = self.logvar_layer(encoder_output_do)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)

        posterior_var = posterior_logvar_bn.exp()
        eps = input.data.new().resize_as_(posterior_mean_bn.data).normal_()

        return posterior_mean_bn + posterior_var.sqrt() * eps * self.var_scale
    
    def decoder(self, z):
        theta = self.z_dropout_layer(z)

        X_recon_no_bn = self.bate_layer(theta)
        X_recon_bn = self.eta_bn_layer(X_recon_no_bn)
        return self.eta_bn_prop * X_recon_bn + (1.0 - self.eta_bn_prop) * X_recon_no_bn
    
    def latent_perturb(self, z):
        decoded_perturb = [[]] * z.size()[0]
        for i in range(0, self.latent_size):
            ### Create a zero matrix and assign each column with 1.o
            dz = torch.zeros(z.size()).to(self.device)
            dz[:,i] = 1.
            dz = dz * z
            decoded_dz = self.decoder(z=dz)
            for idx, vector in enumerate(decoded_dz):
                if len(decoded_perturb[idx]) == 0:
                    decoded_perturb[idx] = [vector]
                else:
                    decoded_perturb[idx] += [vector]
        return decoded_perturb
    
    def cos_word_by_dimension(self, x, hidden_states, perturbed_z, input_ids):
        results = {}
        for idx, each_dim in enumerate(perturbed_z):
            coss = torch.Tensor([prepare_input(float('inf')) if self.cos(each_dim,each) < float(0) else self.cos(each_dim,each) for each in hidden_states ])
            results.update({idx:[input_ids[torch.argmin(coss)],coss[torch.argmin(coss)]]})
        x_dim = torch.argmin(torch.Tensor([prepare_input(float('inf')) if self.cos(each_dim,x) < float(0) else self.cos(each_dim,x) for each_dim in perturbed_z ]))
        return results, x_dim
    
    def cos_dimension_by_word(self, x, hidden_states, perturbed_z, input_ids):
        results = {}
        for idx, each in enumerate(hidden_states):
            coss = torch.Tensor([prepare_input(float('inf')) if self.cos(each_dim,each) < float(0) else self.cos(each_dim,each) for each_dim in perturbed_z ])
            results.update({input_ids[idx]:[torch.argmin(coss), coss[torch.argmin(coss)]]})
        x_dim = torch.argmin(torch.Tensor([prepare_input(float('inf')) if self.cos(each_dim,x) < float(0) else self.cos(each_dim,x) for each_dim in perturbed_z ]))
        return results, x_dim
    
    def cos_delta_by_dimension(self, delta_x, perturbed_z, hidden_states):
        hidden_states = hidden_states
        delta_x_coss = torch.Tensor([prepare_input(float('inf')) if self.cossim(delta_x, each) < float(0) else self.cossim(delta_x, each) for each in perturbed_z])
        influntial_idx = torch.argmin(delta_x_coss)
        if torch.min(delta_x_coss) == float('inf'):
            delta_x_coss = torch.Tensor([self.cossim(delta_x, each) for each in perturbed_z])
            influntial_idx = torch.argmax(delta_x_coss)
        word_coss = [self.cossim(perturbed_z[influntial_idx], each) for each in hidden_states]
        return influntial_idx, delta_x_coss[influntial_idx], word_coss
    
    def cos_delta_by_dimension_batches(self, delta_x, perturbed_z, hidden_states):
        influntial_idxs, influntial_idx_coss, word_cosss = [], [], []
        for delta_x_, perturbed_z_, hidden_states_ in zip(delta_x, perturbed_z, hidden_states):
            influntial_idx, influntial_idx_cos, word_coss = self.cos_delta_by_dimension(delta_x_, perturbed_z_, hidden_states_)
            influntial_idxs.append(prepare_output(influntial_idx))
            influntial_idx_coss.append(prepare_output(influntial_idx_cos))
            word_cosss.append(prepare_output(word_coss))
        return influntial_idxs, influntial_idx_coss, word_cosss
    
    def recon_hidden_states(self, hidden_states):
        recon_hiddens = []
        for each_hidden_state in hidden_states:
            latent_z = self.encoder(each_hidden_state)
            recon_hidden_state = self.decoder(latent_z)
            recon_hiddens += [recon_hidden_state]
        return torch.stack(recon_hiddens)
