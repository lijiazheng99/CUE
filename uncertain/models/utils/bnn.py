import torch
from torch import nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from uncertain.dataloader import prepare_input, prepare_output

from blitz.modules import BayesianLinear

class BNN(PreTrainedModel):
    def __init__(self, config, hidden_size, latent_size, **kwargs):
        super().__init__(config, **kwargs)

        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.bnn = BayesianLinear(config.hidden_size, self.latent_size, bias=False)
        self.linear = nn.Linear(self.latent_size,config.hidden_size, bias=False)

        self.cossim = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Initialize weights and apply final processing
        self.post_init()

    def encoder(self, input):
        return self.bnn(input)
    
    def decoder(self, z):
        return self.linear(z)
    
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
        # start = input_ids.nonzero().min().item()
        # end = input_ids.nonzero().max().item()
        hidden_states = hidden_states
        # input_ids = input_ids[start:end][1:-1]
        # input_ids = input_ids[1:-1]
        delta_x_coss = torch.Tensor([prepare_input(float('inf')) if self.cossim(delta_x, each) < float(0) else self.cossim(delta_x, each) for each in perturbed_z])
        influntial_idx = torch.argmin(delta_x_coss)
        if torch.min(delta_x_coss) == float('inf'):
            delta_x_coss = torch.Tensor([self.cossim(delta_x, each) for each in perturbed_z])
            influntial_idx = torch.argmax(delta_x_coss)
        # word_coss = torch.Tensor([prepare_input(float('inf')) if self.cos(perturbed_z[influntial_idx], each) < float(0) else self.cos(perturbed_z[influntial_idx], each) for each in hidden_states])
        # word_idx = torch.argmin(word_coss)
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
        