import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from .loss import VIMCOLoss

class NECSTBinary(nn.Module):
    """
    NECST Implementation for binary datasets (BinaryMNIST, Omniglot, Random Bits).
    """
    def __init__(self, input_dim=784, z_dim=100, hidden_dim=500, vimco_samples=5, reg_param=1e-4, noise=0.0, test_noise=0.0, discrete_relax=True, lr=1e-3,):
        super().__init__()

        # network params
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.vimco_samples = vimco_samples
        self.reg_param = reg_param
        self.noise = noise
        self.test_noise = test_noise
        self.discrete_relax = discrete_relax
        self.vimco_loss_fn = VIMCOLoss(reg_param=reg_param)
        
        # encoder neural network - one hidden layer MLP (500 units)
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc_out = nn.Linear(hidden_dim, z_dim)

        # decoder neural network: two hidden layers (500 units each)
        self.decoder_net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec_out = nn.Linear(hidden_dim, input_dim)
    
    ### useful functions --- <VIMCO loss> -----
    def vimco_loss(self, x, x_reconstr_logits, log_q_h):
        return self.vimco_loss_fn(self, x, x_reconstr_logits, log_q_h)

    ### useful functions --- <encoder + decoder> -----
    
    # call encoder
    def encoder(self, x):
        """
        give p(y_hat|x) (BSC logits).
        """
        h = x.view(x.size(0), -1)
        h = self.encoder_net(h)
        return self.enc_out(h)
  
    # call decoder
    def decoder(self, z):
        """
        give p(x_hat|y)
        """
        d = self.decoder_net(z)
        return self.dec_out(d)

    ### useful functions --- <computation graphs> ---

    def create_collapsed_computation_graph(self, x, sample_count=None, noise=None):
        """
        PyTorch analogue of the TF create_collapsed_computation_graph for BSC.
        Returns:
            total_prob: Bernoulli probabilities after channel noise
            y: sampled latent bits (shape [k, batch, z_dim])
            classif_y: single-sample latent for downstream classification
            q: Bernoulli distribution object
            x_recon_logits: decoder outputs (shape [k, batch, input_dim])
        """

        if sample_count is None:
            sample_count = self.vimco_samples if self.vimco_samples is not None else 1
        
        if noise is None:
            noise = self.noise

        # encode
        logits = self.encoder(x)

        # for downstream classification
        classif_q = Bernoulli(logits=logits)
        classif_y = classif_q.sample()

        probs = torch.sigmoid(logits)
        if noise and noise > 0:
            total_prob = probs - (2 * probs * noise) + noise
            total_prob = total_prob.clamp(1e-6, 1 - 1e-6)
            q = Bernoulli(probs=total_prob)
        else:
            total_prob = probs
            q = Bernoulli(logits=logits)

        if sample_count > 1:
            y = q.sample((sample_count,)).float()
        else:
            y = q.sample().unsqueeze(0).float()
        y_flat = y.view(-1, self.z_dim)
        
        recon_flat = self.decoder(y_flat)
        x_recon_logits = recon_flat.view(sample_count, x.size(0), -1)

        return total_prob, y, classif_y, q, x_recon_logits

    def get_collapsed_stochastic_test_sample(self, x, noise=None):
        """
        Test-time analogue: single sample with test noise level.
        """
        if noise is None:
            noise = self.test_noise

        logits = self.encoder(x)
        classif_q = Bernoulli(logits=logits)
        classif_y = classif_q.sample()

        probs = torch.sigmoid(logits)
        if noise and noise > 0:
            total_prob = probs - (2 * probs * noise) + noise
            total_prob = total_prob.clamp(1e-6, 1 - 1e-6)
            q = Bernoulli(probs=total_prob)
        else:
            total_prob = probs
            q = Bernoulli(logits=logits)

        y = q.sample().float()
        x_reconstr_logits = self.decoder(y)
        return total_prob, y, classif_y, q, x_reconstr_logits

    # forward
    def forward(self, x, sample_bits=False):
        logits = self.encoder(x)
        probs = torch.sigmoid(logits)
        if sample_bits:
            z = torch.bernoulli(probs)
        else:
            z = probs
        recon_logits = self.decoder(z)
        return recon_logits, logits, z