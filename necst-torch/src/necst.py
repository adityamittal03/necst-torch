import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

class NecstTorch(nn.Module):
    """
    NECST Implementation for binary datasets (BinaryMNIST, Omniglot, Random Bits).
    """
    def __init__(self, 
                 input_dim=784, 
                 z_dim=100, 
                 hidden_dim=500,
                 vimco_samples=5, 
                 noise=0.0, 
                 test_noise=0.0,
                 encoder_layers=None,
                 decoder_layers=None,):
        super().__init__()

        # network params
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.vimco_samples = vimco_samples
        self.noise = noise
        self.test_noise = test_noise
        self.encoder_layers = encoder_layers or [hidden_dim]
        self.decoder_layers = decoder_layers or [hidden_dim, hidden_dim]

        # encoder
        self.encoder_net, enc_last_dim = self._build_mlp(input_dim, self.encoder_layers)
        self.enc_out = nn.Linear(enc_last_dim, z_dim)

        # decoder 
        self.decoder_net, dec_last_dim = self._build_mlp(z_dim, self.decoder_layers)
        self.dec_out = nn.Linear(dec_last_dim, input_dim)

    def _build_mlp(self, in_dim, layer_dims):
        layers = []
        prev_dim = in_dim
        for dim in layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            prev_dim = dim
        return nn.Sequential(*layers), prev_dim

    ### encoder/decoder 
    def encoder(self, x):
        """
        encode into latent representations
        """
        h = x.view(x.size(0), -1)
        h = self.encoder_net(h)
        return self.enc_out(h)

    def decoder(self, z):
        """
        decode into reconstructions
        """
        d = self.decoder_net(z)
        return self.dec_out(d)

    # forward()
    def forward(self, x):
        """
        computes p(y|x) with BSC error
        """

        # encode
        logits = self.encoder(x)
        probs = torch.sigmoid(logits)

        # downstream tasks
        classif_q = Bernoulli(probs=probs)
        classif_y = classif_q.sample()

        # noise
        if self.noise > 0:
            total_prob = probs - (2 * probs * self.noise) + self.noise
            total_prob = total_prob.clamp(1e-6, 1 - 1e-6)
            q = Bernoulli(probs=total_prob)
        else:
            total_prob = probs
            q = Bernoulli(probs=probs)

        # sample 
        sample_count = self.vimco_samples if self.vimco_samples and self.vimco_samples > 1 else 1
        if self.vimco_samples and self.vimco_samples > 1:
            y = q.sample((self.vimco_samples,)).float()
        else:
            y = q.sample().unsqueeze(0).float()
        y_flat = y.view(-1, self.z_dim)

        # decode
        recon_flat = self.decoder(y_flat)
        x_recon_logits = recon_flat.view(sample_count, x.size(0), -1)

        return total_prob, y, classif_y, q, x_recon_logits

    # forward() on test
    def forward_test(self, x):
        """
        same for test data.
        """
        # encode
        logits = self.encoder(x)
        probs = torch.sigmoid(logits)

        # for downstream
        classif_q = Bernoulli(probs=probs)
        classif_y = classif_q.sample()

        # add noise
        if self.test_noise > 0:
            total_prob = probs - (2 * probs * self.test_noise) + self.test_noise
            total_prob = total_prob.clamp(1e-6, 1 - 1e-6)
            q = Bernoulli(probs=total_prob)
        else:
            total_prob = probs
            q = Bernoulli(logits=logits)

        # decode
        y = q.sample().float()
        x_reconstr_logits = self.decoder(y)

        return total_prob, y, classif_y, q, x_reconstr_logits
