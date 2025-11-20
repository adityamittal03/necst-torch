import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

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

        # add noise
        if self.noise > 0:
            total_prob = probs - (2 * probs * self.noise) + self.noise
            total_prob = total_prob.clamp(1e-6, 1 - 1e-6)
            q = Bernoulli(probs=total_prob)
        else:
            total_prob = probs
            q = Bernoulli(probs=probs)

        # vimco samples
        sample_count = self.vimco_samples if self.vimco_samples and self.vimco_samples > 1 else 1
        if self.vimco_samples and self.vimco_samples > 1:
            y = q.sample((self.vimco_samples,)).float()
        else:
            y = q.sample().unsqueeze(0).float()
        y_flat = y.view(-1, self.z_dim)

        # decode
        recon_flat = self.decoder(y_flat)
        x_recon_logits = recon_flat.view(sample_count, x.size(0), -1)

        return y, q, x_recon_logits

    # forward() on test
    def forward_test(self, x):
        """
        same for test data.
        """
        # encode
        logits = self.encoder(x)
        probs = torch.sigmoid(logits)

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

        return y, q, x_reconstr_logits

    ### test error
    def evaluate_reconstruction(self, loader, per_element: bool = True):
        """
        average BCE reconstruction loss over dataset.
        per_element=True -> per-pixel BCE; False -> per-image BCE.
        """

        if loader is None:
            raise ValueError("Loader is required for evaluation.")

        self.eval()
        device = next(self.parameters()).device
        
        total_loss = 0.0
        total_elems = 0
        total_samples = 0
        with torch.no_grad():
            for batch in loader:
                x = batch if not isinstance(batch, (list, tuple)) else batch[0]
                x = x.to(device)
                x_flat = x.view(x.size(0), -1)
                _, _, recon_logits = self.forward_test(x)
                total_loss += F.binary_cross_entropy_with_logits(
                    recon_logits, x_flat, reduction="sum"
                ).item()
                total_elems += x_flat.numel()
                total_samples += x_flat.size(0)

        if per_element:
            return total_loss / max(total_elems, 1)

        return total_loss / max(total_samples, 1)


    def plot_reconstructions(self, loader, n_samples: int = 8, title: str = None):
        """
        plots reconstructions
        """
        import matplotlib.pyplot as plt

        self.eval()
        ds = loader.dataset
        n_samples = min(n_samples, len(ds))
        side = int(round(self.input_dim ** 0.5))
        if side * side != self.input_dim:
            raise ValueError(f"Cannot reshape input_dim={self.input_dim} into a square image.")

        idx = torch.randperm(len(ds))[:n_samples]
        batch = []
        for i in idx:
            sample = ds[i]
            img = sample[0] if isinstance(sample, (list, tuple)) else sample
            batch.append(img)
        batch = torch.stack(batch)
        device = next(self.parameters()).device
        batch = batch.to(device)

        with torch.no_grad():
            _, _, recon_logits = self.forward_test(batch)
            recon = torch.sigmoid(recon_logits)

        orig = batch.view(batch.size(0), side, side).cpu()
        recon = recon.view(recon.size(0), side, side).cpu()

        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.4, 3), squeeze=False)
        for i in range(n_samples):
            axes[0, i].imshow(orig[i], cmap="gray_r", vmin=0, vmax=1)
            axes[0, i].axis("off")
            axes[0, i].set_title("input")
            axes[1, i].imshow(recon[i], cmap="gray_r", vmin=0, vmax=1)
            axes[1, i].axis("off")
            axes[1, i].set_title("recon")
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        return fig
