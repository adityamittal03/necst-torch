import math
import torch
import torch.nn.functional as F

class Loss:
    """
    loss functions:
        1) VIMCO loss: https://github.com/jaanli/vimco_tf?tab=readme-ov-file (initial implementation)
            ### this was converted from tensorflow -> torch
        2) Gumbel softmax: https://github.com/YongfeiYan/Gumbel_Softmax_VAE (initial implementation)
            ### modified their code
    """
    def __init__(self, reg_param=1e-4, use_relaxation=False, temperature=0.67, hard=False):
        self.reg_param = reg_param
        self.use_relaxation = use_relaxation
        self.temperature = temperature
        self.hard = hard

    def _get_device(self, params):
        for param in params:
            if param.device is not None:
                return param.device
        return torch.device("cpu")

    # l_2 regularization for encoder/decoder parameters 
    def regularization_loss(self, params):
        params = list(params)
        device = self._get_device(params)
        if self.reg_param <= 0:
            return torch.tensor(0.0, device=device)
        reg = torch.tensor(0.0, device=device)
        for param in params:
            reg = reg + param.pow(2).sum()
        return self.reg_param * reg

    # 1. vimco loss
    def build_vimco_loss(self, l):
        """
        building the Importance Weighted Autoencoder https://arxiv.org/abs/1509.00519
        Args:
            l: tensor of shape (k, batch_size)
        """
        k, _ = l.shape
        if k == 1:
            zeros = torch.zeros_like(l)
            return zeros, torch.ones_like(l), l.squeeze(0)
        l_logsumexp = torch.logsumexp(l, dim=0, keepdim=True)
        L_hat = l_logsumexp - math.log(k)
        s = l.sum(dim=0, keepdim=True)
        diag_mask = torch.eye(k, device=l.device).unsqueeze(-1)
        off_diag_mask = 1.0 - diag_mask
        diff = (s - l).unsqueeze(0)
        l_i_diag = diff * diag_mask / (k - 1)
        l_i_off_diag = off_diag_mask * l.unsqueeze(0)
        l_i = l_i_diag + l_i_off_diag
        L_hat_minus_i = torch.logsumexp(l_i, dim=1) - math.log(k)
        w = torch.exp(l - l_logsumexp).detach()
        local_l = (L_hat - L_hat_minus_i).detach()
        return local_l, w, L_hat.squeeze(0)

    def vimco_loss(self, model, x, x_reconstr_logits, log_q_h):
        params = list(model.parameters())
        reg_loss = self.regularization_loss(params)
        target = x.unsqueeze(0).expand_as(x_reconstr_logits)
        reconstr_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            x_reconstr_logits, target, reduction="none"
        ).sum(dim=-1)
        if log_q_h.dim() > 2:
            log_q_h = log_q_h.sum(dim=-1)
        local_l, w, full_loss = self.build_vimco_loss(reconstr_loss)
        theta_loss = torch.mean(torch.sum(w * reconstr_loss, dim=0))
        phi_loss = torch.mean(torch.sum(local_l * log_q_h + w * reconstr_loss, dim=0))
        phi_loss = phi_loss + reg_loss
        full_loss = full_loss.mean()
        return theta_loss, phi_loss, full_loss

    # 2. gumbel-softmax
    def gumbel_sigmoid_sample(self, logits, temperature):
        """
        Samples from a Relaxed Bernoulli distribution (Binary Gumbel).
        Math: y = sigmoid((logits + log(u) - log(1-u)) / temp)
        """
        eps = 1e-20
        # Sample uniform noise
        u = torch.rand_like(logits)
        # Convert to Logistic noise
        logistic_noise = torch.log(u + eps) - torch.log(1 - u + eps)
        # Apply temperature and sigmoid
        y = torch.sigmoid((logits + logistic_noise) / temperature)
        return y
    

    def gumbel_softmax(self, logits, temperature=None, hard=None):
            """
            Modified to perform BINARY Gumbel-Sigmoid sampling.
            """
            temperature = self.temperature if temperature is None else temperature
            hard = self.hard if hard is None else hard
            
            y = self.gumbel_sigmoid_sample(logits, temperature)
            if not hard:
                return y

            y_hard = torch.round(y)
            
            y_hard = (y_hard - y).detach() + y
            return y_hard
    

    def gumbel_loss(self, recon_logits, x, qy_soft):
            """
            recon_logits: Decoder output (unnormalized)
            x: Target image
            qy_soft: Probability of bit=1 (from encoder, BEFORE noise/sampling)
            """
            BCE = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum") / x.shape[0]
            p = torch.clamp(qy_soft, 1e-6, 1 - 1e-6) 
            prior = 0.5
            
            KLD_element = p * torch.log(p / prior) + (1 - p) * torch.log((1 - p) / (1 - prior))
            KLD = torch.sum(KLD_element, dim=-1).mean()

            return BCE + KLD


    def _gumbel_softmax_loss(self, model, recon_x, x, qy):
        params = list(model.parameters())
        reg_loss = self.regularization_loss(params)
        return self.gumbel_loss(recon_x, x, qy) + reg_loss

    def __call__(self, model, *args, **kwargs):
        if self.use_relaxation:
            return self._gumbel_softmax_loss(model, *args, **kwargs)
        return self.vimco_loss(model, *args, **kwargs)
