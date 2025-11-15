import math
import torch

### this was converted from tensorflow -> torch (used chatGPT lol pls confirm)
class VIMCOLoss:
    """
    VIMCO loss.
    """
    def __init__(self, reg_param=1e-4):
        self.reg_param = reg_param

    def _get_device(self, params):
        for param in params:
            if param.device is not None:
                return param.device
        return torch.device("cpu")

    def regularization_loss(self, params):
        params = list(params)
        device = self._get_device(params)
        if self.reg_param <= 0:
            return torch.tensor(0.0, device=device)
        reg = torch.tensor(0.0, device=device)
        for param in params:
            reg = reg + param.pow(2).sum()
        return self.reg_param * reg

    def build_vimco_loss(self, l):
        """
        Torch equivalent of TF build_vimco_loss.
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

    def __call__(self, model, x, x_reconstr_logits, log_q_h):
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
