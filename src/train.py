"""training script for NECST Torch models (using Gumbel & VIMCO optimization)."""

import random
import sys
import csv  
import time
from pathlib import Path

# Allow running `python src/train.py` by injecting the repo root onto sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import trange

from src.args import parse_args, parse_architecture
from src.loss import Loss
from src.necst import NecstTorch
from src.utils import build_dataloaders

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

### use gumbel-softmax optimization
def train_epoch_gumbel(model, loader, loss_fn, optimizer, device):
    model.train()
    
    stats = {
        "loss": 0.0,           # Total Loss
        "bce_soft": 0.0,       # Soft Reconstruction Error
        "bce_hard": 0.0,       # Hard (Binary) Reconstruction Error
        "kld": 0.0,            # Regularization term
        "grad_norm": 0.0,      # Stability metric
        "active_bits": 0.0,    # How many bits are carrying info
        "saturation": 0.0,     # How confident is the encoder
    }
    
    steps = 0
    
    for batch in loader:
        batch = batch.to(device)
        batch_flat = batch.view(batch.size(0), -1)
        
        logits = model.encoder(batch)
        
        z_soft = loss_fn.gumbel_softmax(logits, hard=False) 
        
        z_hard = (torch.sigmoid(logits) > 0.5).float()
        recon_logits_soft = model.decoder(z_soft)
        recon_logits_hard = model.decoder(z_hard)
        qy_probs = torch.sigmoid(logits)
        loss = loss_fn(model, recon_logits_soft, batch_flat, qy_probs)
        
        with torch.no_grad():
            bce_soft = torch.nn.functional.binary_cross_entropy_with_logits(
                recon_logits_soft, batch_flat, reduction='mean'
            )
            bce_hard = torch.nn.functional.binary_cross_entropy_with_logits(
                recon_logits_hard, batch_flat, reduction='mean'
            )
            kl_term = loss.item() - bce_soft.item() # Rough estimate based on total - bce
            
            p = qy_probs
            active_mask = (p > 0.1) & (p < 0.9)
            active_count = active_mask.sum(dim=1).float().mean()
            
            saturated_mask = (p < 0.05) | (p > 0.95)
            saturation_pct = saturated_mask.float().mean()

        optimizer.zero_grad()
        loss.backward()
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()

        stats["loss"] += loss.item()
        stats["bce_soft"] += bce_soft.item()
        stats["bce_hard"] += bce_hard.item()
        stats["kld"] += kl_term
        stats["grad_norm"] += total_norm
        stats["active_bits"] += active_count.item()
        stats["saturation"] += saturation_pct.item()
        steps += 1
    return {k: v / max(steps, 1) for k, v in stats.items()}


### use vimco loss optimization
def train_epoch_vimco(model, loader, loss_fn, theta_opt, phi_opt, device):
    model.train()
    running = 0.0
    for batch in loader:
        batch = batch.to(device)
        y, q, x_recon_logits = model(batch)
        log_q_h = q.log_prob(y).sum(dim=-1)
        flat = batch.view(batch.size(0), -1)
        theta_loss, phi_loss, _ = loss_fn(model, flat, x_recon_logits, log_q_h)
        theta_opt.zero_grad()
        phi_opt.zero_grad()
        theta_loss.backward(retain_graph=True)
        phi_loss.backward()
        theta_opt.step()
        phi_opt.step()
        running += theta_loss.item()
    return running / max(len(loader), 1)

### compute validation loss using the same objective as training
def evaluate_objective(model, loader, loss_fn, use_gumbel, device):
    model.eval()
    running = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch if not isinstance(batch, (list, tuple)) else batch[0]
            batch = batch.to(device)
            if use_gumbel:
                logits = model.encoder(batch)
                z = loss_fn.gumbel_softmax(logits)
                recon_logits = model.decoder(z)
                batch_flat = batch.view(batch.size(0), -1)
                qy_probs = torch.sigmoid(logits) 
                loss = loss_fn(model, recon_logits, batch_flat, qy_probs)
                running += loss.item()
            else:
                y, q, x_recon_logits = model(batch)
                log_q_h = q.log_prob(y).sum(dim=-1)
                flat = batch.view(batch.size(0), -1)
                theta_loss, _, _ = loss_fn(model, flat, x_recon_logits, log_q_h)
                running += theta_loss.item()
            count += 1
    return running / max(count, 1)

# plot train + validation loss
def plot_losses(train_losses, val_losses, save_dir=Path("plots"), filename="training_loss.png"):
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train loss")
    if val_losses:
        plt.plot(epochs, val_losses, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training curves")
    plt.legend()
    out_path = save_dir / filename
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved loss plot to {out_path}")
    plt.close()


# save model here
def save_checkpoint(model, args):
    if not args.save_path:
        return
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "config": vars(args)}
    torch.save(payload, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    
    # create model architecture
    set_seed(args.seed)
    if args.device.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_loader, valid_loader, _, input_dim = build_dataloaders(
        args.dataset, args.datadir, args.batch_size
    )

    enc_layers = parse_architecture(args.enc_arch)
    dec_layers = parse_architecture(args.dec_arch)
    model = NecstTorch(
        input_dim=input_dim,
        z_dim=args.latent_dim,
        vimco_samples=args.vimco_samples,
        noise=args.noise,
        test_noise=args.test_noise,
        encoder_layers=enc_layers or None,
        decoder_layers=dec_layers or None,
    ).to(device)

    loss_type = args.loss_type.lower()
    use_gumbel = loss_type == "gumbel"
    loss_fn = Loss(
        reg_param=args.reg_param,
        use_relaxation=use_gumbel,
        temperature=args.temperature,
        hard=args.gumbel_hard,
    )

    # Setup CSV Logging for Gumbel
    log_file = None
    if use_gumbel:
        timestamp = int(time.time())
        # Naming convention: gumbel_{dataset}_z{latent}_lr{lr}_{timestamp}.csv
        log_name = f"gumbel_{args.dataset}_z{args.latent_dim}_lr{args.lr}_{timestamp}.csv"
        log_path = Path("logs") / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "loss", "val_loss", "bce_soft", "bce_hard", 
                "kld", "grad_norm", "active_bits", "saturation", "temperature"
            ])
        print(f"Logging stats to {log_path}")

    # optimizer
    opt_name = args.optimizer.lower()
    if use_gumbel:
        if opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        elif opt_name == "momentum":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer '{args.optimizer}'")
    else:
        phi_params = list(model.encoder_net.parameters()) + list(model.enc_out.parameters())
        theta_params = list(model.decoder_net.parameters()) + list(model.dec_out.parameters())

        if opt_name == "adam":
            phi_opt = torch.optim.Adam(phi_params, lr=args.lr)
            theta_opt = torch.optim.Adam(theta_params, lr=args.lr)
        elif opt_name == "sgd":
            phi_opt = torch.optim.SGD(phi_params, lr=args.lr)
            theta_opt = torch.optim.SGD(theta_params, lr=args.lr)
        elif opt_name == "momentum":
            phi_opt = torch.optim.SGD(phi_params, lr=args.lr, momentum=0.9)
            theta_opt = torch.optim.SGD(theta_params, lr=args.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer '{args.optimizer}'")

    # train over epoch + batches
    train_curve = []
    val_curve = []

    for epoch in trange(1, args.epochs + 1, desc="Epochs"):
        if use_gumbel:
            epoch_stats = train_epoch_gumbel(model, train_loader, loss_fn, optimizer, device)
            train_loss = epoch_stats["loss"]
        else:
            train_loss = train_epoch_vimco(model, train_loader, loss_fn, theta_opt, phi_opt, device)

        train_curve.append(train_loss)

        # validation
        val_loss = 0.0
        if valid_loader is not None:
            val_loss = evaluate_objective(model, valid_loader, loss_fn, use_gumbel, device)
            val_curve.append(val_loss)

        # print every 10 epoch
        if epoch % 10 == 0:
            msg = f"Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f}"
            if valid_loader is not None:
                msg += f" | val loss: {val_curve[-1]:.4f}"
            if use_gumbel:
                 msg += f" | Hard BCE: {epoch_stats['bce_hard']:.2f} | Active: {epoch_stats['active_bits']:.1f}"
            print(msg)

        # Log to CSV
        if use_gumbel:
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, 
                    epoch_stats["loss"], 
                    val_loss if valid_loader else "",
                    epoch_stats["bce_soft"],
                    epoch_stats["bce_hard"],
                    epoch_stats["kld"],
                    epoch_stats["grad_norm"],
                    epoch_stats["active_bits"],
                    epoch_stats["saturation"],
                    loss_fn.temperature
                ])

    save_checkpoint(model, args)

    if args.plot_loss:
        plot_losses(train_curve, val_curve, save_dir=Path("plots"))