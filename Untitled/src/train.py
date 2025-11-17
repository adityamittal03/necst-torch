"""training script for NECST Torch models (using Gumbel or VIMCO optimization)."""

import random
import sys
from pathlib import Path

# Allow running `python src/train.py` by injecting the repo root onto sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange

from src.args import parse_args, parse_architecture
from src.loss import Loss
from src.necst import NecstTorch
from src.utils import build_dataloaders, save_markov_chain_plot, save_reconstruction_plot

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch_gumbel(model, loader, loss_fn, optimizer, device):
    model.train()
    running = 0.0
    for batch in loader:
        batch = batch.to(device)
        logits = model.encoder(batch)
        z = loss_fn.gumbel_softmax(logits)
        recon_logits = model.decoder(z)
        recon_probs = torch.sigmoid(recon_logits)
        qy = torch.softmax(logits / loss_fn.temperature, dim=-1)

        loss = loss_fn(model, recon_probs, batch, qy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(len(loader), 1)


def train_epoch_vimco(model, loader, loss_fn, theta_opt, phi_opt, device):
    model.train()
    running = 0.0
    for batch in loader:
        batch = batch.to(device)
        _, y, _, q, x_recon_logits = model(batch)
        log_q_h = q.log_prob(y).sum(dim=-1)

        theta_loss, phi_loss, _ = loss_fn(model, batch, x_recon_logits, log_q_h)

        theta_opt.zero_grad()
        phi_opt.zero_grad()
        theta_loss.backward(retain_graph=True)
        phi_loss.backward()
        theta_opt.step()
        phi_opt.step()
        running += theta_loss.item()
    return running / max(len(loader), 1)


def evaluate_reconstruction(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, _, _, _, recon_logits = model.forward_test(batch)
            loss = F.binary_cross_entropy_with_logits(recon_logits, batch, reduction="sum")
            total += loss.item()
    return total / len(loader.dataset)


def save_checkpoint(model, args):
    if not args.save_path:
        return
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "config": vars(args)}
    torch.save(payload, save_path)
    print(f"Saved checkpoint to {save_path}")


def plot_training_loss(losses, destination: Path):
    """Persist a line plot of training loss vs. epoch for quick inspection."""
    if not losses:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training loss over epochs")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()
    print(f"Saved training loss plot to {destination}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    if args.device.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_loader, valid_loader, test_loader, input_dim = build_dataloaders(
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

    print(f"Model beginning training for {args.epochs} epochs. Device chosen: {device}")

    train_losses = []
    for epoch in trange(1, args.epochs + 1, desc="Epochs"):
        if use_gumbel:
            train_loss = train_epoch_gumbel(model, train_loader, loss_fn, optimizer, device)
        else:
            train_loss = train_epoch_vimco(model, train_loader, loss_fn, theta_opt, phi_opt, device)

        train_losses.append(train_loss)

        if epoch % 100 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f}")
            if valid_loader is not None:
                val_loss = evaluate_reconstruction(model, valid_loader, device)
                print(f"    validation BCE: {val_loss:.4f}")

    if test_loader is not None:
        test_loss = evaluate_reconstruction(model, test_loader, device)
        print(f"Test BCE: {test_loss:.4f}")

    save_checkpoint(model, args)
    if args.plot_reconstruction:
        recon_loader = test_loader or valid_loader or train_loader
        if recon_loader is None:
            print("Skipping reconstruction plot (no loader available).")
        else:
            plot_path = Path("results") / "reconstruction.png"
            save_reconstruction_plot(
                model,
                recon_loader,
                device,
                plot_path,
                title="Model reconstructions",
            )
    if args.plot_markov:
        markov_path = Path("results") / "markov_chain.png"
        save_markov_chain_plot(model, device, markov_path)
    if args.plot_loss:
        plot_path = Path("results") / "training_loss.png"
        plot_training_loss(train_losses, plot_path)
