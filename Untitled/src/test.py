"""Evaluate a trained NECST Torch model on the test split."""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Allow running `python src/test.py` without needing PYTHONPATH set.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.args import parse_args, parse_architecture
from src.necst import NecstTorch
from src.utils import (
    build_dataloaders,
    save_markov_chain_plot,
    save_reconstruction_plot,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_pref: str) -> torch.device:
    device_pref = device_pref.lower()
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(args, input_dim, device):
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
    )
    return model.to(device)


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


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def main():
    args = parse_args()
    if not args.checkpoint:
        raise ValueError("--checkpoint path is required for testing")

    set_seed(args.seed)
    device = get_device(args.device)
    _, _, test_loader, input_dim = build_dataloaders(
        args.dataset, args.datadir, args.batch_size, need_train=False, need_valid=False, need_test=True
    )
    model = build_model(args, input_dim, device)
    load_checkpoint(model, args.checkpoint, device)

    test_loss = evaluate_reconstruction(model, test_loader, device)
    print(f"Loaded checkpoint '{args.checkpoint}' - Test BCE: {test_loss:.4f}")

    results_dir = Path("results")
    if args.plot_reconstruction:
        plot_path = results_dir / "test_reconstruction.png"
        save_reconstruction_plot(
            model,
            test_loader,
            device,
            plot_path,
            title="Test reconstructions",
        )
    if args.plot_markov:
        markov_path = results_dir / "test_markov_chain.png"
        save_markov_chain_plot(model, device, markov_path)


if __name__ == "__main__":
    main()
