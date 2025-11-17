"""Standalone plotting utility for NECST reconstructions and Markov chains."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repo root is on sys.path when running as `python src/plot.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.args import parse_architecture  # noqa: E402
from src.necst import NecstTorch  # noqa: E402
from src.utils import (  # noqa: E402
    build_dataloaders,
    save_markov_chain_plot,
    save_reconstruction_plot,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate reconstruction and Markov-chain plots for trained NECST models."
    )
    parser.add_argument("--datadir", type=str, default="./data", help="Dataset root directory")
    parser.add_argument(
        "--dataset",
        type=str.lower,
        default="binarymnist",
        help="Dataset to load (binarymnist or random_bits)",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for evaluation loader")
    parser.add_argument("--device", type=str.lower, default="cpu", help="Device to run on (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for sampling")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")

    # model configuration (must match training run)
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent dimensionality")
    parser.add_argument("--enc-arch", type=str, default="500", help="Encoder layer sizes (comma-separated)")
    parser.add_argument("--dec-arch", type=str, default="500,500", help="Decoder layer sizes (comma-separated)")
    parser.add_argument("--vimco-samples", type=int, default=5, help="Samples used during training")
    parser.add_argument("--noise", type=float, default=0.0, help="Channel noise during training")
    parser.add_argument("--test-noise", type=float, default=0.0, help="Channel noise during evaluation")

    # plotting controls
    parser.add_argument(
        "--recon-samples",
        type=int,
        default=8,
        help="Number of test examples to plot (1-8)",
    )
    parser.add_argument(
        "--chain-samples",
        type=int,
        default=8,
        help="Number of Markov chains to display (1-8)",
    )
    parser.add_argument("--chain-steps", type=int, default=2000, help="Total Markov chain steps")
    parser.add_argument("--chain-interval", type=int, default=500, help="Snapshot interval for the chain plot")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where plot images are written",
    )
    parser.add_argument(
        "--recon-output",
        type=str,
        default="plot_reconstruction.png",
        help="Filename for the reconstruction plot",
    )
    parser.add_argument(
        "--markov-output",
        type=str,
        default="plot_markov_chain.png",
        help="Filename for the Markov chain plot",
    )
    return parser


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str) -> torch.device:
    pref = pref.lower()
    if pref == "cuda" and torch.cuda.is_available():
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


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def clamp_samples(count: int) -> int:
    return max(1, min(8, count))


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not Path(args.checkpoint).is_file():
        parser.error(f"Checkpoint '{args.checkpoint}' not found")

    set_seed(args.seed)
    device = get_device(args.device)
    _, _, test_loader, input_dim = build_dataloaders(
        args.dataset, args.datadir, args.batch_size, need_train=False, need_valid=False, need_test=True
    )
    if test_loader is None:
        raise RuntimeError("Test loader unavailable for selected dataset; cannot plot reconstructions.")

    model = build_model(args, input_dim, device)
    load_checkpoint(model, args.checkpoint, device)

    output_dir = Path(args.output_dir)
    recon_samples = clamp_samples(args.recon_samples)
    chain_samples = clamp_samples(args.chain_samples)
    chain_steps = max(1, args.chain_steps)
    chain_interval = max(1, args.chain_interval)

    recon_path = output_dir / args.recon_output
    save_reconstruction_plot(
        model,
        test_loader,
        device,
        recon_path,
        n_samples=recon_samples,
        title=f"Test reconstructions (n={recon_samples})",
    )

    markov_path = output_dir / args.markov_output
    save_markov_chain_plot(
        model,
        device,
        markov_path,
        start_samples=chain_samples,
        steps=chain_steps,
        interval=chain_interval,
    )

    print(f"Saved plots to: {recon_path} and {markov_path}")


if __name__ == "__main__":
    main()
