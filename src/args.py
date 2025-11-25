"""input cli args for train.py"""

import argparse

def parse_architecture(spec: str):
    """comma-separated encoder/decoder layers ("500,500") into a list of ints."""
    spec = (spec or "").strip()
    if not spec:
        return []
    return [int(part.strip()) for part in spec.split(",") if part.strip()]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training NECST-Torch")

    # data + environment
    parser.add_argument("--datadir", type=str, default="./data", help="Root directory containing datasets")
    parser.add_argument(
        "--dataset",
        type=str.lower,
        default="binarymnist",
        help="Dataset to use (binarymnist, random_bits, omniglot)",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Mini-batch size")
    parser.add_argument(
        "--device",
        type=str.lower,
        default="cpu",
        help="Device to use: cuda or cpu (defaults to cpu when available)",
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")

    # model + latent space
    parser.add_argument("--latent-dim", type=int, default=100, help="Number of latent bits (z-dim)")
    parser.add_argument("--enc-arch", type=str, default="500", help="Comma-separated encoder layer sizes")
    parser.add_argument("--dec-arch", type=str, default="500,500", help="Comma-separated decoder layer sizes")
    parser.add_argument("--vimco-samples", type=int, default=5, help="Number of samples when using VIMCO")
    parser.add_argument("--noise", type=float, default=0.0, help="Channel noise level during training")
    parser.add_argument("--test-noise", type=float, default=0.0, help="Channel noise level during evaluation") 

    # optimization options
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--optimizer",
        type=str.lower,
        default="adam",
        choices=["adam", "sgd", "momentum"],
        help="Optimizer",
    )
    parser.add_argument(
        "--loss-type",
        type=str.lower,
        default="vimco",
        choices=["gumbel", "vimco"],
        help="Which loss/estimator to train with",
    )
    parser.add_argument("--reg-param", type=float, default=1e-3, help="L2 regularization strength")
    parser.add_argument("--temperature", type=float, default=0.67, help="Gumbel-Softmax initial temperature")
    parser.add_argument("--gumbel-hard", action="store_true", help="Use hard straight-through Gumbel samples")
    parser.add_argument("--temp-anneal", action="store_true", help="Enable linear temperature annealing for Gumbel-Softmax")
    parser.add_argument("--temp-final", type=float, default=0.1, help="Final temperature for annealing (default: 0.1)")

    # checkpointing
    parser.add_argument(
        "--save-path", type=str, default="models/necst.pt", help="Where to save the trained model"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for evaluation or fine-tuning")
    
    # training loss plot
    parser.add_argument(
        "--plot-loss",
        action="store_true",
        help="Plot train/validation loss curves to plots/training_loss.png",
    )

    return parser

def parse_args(args=None):
    parser = build_parser()
    return parser.parse_args(args=args)
