# cs274e-project

Final project for CS274E. We reimplement the NECST generative model with Gumbel-Softmax and VIMCO estimators for discrete latent variables.

## References

- Original paper: _NECST: Neural Error-Correcting and Source-Tracing Codes_ (ICML 2018) — Valerio M. Sitzmann et al.
- Reference implementation (TensorFlow): https://github.com/vsitzmann/NECST

## Getting started

All commands below are run from the `necst-torch/` directory. Feel free to use Conda or any other environment manager; the snippet below uses the built-in `venv`.

```bash
# optional: create and activate an isolated environment with conda
conda create -n necst python=3.12 -y
conda activate necst

# install dependencies
pip install -r requirements.txt

# download Binary MNIST or generate random bit datasets
python data_setup/mnist.py BinaryMNIST
python data_setup/random_bits.py data/random_bits 100
```

## Training

```bash
python src/train.py \
  --dataset binarymnist \
  --device cuda \
  --epochs 200 \
  --loss-type vimco \
  --save-path models/necst.pt \
  --plot-reconstruction \
  --plot-markov
```

### Outputs

- **Model weights**: saved to `models/necst.pt` (configurable via `--save-path`).
- **Plots**: stored in `results/`  
  - `training_loss.png` (enabled by default; disable via `--no-plot-loss`)  
  - `reconstruction.png` (enable with `--plot-reconstruction`)  
  - `markov_chain.png` (enable with `--plot-markov`)

Validation/test BCE metrics are printed during training. Use `python src/test.py --checkpoint models/necst.pt` to evaluate a saved model on the test split.
