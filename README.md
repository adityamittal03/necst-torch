# necst-torch implementation 

Final project for CS274E @ UCI. We reimplement the NECST generative model with Gumbel-Softmax and VIMCO estimators for discrete latent variables.

## References

- Original paper: https://arxiv.org/abs/1811.07557.
- Reference implementation (TensorFlow): https://github.com/ermongroup/necst.

## Getting started

All commands below are run from the `necst-torch/` directory. Feel free to use Conda or any other environment manager; the snippet below uses the built-in `venv`.

```bash
# optional: create and activate an isolated environment with conda
conda create -n necst python=3.12 -y
conda activate necst

# install dependencies
pip install -r requirements.txt

# download Binary MNIST (binarized torchvision MNIST) or generate random bit datasets
python data_setup/download.py BINARYMNIST
python data_setup/download.py random_bits --n-bits 100
```

## Training

```bash
python src/train.py \
  --dataset binarymnist \
  --datadir data \
  --device cuda \
  --batch-size 100 \
  --epochs 200 \
  --loss-type vimco \
  --latent-dim 100 \
  --vimco-samples 5 \
  --lr 1e-3 \
  --noise 0.1 \
  --save-path models/mnist_vimco_0.1.pt \
  --plot-loss
```

### Outputs

- **Model weights**: saved to `models/necst.pt` (configurable via `--save-path`).
- **Plots**: stored in `results/`  
  - `training_loss.png` (enabled by default; disable via `--no-plot-loss`)  
  - `reconstruction.png` (enable with `--plot-reconstruction`)  
  - `markov_chain.png` (enable with `--plot-markov`)

Validation/test BCE metrics are printed during training. Use `python src/test.py --checkpoint models/necst.pt` to evaluate a saved model on the test split.
