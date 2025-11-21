# necst-torch implementation 

Final project for CS274E @ UCI. We reimplement the NECST generative model with Gumbel-Softmax and VIMCO estimators for discrete latent variables.

## References

- Original paper: https://arxiv.org/abs/1811.07557.
- Reference implementation (TensorFlow): https://github.com/ermongroup/necst.

**Short description:** NECST is a neural joint source–channel coder that compresses images into binary codes and decodes them robustly under channel noise. It trains end-to-end with VIMCO and continuous relaxations for discrete bits, outperforming traditional separated source/error-correction baselines at comparable rates.

## Getting started

All commands below are run from the `necst-torch/` directory. 

```bash
# optional: create and activate an isolated environment with conda
conda create -n necst python=3.12 -y
conda activate necst

# install dependencies // ensure correct PyTorch w/ cuda: https://pytorch.org/get-started/locally/.
pip install -r requirements.txt

# download data
python data_setup/download.py binarymnist
python data_setup/download.py random_bits --n-bits 256
python data_setup/download.py omniglot
```

## Model training

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
  --save-path models/mnist_vimco.pt \
  --plot-loss
```

### Outputs

TBD.


### Group Members:
1. XXX
2. XXX
3. XXX