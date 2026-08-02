

# Implementación de necst-torch 

Proyecto final para CS274E @ UCI (OTOÑO 2025). Reimplementamos el modelo generativo NECST con Gumbel-Softmax y estimadores VIMCO para variables latentes discretas.

## Referencias

- Artículo original: https://arxiv.org/abs/1811.07557.
- Implementación de referencia (TensorFlow): https://github.com/ermongroup/necst.

**Descripción rápida:** NECST es un codificador neural conjunto fuente-canal que comprime imágenes en códigos binarios latentes y los decodifica bajo ruido de canal. Este modelo generativo de extremo a extremo supera a las líneas base clásicas de pipelines separados/corrección de errores para longitudes de bits similares.

## Primeros pasos

Todos los comandos a continuación se ejecutan desde el directorio `necst-torch/`. 

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

## Entrenamiento del modelo

Ejemplo de ejecución de entrenamiento utilizando la pérdida VIMCO en datos MNIST:

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

### Miembros del grupo:
1. Aditya Mittal
2. Nick Du
3. ZhaoBin Li
