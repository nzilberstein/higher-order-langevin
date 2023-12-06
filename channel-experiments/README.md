# MIMO channel estimation - experiments 

This repo contains the official implementation of the "Solving Linear Inverse Problems using Higher-Order Annealed Langevin Diffusion" paper (arXiv.2205.05776, link: https://arxiv.org/pdf/2305.05014.pdf) 

## Description

The code in this folder implements higher-order Langevin with score-based prior (pre-trained diffusion model) for MIMO channel estimation. We implement the first, second and third order Langevin dynamics using splitting techinques (for second and third).

## Preparing the environment and running inference

### Install packages

Create a clean environment and then you can install the packages by running 

```
pip install -r requirements.txt
```

### Downloading data

In our experiments, we only used CDL-C. 
The training and validation data for CDL-C channels can be directly downloadedfrom the command line using the following:

```
mkdir data
curl -L https://utexas.box.com/shared/static/nmyg5s06r6m2i5u0ykzlhm4vjiqr253m.mat --output ./data/CDL-C_Nt64_Nr16_ULA0.50_seed1234.mat
curl -L https://utexas.box.com/shared/static/2a7tavjo9hk3wyhe9vv0j7s2l6en4mj7.mat --output ./data/CDL-C_Nt64_Nr16_ULA0.50_seed4321.mat
```
Once downloaded, place these files in the `data` folder under the main directory.

### Pretrained Checkpoints

A pre-trained diffusion model for CDL-C channels can be directly downloaded from the command line using the following:
```
mkdir models/score/CDL-C
curl -L https://utexas.box.com/shared/static/4nubcpvpuv3gkzfk8dgjo6ay0ssps66w.pt --output ./models/score/CDL-C/final_model.pt
```

This will create the nested directories `models/score/CDL-C` and place the weights there. Weights for models trained on other distributions (CDL-A, CDL-B, CDL-D, Mixed) shown in the paper can be downloaded from the following public repository:

https://utexas.box.com/s/m58udx6h0glwxua88zgdwrff87jvy3qw

Once downloaded, places these files in their matching directory structure as `final_model.pt`.


### Running inference

We only considered CDL-C channel models. If we want to run 3rd order Langevin with BACOCAB with spacing of 40 levels of noise (i.e., skip 40 levels), we can run the following
```
python test_score.py -d BACOCAB --spacing_classes 40
```

This will generate a file placed in results. Then, you can plot using the function in `generative_plots.ipynb`

## Acknowledgement

Full credits for this repository go to the [MIMO-score-based]: https://github.com/utcsilab/score-based-channels.


