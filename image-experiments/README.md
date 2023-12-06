# MIMO channel estimation - experiments 

This repo contains the official implementation of the "Solving Linear Inverse Problems using Higher-Order Annealed Langevin Diffusion" paper (arXiv.2205.05776, link: https://arxiv.org/pdf/2305.05014.pdf) 

## Description

The code in this folder implements higher-order Langevin with score-based prior (pre-trained diffusion model) for MIMO channel estimation. We implement the first, second and third order Langevin dynamics using splitting techinques (for second and third). 

## Preparing the environment and running inference

### Dependencies

Run the following conda line to install all necessary python packages for our code and set up the snips environment.

```bash
conda env create -f environment.yml
```

The environment includes `cudatoolkit=11.0`. You may change that depending on your hardware.

Alternatively, you can install the packages by running 

```
pip install -r requirements.txt
```

### Downloading data

You can download as follow

* FFHQ data from the drive in [NCSNv2] https://drive.google.com/drive/folders/1217uhIvLg9ZrYNKOR3XTRFSurt4miQrd.
* The aligned and cropped CelebA files from their official source [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 
* The LSUN files can be downloaded using [this script](https://github.com/fyu/lsun).

The experiments in the paper are with FFHQ data. Please, download the .zip file and place in `exp/dataset`.

### Running inference

If we want to run 3rd order Langevin on FFHQ for the problem of super resolution by 2, with added noise of standard deviation 0.1, and obtain 1 variations, we can run the following

```bash
python main.py -i ffhq --config ffhq.yml --doc ffhq -n 1 --degradation sr2 --sigma_0 0.1 --order_lang 3rd
```

Samples will be saved in `<exp>/image_samples3rd/ffhq`.

The available degradations are: Inpainting (`inp`), Uniform deblurring (`deblur_uni`), Gaussian deblurring (`deblur_gauss`), , Gaussian deblurring anisotropic (`deblur_gauss_aniso`), Super resolution by 2 (`sr2`) or by 4 (`sr4`), Compressive sensing by 4 (`cs4`), 8 (`cs8`), or 16 (`cs16`). The sigma_0 can be any value from 0 to 1.

## Pretrained Checkpoints

These checkpoint files are provided as-is from the authors of [NCSNv2](https://github.com/ermongroup/ncsnv2), and you can downoad from this link Link: https://drive.google.com/drive/folders/1217uhIvLg9ZrYNKOR3XTRFSurt4miQrd?usp=sharing.
You can use the FFHQ, CelebA, LSUN-bedroom, and LSUN-tower datasets' pretrained checkpoints. We assume the `--exp` argument is set to `exp`.

Please, place the checkpoint in `exp/logs/ffhq/`


## Project structure

`main.py` is the file that you should run for sampling. Execute ```python main.py --help``` to get its usage description:

```
usage: main.py [-h] --config CONFIG [--seed SEED] [--exp EXP] --doc DOC
               [--comment COMMENT] [--verbose VERBOSE] [-i IMAGE_FOLDER]
               [-n NUM_VARIATIONS] [-s SIGMA_0] [--degradation DEGRADATION] 
               [--order_lang LANGEVIN_TYPE]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file
  --seed SEED           Random seed
  --exp EXP             Path for saving running related data.
  --doc DOC             A string for documentation purpose. Will be the name
                        of the log folder.
  --comment COMMENT     A string for experiment comment
  --verbose VERBOSE     Verbose level: info | debug | warning | critical
  -i IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The folder name of samples
  -n NUM_VARIATIONS, --num_variations NUM_VARIATIONS
                        Number of variations to produce
  -s SIGMA_0, --sigma_0 SIGMA_0
                        Noise std to add to observation
  --degradation DEGRADATION
                        Degradation: inp | deblur_uni | deblur_gauss | sr2 |
                        sr4 | cs4 | cs8 | cs16
  --order_lang LANGEVIN_TYPE
                        Type: 1st, 2nd, 3rd

```

Configuration files are in `configs/`. You don't need to include the prefix `configs/` when specifying  `--config` . All files generated when running the code is under the directory specified by `--exp`. They are structured as:

```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files
│   ├── celeba # all CelebA files
│   └── lsun # all LSUN files
│   └── FFHQ # all FFHQ files
├── logs # contains checkpoints and samples produced during training
│   └── <doc> # a folder named by the argument `--doc` specified to main.py
│      └── checkpoint_x.pth # the checkpoint file saved at the x-th training iteration
├── image_samples # contains generated samples
│   └── <i>
│       ├── stochastic_variation.png # samples generated from checkpoint_x.pth, including original, degraded, mean, and std   
│       ├── results.pt # the pytorch tensor corresponding to stochastic_variation.png
│       └── y_0.pt # the pytorch tensor containing the input y of SNIPS
```

## Acknowledgement

This repo is largely based on the [SNIPS](https://github.com/bahjat-kawar/snips_torch) and [NCSNv2](https://github.com/ermongroup/ncsnv2) repos, and uses modified code from [DDRM](https://github.com/bahjat-kawar/ddrm) for implementing the degradation operations.

## References

If you find the code/idea useful for your research, please consider citing

```bib
@article{zilberstein2023solving,
  title={Solving Linear Inverse Problems using Higher-Order Annealed Langevin Diffusion},
  author={Zilberstein, Nicolas and Sabharwal, Ashutosh and Segarra, Santiago},
  journal={arXiv preprint arXiv:2305.05014},
  year={2023}
}
```

