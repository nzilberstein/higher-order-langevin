# Solving Linear Inverse Problems using Higher-Order Annealed Langevin Diffusion

This repo contains the official implementation of the "Solving Linear Inverse Problems using Higher-Order Annealed Langevin Diffusion" paper (arXiv.2205.05776, link: https://arxiv.org/pdf/2305.05014.pdf) 

## Description


The repo contains one folder for each type of experiment.

* channel-experiments

This folder contains all the files and folders to run the channel estimation experiments. By running the experiments you will get something like the following

<img src="https://github.com/nzilberstein/joint-score-based-channel/blob/main/figures/discretization_methods.png" width="500" height="425">
<img src="https://github.com/nzilberstein/joint-score-based-channel/blob/main/figures/langevin_order_comparison.png" width="500" height="425">


* Image experiments

This folder contains all the files and folders to run the image experiments

## Clone the repository 

```
cd <folder>
git clone https://github.com/nzilberstein/higher-order-langevin.git
```

The package installation as well the inference process is explained for each case

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

