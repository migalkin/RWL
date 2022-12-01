# Weisfeiler and Leman Go Relational (LOG 2022) #

<p align="center">
<a href="https://openreview.net/pdf?id=wY_IYhh6pqj"><img src="http://img.shields.io/badge/OpenReview-PDF-red.svg" alt="LOG paper"></a>
<a href="arxiv TODO"><img src="http://img.shields.io/badge/arxiv-abs-red.svg" alt="arxiv"></a>
</p>

This is the official code base of the paper

[Weisfeiler and Leman Go Relational][paper]

[Pablo Barcelo](https://pbarcelo.ing.uc.cl/)
[Mikhail Galkin](https://migalkin.github.io),
[Christopher Morris](https://chrsmrrs.github.io/)
[Miguel Romero Orth](http://mromero.cl/)

[paper]: https://openreview.net/pdf?id=wY_IYhh6pqj

## Overview ##

This repo contains the code for reproducing the experiments on R-GCN and CompGCN with the one-hot feature initialization
strategy.

**Notice on the k-RN architecture:** We plan to update the repo with the k-RN implementation as soon as we come up with the meaningful relational dataset to evaluate k-RNs.

### Installation ###

The experiments were performed on Python 3.8.

Dependencies:
```
torch                 1.10.0
torch-cluster         1.5.9
torch-geometric       2.0.2
torch-scatter         2.0.9
torch-sparse          0.6.12
```

Optionally, install `wandb` for results tracking, prepend `WANDB_ENTITY=yourentity` to the running script and use the `--wandb` flag.



* To run experiments on the AIFB dataset with the fast version of R-GCN and 4d features:

```bash
python main.py --dataset AIFB --lr 0.001 --epochs 8001 --rgcn_fast --drop_bias --dim 4
```

* For the modified R-GCN with the additional MLP over aggregated node features:

```bash
python main.py --dataset AIFB --lr 0.001 --epochs 8001 --rgcn_fast --mod_rgcn --drop_bias --dim 4
```

* For CompGCN:

```bash
python main.py --dataset AIFB --lr 0.001 --epochs 8001 --compgcn --dim 4
```

* For CompGCN without directional updates:

```bash
python main.py --dataset AIFB --lr 0.001 --epochs 8001 --compgcn --dim 4 --compgcn_no_dir
```

* For CompGCN without relation updates:

```bash
python main.py --dataset AIFB --lr 0.001 --epochs 8001 --compgcn --dim 4 --compgcn_no_relupd
```

* For CompGCN without adjacency normalization:

```bash
python main.py --dataset AIFB --lr 0.001 --epochs 8001 --compgcn --dim 4 --no_norm
```

You can combine those flags for CompGCN as well.

Experiments on the big AM dataset are forced on a CPU due to the dataset size. 

Options for `--msg_func`: `transe`, `distmult`, `rotate`

Options for `--aggr_func`: `add`, `mean`

Please refer to the Appendix F in the paper for the full set of hyperparameters.

## Citation ##

If you find this project useful in your research, please cite the following paper

```bibtex
@inproceedings{
    barcelo2022weisfeiler,
    title={Weisfeiler and Leman Go Relational},
    author={Pablo Barcelo and Mikhail Galkin and Christopher Morris and Miguel Romero Orth},
    booktitle={Learning on Graphs Conference},
    year={2022},
    url={https://openreview.net/forum?id=wY_IYhh6pqj}
}
```