# Embedding Adaptation is Still Needed for Few-Shot Learning

[![arXiv](https://img.shields.io/badge/arXiv-2101.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2101.XXXXX)

Code Release for "Embedding Adaptation is Still Needed for Few-Shot Learning"

This code provides:

* Re-implementation of the ATG algorithm in `examples/atg.py`.
* Loaders for the dataset splits introduced in the paper.
* Demonstration code for training the algorithms, borrowed from [learn2learn](https://github.com/learnables/learn2learn).

## Resources

* Website:
* Preprint:
* Code:

## Citation

Please cite this work as follows:

> "Embedding Adaptation is Still Needed for Few-Shot Learning", Sébastien M. R. Arnold and Fei Sha

or with the following BibTex entry:

~~~bibtex
@article{arnold2021embedding,
    title={Embedding Adaptation is Still Needed for Few-Shot Learning},
    author={Sebastien M. R. Arnold, Fei Sha},
    year={2021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
~~~

## Usage

~~~bash
python examples/train.py --algorithm='protonet' --dataset='mini-imagenet' --taskset='original'
~~~

where

* `taskset` takes values `easy`, `medium-easy`, `medium-hard`, `hard` or `randomX` where `X` is the seed to reproduce random splits.
* `dataset` takes values `mini-imagenet`, `tiered-imagenet`, `emnist`, `lfw10`, `cifar100`.
* `algorithm` takes values ``, ``, ``, ``, ``, ``.

For more help on the interface, run: `python examples/train.py --help`.
