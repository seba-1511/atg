# Embedding Adaptation is Still Needed for Few-Shot Learning

[![arXiv](https://img.shields.io/badge/arXiv-2101.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2101.XXXXX)

Code Release for "Embedding Adaptation is Still Needed for Few-Shot Learning"

This code provides:

* Re-implementation of the ATG algorithm in `examples/atg.py`.
* Loaders for the dataset splits introduced in the paper.
* Demonstration code for training the algorithms, borrowed from [learn2learn](https://github.com/learnables/learn2learn).

## Resources

* Website: [seba1511.net/projects/atg](seba1511.net/projects/atg)
* Preprint: [arxiv.org/abs/2101.XXXXX](https://arxiv.org/abs/2101.XXXXX)
* Code: [github.com/Sha-Lab/atg](https://github.com/Sha-Lab/atg)

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

Dependencies include the following Python packages:

* PyTorch>=1.3.0
* torchvision>=0.5.0
* scikit-learn>=0.19.2
* tqdm>=4.48.2
* learn2learn on the master branch

### Running ATG

A standalone re-implementation of ATG is provided in `examples/atg.py`. To run it on a synthetic dataset:

```bash
python examples/atg.py
```

### Training on ATG Partitions

~~~bash
python examples/train.py --algorithm='protonet' --dataset='mini-imagenet' --taskset='original'
~~~

where

* `taskset` takes values `easy`, `medium-easy`, `medium-hard`, `hard` or `randomX` where `X` is the seed to reproduce random splits.
* `dataset` takes values `mini-imagenet`, `tiered-imagenet`, `emnist`, `lfw10`, `cifar100`.
* `algorithm` takes values `protonet`, `maml`, `anil`.

For more help on the interface, run: `python examples/train.py --help`.
