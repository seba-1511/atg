#!/usr/bin/env python3

import torch
import tqdm

N_CLASSES = 100
N_FEATURES = 512
TRAIN_PERC = 0.6
T = 1.0


def main(
        lr=0.1,
        lmbd=1.0,
        num_iterations=7000,
        rate=0.32,
):
    """
    Example implementation of ATG.

    Replace the `class_embeddings` variable with the mean class
    embedding as described in the paper.
    """
    torch.manual_seed(1234)
    class_embeddings = torch.randn(N_CLASSES, N_FEATURES)
    class_embeddings = torch.nn.functional.normalize(class_embeddings, dim=1)
    mu_train = torch.randn(1, N_FEATURES, requires_grad=True)
    mu_test = torch.randn(1, N_FEATURES, requires_grad=True)

    # optimize mu_train and mu_test
    optimizer = torch.optim.SGD([mu_train, mu_test], lr=lr, momentum=0.9)
    for step in tqdm.trange(num_iterations):
        optimizer.zero_grad()

        # compute loglikelihoods
        train_distances = (class_embeddings - mu_train).pow(2).sum(dim=1, keepdim=True)
        p_train = torch.nn.functional.softmax(-train_distances * T, dim=0)
        log_p_train = torch.nn.functional.log_softmax(-train_distances * T, dim=0)

        test_distances = (class_embeddings - mu_test).pow(2).sum(dim=1, keepdim=True)
        p_test = torch.nn.functional.softmax(-test_distances * T, dim=0)
        log_p_test = torch.nn.functional.log_softmax(-test_distances * T, dim=0)

        nll = - torch.logaddexp(log_p_train, log_p_test).sum()  # discard 0.5 factor

        # compute divergences
        D_train_test = torch.sum(p_train * (log_p_train - log_p_test))
        D_test_train = torch.sum(p_test * (log_p_test - log_p_train))
        distance = D_train_test + D_test_train
        penalty = (distance - rate).pow(2).mul(lmbd)

        criterion = nll + penalty
        criterion.backward()
        optimizer.step()

    print(f'KL: {distance:.2f}')
    print(f'Target KL: {rate:.2f}')

    # assign class to partitions
    log_ratios = (log_p_train - log_p_test).view(-1).tolist()
    classes = list(range(N_CLASSES))
    ratio_class = list(zip(log_ratios, classes))
    ratio_class = sorted(ratio_class, key=lambda x: x[0])

    n_train = int(N_CLASSES * TRAIN_PERC)
    sorted_classes = [rc[1] for rc in ratio_class]
    train_classes = set(sorted_classes[-n_train:])
    test_classes = set(sorted_classes[0:-n_train:2])
    valid_classes = set(sorted_classes[1:-n_train:2])

    print(f'Train classes: {train_classes}')
    print(f'Validation classes: {valid_classes}')
    print(f'Test classes: {test_classes}')

    assert len(train_classes) + len(valid_classes) + len(test_classes) == N_CLASSES
    assert len(train_classes & test_classes) == 0
    assert len(train_classes & valid_classes) == 0
    assert len(test_classes & valid_classes) == 0


if __name__ == '__main__':
    main()
