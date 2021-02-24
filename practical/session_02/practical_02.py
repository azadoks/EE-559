import sys
sys.path.append('../')

import torch
import prologue_cli as prologue

train_input, train_target, test_input, test_target = prologue.load_data()

def nearest_classification(train_input, train_target, x):
    """Find the class of the training sample which is the closest to `x` with the L2 norm.

    :param train_input: 2D (n x d) float Tensor containing training vectors
    :param train_target: 1D (n) Tensor containing training labels
    :param x: 1D (d) float Tensor containing the test vector
    :returns: the label from `train_target` for the `train_input` closest to `x` with the L2 norm
    """
    dist = torch.sum((train_input - x).pow(2), dim=1).sqrt_()

    return train_target[dist.argmin()]


def compute_nb_errors(train_input, train_target, test_input, test_target,
    mean=None, proj=None):
    """Return the number of classification errors using 1-nearest-neighbor analysis.

    :param train_input: 2D (n x d) float Tensor containing training vectors
    :param train_target: 1D (n) Tensor containing training labels
    :param test_input: 2D (m x d) float Tensor containing test vectors
    :param test_target: 1D (m) Tensor contianing test labels
    :param mean: None or 1D (d) float Tensor
    :param proj: None or 2D (c x d) float Tensor
    :returns: number of classification errors
    """

    if mean is not None:
        train_input = train_input.sub(mean)
        test_input = test_input.sub(mean)
    
    if proj is not None:
        train_input = train_input.mm(proj)
        test_input = test_input.mm(proj)

    test_class = torch.empty(test_target.size())
    for i, test_i in enumerate(test_input):
        dist = torch.sum((test_input - test_i).pow(2), dim=1).sqrt_()
        test_class[i] = train_target[dist.argmin()]

    return torch.sum(test_class == test_target)


if __name__ == '__main__':
    N_RUNS = 100
    D_DIM = train_input.size()[1]

    mismatches = 0
    for i in range(N_RUNS):
        nc = nearest_classification(train_input, train_target, test_input[i])
        if (nc.item() - test_target[i].item()) != 0:
            mismatches += 1
    print(f'`nearest_classification: {mismatches} mismatches')

    cnbe = compute_nb_errors(train_input, train_target, test_input, test_target)
    print(f'`nb_classification_errors`: {cnbe}')

    cnbe = compute_nb_errors(train_input, train_target, test_input, test_target,
        mean=torch.empty(D_DIM).normal_())
    print(f'`nb_classification_errors` (mean): {cnbe}')

    # cnbe = compute_nb_errors(train_input, train_target, test_input, test_target,
    #     proj=torch.empty(100, D_DIM).normal_())
    # print(f'`nb_classification_errors` (proj): {cnbe}')

    # cnbe = compute_nb_errors(train_input, train_target, test_input, test_target,
    #     mean=torch.empty(D_DIM).normal_(), proj=torch.empty(100, D_DIM).normal_())
    # print(f'`nb_classification_errors` (mean, proj): {cnbe}')