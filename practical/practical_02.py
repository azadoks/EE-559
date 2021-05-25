import dlc_practical_prologue as prologue

train_input, train_target, test_input, test_target = prologue.load_data()


def nearest_classification(train_input, train_target, x):
    """Find the class of the training sample which is the closest to `x` with the L2 norm.

    :param train_input: 2D (n x d) float Tensor containing training vectors
    :param train_target: 1D (n) Tensor containing training labels
    :param x: 1D (d) float Tensor containing the test vector
    :returns: the label from `train_target` for the `train_input` closest to `x` with the L2 norm
    """
    dist = (train_input - x).pow(2).sum(1)
    return train_target[dist.argmin()]


# pylint: disable=too-many-arguments
def compute_nb_errors(train_input,
                      train_target,
                      test_input,
                      test_target,
                      mean=None,
                      proj=None):
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
        train_input = train_input.mm(proj.t())
        test_input = test_input.mm(proj.t())

    nb_errors = 0
    for i in range(test_input.size(0)):
        if test_target[i] != nearest_classification(train_input, train_target,
                                                    test_input[i]):
            nb_errors += 1

    return nb_errors


def PCA(x):
    """Compute the PCA mean and basis vectors.
    :param x: 2D (n x d) float Tensor containing training vectors
    :returns: mean of x and PCA basis vectors
    """
    mean = x.mean()
    b = x - mean
    eigvals, eigvecs = (b.t() @ b).eig(eigenvectors=True)
    pca_basis = eigvecs.t()[eigvals[:, 0].abs().sort(descending=True).indices]
    return mean, pca_basis


if __name__ == '__main__':
    def main():
        for c in [False, True]:
            train_input, train_target, test_input, test_target = prologue.load_data(
                cifar=c)
            ## Baseline errors
            nb_errors = compute_nb_errors(train_input, train_target,
                                          test_input, test_target)
            print('Baseline nb_errors {:d} error {:.02f}%'.format(
                nb_errors, 100 * nb_errors / test_input.size(0)))
            ## Random errors
            basis = train_input.new(100, train_input.size(1)).normal_()
            nb_errors = compute_nb_errors(train_input, train_target,
                                          test_input, test_target, None, basis)
            print('Random {:d}d nb_errors {:d} error {:.02f}%'.format(
                basis.size(0), nb_errors,
                100 * nb_errors / test_input.size(0)))
            ## PCA errors
            mean, basis = PCA(train_input)
            for d in [100, 50, 10, 3]:
                nb_errors = compute_nb_errors(train_input, train_target,
                                              test_input, test_target, mean,
                                              basis[:d])
                print('PCA {:d}d nb_errors {:d} error {:.02f}%'.format(
                    d, nb_errors, 100 * nb_errors / test_input.size(0)))

    main()
