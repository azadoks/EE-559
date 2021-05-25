#/usr/bin/python

import time

import torch


def multiple_views():
    # 13x13 tensor filled with ones
    m = torch.full((13, 13), 1, dtype=torch.int)
    # Rows of twos starting with second row, every five rows
    m[1::5, :] = 2
    # Columns of twos starting with second column, every five columns
    m[:, 1::5] = 2
    # 2x2 areas of threes at intersections of (fourth and fifth) and (ninth and tenth) rows and columns
    m[3:5, 3:5] = 3
    m[3:5, 8:10] = 3
    m[8:10, 3:5] = 3
    m[8:10, 8:10] = 3

    return m


def eigendecomposition(size):
    # square diagonal matrix, note: float-type (1.) (D)
    diag_m = torch.diag(torch.arange(1., size + 1.))
    # square random normal matrix (M)
    gauss_m = torch.empty((size, size)).normal_()  
    # sort(eigvals(M^-1 D M))
    eigvals, _ = gauss_m.inverse().mm(diag_m).mm(gauss_m).eig()[0].sort(dim=0)
    
    return eigvals


def flops(size):
    # Generate matrices
    m1 = torch.empty((size, size)).normal_()
    m2 = torch.empty((size, size)).normal_()
    # Record initial and final time for multiplication
    t0 = time.perf_counter()
    _ = torch.mm(m1, m2)
    t1 = time.perf_counter()
    # Matrix multipliciation of (A @ B) takes (A_ncol * A_nrow * B_nrow) operations
    n_ops = size ** 3
    flops = n_ops / (t1 - t0)

    return flops


def mul_row(m):
    """Multiply rows of a matrix by their 1-based index using a Python loop."""
    result = m.clone()
    for i in range(m.size(0)):
        for j in range(m.size(1)):
            result[i, j] *= i + 1
    return result


def mul_row_fast(m):
    """Multiply rows of a matrix by their 1-based index using PyTorch."""
    c = torch.arange(1, m.size(0) + 1).view(-1, 1)  # a `column`; c[i] = i + 1
    return m.mul(c)


def time_function(f, *args, **kwargs):
    t0 = time.perf_counter()
    _ = f(*args, **kwargs)
    t1 = time.perf_counter()

    return (t1 - t0)


if __name__ == '__main__':
    def main():
        eigen_size = 20
        flops_size = 5000
        mul_m = torch.full((1000, 400), 2.0)
        print('MULTIPLE VIEWS\n' + ''.join(['-'] * 80))
        print(multiple_views(), '\n')
        print('EIGENDECOMPOSITION\n' + ''.join(['-'] * 80))
        print(eigendecomposition(eigen_size), '\n')
        print('FLOPS\n' + ''.join(['-'] * 80))
        print(f'{flops(flops_size):0.4e} [FLOPS]\n')
        print('PLAYING WITH STRIDES\n' + ''.join(['-'] * 80))
        print(f'mul_row: {time_function(mul_row, mul_m):0.6e} [s]')
        print(f'mul_row_fast: {time_function(mul_row_fast, mul_m):0.6e} [s]')
    
    main()