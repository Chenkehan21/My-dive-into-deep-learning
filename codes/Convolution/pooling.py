import torch
import torch.nn as nn


def pooling(x, pool_size, mode='max'):
    hstride, wstride = pool_size
    h_out, w_out = x.shape[0] - hstride + 1, x.shape[1] - wstride + 1
    res = torch.zeros(h_out, w_out)
    h, w = x.shape
    for i in range(0, h_out):
        for j in range(0, w_out):
            if mode == 'max':
                res[i, j] = x[i : i + hstride, j : j + wstride].max()
            elif mode == 'mean':
                res[i, j] = x[i : i + hstride, j : j + wstride].mean()
    
    return res


if __name__ == '__main__':
    data = torch.randn(5, 5)
    res = pooling(data, (3, 3))
    print(data)
    print(res)