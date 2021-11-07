import torch
import torch.nn as nn


def conv2d(x, k, p, s):
    h_in, w_in= x.shape
    h_k, w_k = k.shape
    h_out, w_out = int((h_in - h_k + 2 * p + s) / s), int((w_in - w_k + 2 * p + s) / s)
    res = torch.zeros(h_out, w_out)
    x2 = torch.zeros(h_in + p*2, w_in + p*2)
    x2[p:-p, p:-p] = x
    for i in range(h_out):
        for j in range(w_out):
            # print(x2[i : i + h_k, j : j + w_k], i, j)
            res[i, j] = (x2[i : i + h_k, j : j + w_k] * k).sum()
    
    return res


if __name__ == "__main__":
    data = torch.ones(5, 5)
    k = torch.randn(3, 3)
    res = conv2d(data, k, 1, 1)
    print(res, res.shape)

    data2 = torch.ones(1, 1, 5, 5)
    conv = nn.Conv2d(1, 1, kernel_size = (3, 3), padding=(0, 1), stride=(1, 2))
    res2 = conv(data2)
    print(res2, res2.shape)