import torch
import torch.nn as nn


def conv2d(x, k):
    h_in, w_in= x.shape
    h_k, w_k = k.shape
    h_out, w_out = h_in - h_k + 1, w_in - w_k + 1
    res = torch.zeros(h_out, w_out)
    for i in range(h_out):
        for j in range(w_out):
            res[i, j] = (x[i : i + h_k, j : j + w_k] * k).sum()
    
    return res

def conv2d_multi_channel(batch_images, kernals):
    res = []
    for image in batch_images:
        kernal_stacks = []
        for kernal in kernals:
            channel_stacks = []
            for channel_img, channel_k in zip(image, kernal):
                channel_stacks.append(conv2d(channel_img, channel_k))
            kernal_stacks.append(sum(channel_stacks))
        res.append(torch.stack(kernal_stacks))
    return torch.stack(res)


if __name__ == "__main__":
    data = torch.ones(1, 3, 5, 5)
    k = torch.zeros(2, 3, 3, 3)
    k[0, :, :, :] = 0
    k[1, :, :, :] = 1
    res = conv2d_multi_channel(data, k)
    print(data)
    print(k)
    print(res, res.shape)