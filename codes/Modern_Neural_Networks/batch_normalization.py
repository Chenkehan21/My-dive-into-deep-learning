import torch
import torch.nn as nn

def batch_norm(x, moving_mean, moving_var, epsilon, gamma, beta, momentum):
    if not torch.is_grad_enabled():
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + epsilon)
    else:
        assert x.dim() in (2, 4)
        if x.dim() == 2:
            mean = torch.mean(x, dim=0, keepdim=True)
            var = torch.mean(x**2, dim=0) - mean**2
        elif x.dim() == 4:
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            var = torch.mean(x**2, dim=(0, 2, 3), keepdim=True) - mean**2 + epsilon
        x_hat = (x - mean) / torch.sqrt(var)
        moving_mean = moving_mean * momentum + (1 - momentum) * mean
        moving_var = moving_var * momentum + (1 - momentum) * var
    y = gamma * x_hat + beta
        
    return y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_features, dim) -> None:
        super().__init__()
        if dim == 2:
            input_shape = (1, num_features)
        elif dim == 4:
            input_shape = (1, num_features, 1, 1)
        self.gamma = torch.ones(input_shape)
        self.beta = torch.zeros(input_shape)
        
        self.moving_mean = torch.zeros(input_shape)
        self.moving_var = torch.ones(input_shape)
        
        self.epsilon = 1e-5
        self.momentum = 0.9
        
    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
            self.gamma = self.gamma.to(x.device)
            self.beta = self.beta.to(x.device)
        y, self.moving_mean, self.moving_var = batch_norm(x, 
                                                          self.moving_mean, self.moving_var, 
                                                          self.epsilon, self.gamma, self.beta,
                                                          self.momentum)

        return y