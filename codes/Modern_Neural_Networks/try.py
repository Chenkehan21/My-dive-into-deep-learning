import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([64, 96, 26, 26], dtype=torch.half, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(96, 256, kernel_size=[5, 5], padding=[2, 2], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().half()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()
print("done")