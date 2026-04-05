import torch
from torch.cuda.amp import autocast
x = torch.zeros(2, requires_grad=True).cuda()
with autocast():
    l = torch.norm(x, p='fro')
l.backward()
print(x.grad)
