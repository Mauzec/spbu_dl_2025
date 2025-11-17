import torch
from num1 import (
    CustomLinear,
    CustomBatchNorm,
    CustomReLU,
    CustomDropout,
    CustomSoftmax,
)

from pathlib import Path
import sys
print("cwd:", Path.cwd())
print("sys.path[0]:", sys.path[0])

in_dim = 8
hidden = 16
num_classes = 4

lin = CustomLinear(in_dim, hidden)
bn = CustomBatchNorm(hidden)
relu = CustomReLU()
dropout = CustomDropout(p=0.5)
softmax = CustomSoftmax(dim=1)
lin2 = CustomLinear(hidden, num_classes)

def forward(x: torch.Tensor) -> torch.Tensor:
    x = lin(x)
    x = bn(x)
    x = relu(x)
    x = dropout(x)
    x = lin2(x)
    x = softmax(x)
    return x

def collect_params(*layers):
    params = []
    for layer in layers:
        for attr in vars(layer).values():
            if isinstance(attr, torch.Tensor) and attr.requires_grad:
                params.append(attr)
    return params

params = collect_params(lin, bn, lin2)
print(f'params size:', [p.shape for p in params])

device = torch.device("cpu")
for p in params:
    p.data = p.data.to(device)
    p.requires_grad_(True)
    

def move_layer_to(l):
    for n, v in list(vars(l).items()):
        if isinstance(v, torch.Tensor):
            setattr(l, n, v.to(device))

move_layer_to(lin)
move_layer_to(bn)
move_layer_to(lin2)

optim = torch.optim.SGD(params, lr=.01)

N = 64
X = torch.randn(N, in_dim, device=device)
y = torch.randint(0, num_classes, (N,), device=device)

def nll_from_soft(preds, labels, eps=1e-8):
    probs = preds.clamp(eps, 1.0 - eps)
    idx = labels.view(-1, 1)
    picked = probs.gather(1, idx).squeeze(1)
    return -torch.log(picked).mean()

for epoch in range(120):
    bn.training = True
    dropout.training = True
    
    optim.zero_grad()
    preds = forward(X)
    loss = nll_from_soft(preds, y)
    loss.backward()
    optim.step()
    
    bn.training = False
    dropout.training = False
    with torch.no_grad():
        val_preds = forward(X)
        acc = (val_preds.argmax(dim=1) == y).float().mean().item()
        print(f'{epoch:2d} loss={loss.item():.4f} acc={acc:.4f}')
        
    
    