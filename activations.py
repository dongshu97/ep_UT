import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsifyBase(nn.Module):
    def __init__(self, sparse_ratio=0.2):
        super(SparsifyBase, self).__init__()
        self.sr = sparse_ratio
        self.preact = None
        self.act = None

    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()

        return hook

    def record_activation(self):
        self.register_forward_hook(self.get_activation())


class Sparsify1D(SparsifyBase):
    def __init__(self, sparse_ratio=0.2):
        super(Sparsify1D, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        k = int(self.sr * x.shape[1])
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        return comp * x
    def manual_grad(self, x):
        with torch.no_grad():
            k = int(self.sr * x.shape[1])
            topval = x.topk(k, dim=1)[0][:, -1]
            topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
            comp = (x >= topval).to(x)
        return comp

class Sparsify2D(SparsifyBase):
    def __init__(self, sparse_ratio=0.2):
        super(Sparsify2D, self).__init__()
        self.sr = sparse_ratio

        self.preact = None
        self.act = None

    def forward(self, x):
        layer_size = x.shape[2]*x.shape[3]
        k = int(self.sr*layer_size)
        tmpx = x.view(x.shape[0], x.shape[1], -1)
        topval = tmpx.topk(k, dim=2)[0][:,:,-1]
        topval = topval.expand(x.shape[2], x.shape[3], x.shape[0], x.shape[1]).permute(2,3,0,1)
        comp = (x>=topval).to(x)
        return comp*x

    def manual_grad(self, x):
        with torch.no_grad():
            layer_size = x.shape[2] * x.shape[3]
            k = int(self.sr * layer_size)
            tmpx = x.view(x.shape[0], x.shape[1], -1)
            topval = tmpx.topk(k, dim=2)[0][:, :, -1]
            topval = topval.expand(x.shape[2], x.shape[3], x.shape[0], x.shape[1]).permute(2, 3, 0, 1)
            comp = (x >= topval).to(x)
        return comp

class Triangle(nn.Module):
    r"""Applies the Sigmoid element-wise function:
    """
    def __init__(self,  power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input):
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power

    def extra_repr(self) -> str:
        return 'power=%s'%self.power


class Hardsigm(nn.Module):
    def __init__(self,  low: float = 0, high: float = 1):
        super(Hardsigm, self).__init__()
        self.low = low
        self.high = high

    def forward(self, x):
        return torch.clamp(x, self.low, self.high)

    def manual_grad(self, x):
        return (x >= self.low) & (x <= self.high)


class TanhSelf(nn.Module):
    def forward(self, x):
        return 0.5 + 0.5 * torch.tanh(x)

    def manual_grad(self, x):
        return (1 - torch.tanh(x)**2)*0.5


func_dict = {
    'sparse2d':Sparsify2D(),  #top-k value
    'sparse1d':Sparsify1D(),
    # 'abs':Sparsify2D_abs,  #top-k absolute value
    # 'invabs':Sparsify2D_invabs, #top-k minimal absolute value
    # 'vol':Sparsify2D_vol,  #cross channel top-k
    # 'brelu':breakReLU, #break relu
    # 'kact':Sparsify2D_kactive,
    'sigmoid':nn.Sigmoid(),
    'softmax':nn.Softmax(dim=0),
    'tanh': TanhSelf(),
    'relu':nn.ReLU(),
    'hardsigm': Hardsigm(),
    'x':nn.Identity(),
    'triangle':Triangle()
}