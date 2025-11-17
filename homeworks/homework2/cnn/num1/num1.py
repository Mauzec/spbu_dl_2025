import torch
import math

class CustomBatchNorm:
    """
    Custom Batch Normalization Layer
    """
    def __init__(self, 
                 num_features, 
                 eps=1e-4, 
                 momentum=.1, 
                 affine: bool = True, 
                 track_run_stats: bool = True,
    ) -> None:
        if not isinstance(num_features, int) or num_features <= 0:
            raise ValueError("num_features should be a positive int")
        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_run_stats = bool(track_run_stats)
        
        if self.affine:
            self.gamma = torch.ones(num_features, requires_grad=True)
            self.beta = torch.zeros(num_features, requires_grad=True)
        else:
            self.gamma, self.beta = None, None
        
        if self.track_run_stats:
            self.run_mean = torch.zeros(num_features)
            self.run_var = torch.ones(num_features)
        else:
            self.run_mean, self.run_var = None, None
        
        self.training = True
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input should be a torch.Tensor")
        if x.dim() < 2 or x.size(1) != self.num_features:
            raise ValueError(f"Input should have shape (N, {self.num_features})")
    
        bshape = [1]*x.dim()
        bshape[1] = self.num_features
        reduce_dims=[0]+list(range(2,x.dim()))
        
        if self.training:
            mean = x.mean(dim=reduce_dims)
            var = x.var(dim=reduce_dims, unbiased=False)
            if self.track_run_stats:
                self.run_mean = self.run_mean.to(device=x.device,dtype=x.dtype) # type: ignore
                self.run_var = self.run_var.to(device=x.device,dtype=x.dtype) # type: ignore
                self.run_mean.mul_(1-self.momentum).add_(self.momentum * mean)
                self.run_var.mul_(1-self.momentum).add_(self.momentum * var)
        else:
            if not self.track_run_stats:
                raise ValueError("track_run_stats is False, cannot use running stats in eval mode")
            mean = self.run_mean.to(device=x.device,dtype=x.dtype) # type: ignore
            var = self.run_var.to(device=x.device,dtype=x.dtype) # type: ignore
            
        y = (x-mean.view(bshape)) / torch.sqrt(var.view(bshape) + self.eps)
        if self.affine:
            gamma = self.gamma.to(device=x.device,dtype=x.dtype).view(bshape) # type: ignore
            beta = self.beta.to(device=x.device,dtype=x.dtype).view(bshape) # type: ignore
            y = y * gamma + beta
        return y
    
class CustomLinear:
    def __init__(self, 
                 in_features: int,
                 out_features: int, 
                 bias:bool = True
    ) -> None:
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError("in_features should be a positive int")
        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError("out_features should be a positive int")
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bool(bias)
        
        bound = 1.0 / math.sqrt(in_features)
        self.weight = torch.empty(out_features, in_features, requires_grad=True)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        if self.bias_flag:
            self.bias = torch.empty(out_features, requires_grad=True)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input should be a torch.Tensor")
        if x.dim() < 2 or x.size(1) != self.in_features:
            raise ValueError(f"Input should have shape (N, {self.in_features})")
        
        weight = self.weight.to(device=x.device,dtype=x.dtype) # type: ignore
        y = x.matmul(weight.t())
        
        if self.bias_flag:
            bias = self.bias.to(device=x.device,dtype=x.dtype) # type: ignore
            y = y + bias
        return y
    
class CustomDropout:
    def __init__(self,
                 p: float = .5
    ) -> None:
        if not (.0 <= p < 1.0):
            raise ValueError("p must be in [0,1)")
        self.p = float(p)
        self.training = True
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input should be a torch.Tensor")
        if not x.is_floating_point():
            raise ValueError("Input should be a floating point tensor")
        
        if not self.training or self.p == .0:
            return x
        
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) < keep_prob).to(dtype=x.dtype)
        return x * mask / keep_prob
            
class CustomReLU:
    """
    max(0, x)
    """
    
    def __init__(self, inplace: bool = False) -> None:
        self.inplace = bool(inplace)
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input should be a torch.Tensor")
        
        return x.clamp_(min=0) if self.inplace else x.clamp(min=0)

class CustomSigmoid:
    """
    1 / (1 + exp(-x))
    """
    def __init__(self) -> None:
        pass
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input should be a torch.Tensor")
        if not x.is_floating_point():
            raise ValueError("Input should be a floating point tensor")
        
        return 1.0 / (1.0 + torch.exp(-x))

class CustomSoftmax:
    def __init__(self, dim: int = 1) -> None:
        self.dim = int(dim)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input should be a torch.Tensor")
        if not x.is_floating_point():
            raise ValueError("Input should be a floating point tensor")
        
        dim = self.dim if self.dim >=0 else x.dim() + self.dim
        if dim < 0 or dim >= x.dim():
            raise ValueError(f"dim out of range")
        
        x_max = x.amax(dim=dim, keepdim=True)
        x_stable = x - x_max
        exps = torch.exp(x_stable)
        denom = exps.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        return exps / denom
    