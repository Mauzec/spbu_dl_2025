import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union, Optional, Any, Dict
import unittest
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

import numpy as np
import torch

from torch import sigmoid

TensorLike = Union[torch.Tensor, Sequence[float]]


def float_tensor(x) -> torch.Tensor:
    """Returns detached float32 tensor"""
    if isinstance(x, torch.Tensor):
        return x.clone().detach().float()

    return torch.tensor(x, dtype=torch.float32)


def nll_loss(
    preds: torch.Tensor, lbls: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    probs = torch.clamp(preds, eps, 1.0 - eps)
    return -(lbls * torch.log(probs) + (1.0 - lbls) * torch.log(1.0 - probs)).mean()


"""------- Optimizers -------"""

@dataclass
class Opt(ABC):
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def step(self, params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor]) -> None:
        raise NotImplementedError



@dataclass
class MomentumOpt(Opt):
    lr: float = 1e-2
    momentum: float = 0.9

    def __post_init__(self) -> None:
        self.v: list[torch.Tensor] = []

    def step(
        self,
        params: Sequence[torch.Tensor],
        grads: Sequence[torch.Tensor],
    ) -> None:
        if len(self.v) == 0:
            self.v = [torch.zeros_like(p) for p in params]

        for idx, (p, g) in enumerate(zip(params, grads)):
            self.v[idx] = self.momentum * self.v[idx] - self.lr * g
            p += self.v[idx]

@dataclass
class NAGOpt(Opt):
    lr: float = 1e-2
    momentum: float = 0.9

    def __post_init__(self) -> None:
        self.v: list[torch.Tensor] = []

    def step(
        self,
        params: Sequence[torch.Tensor],
        grads: Sequence[torch.Tensor],
    ) -> None:
        if len(self.v) == 0:
            self.v = [torch.zeros_like(p) for p in params]

        for idx, (p, g) in enumerate(zip(params, grads)):
            v_prev = self.v[idx].clone()
            self.v[idx] = self.momentum * self.v[idx] - self.lr * g
            
            '''
            хитрость:
            добавляем текущую сокрость v
            затем self.momentum * self.v[idx] - self.momentum * v_prev
            доказано, что это приблизительно соотвествует дефолтному NAG
            '''
            p += -self.momentum * v_prev + (1 + self.momentum) * self.v[idx]
            

@dataclass
class AdagradOpt(Opt):
    lr: float = 1e-2
    eps: float = 1e-10

    def __post_init__(self) -> None:
        self.sq_grad_accum: list[torch.Tensor] = []

    def step(
        self,
        params: Sequence[torch.Tensor],
        grads: Sequence[torch.Tensor],
    ) -> None:
        if len(self.sq_grad_accum) == 0:
            self.sq_grad_accum = [torch.zeros_like(p) for p in params]

        for idx, (p, g) in enumerate(zip(params, grads)):
            self.sq_grad_accum[idx] += g * g
            p -= self.lr * g / (torch.sqrt(self.sq_grad_accum[idx]) + self.eps)
            
@dataclass
class RMSpropOpt(Opt):
    lr: float = 1e-2
    gamma: float = 0.9
    eps: float = 1e-10

    def __post_init__(self) -> None:
        self.sq_grad_avg: list[torch.Tensor] = []

    def step(
        self,
        params: Sequence[torch.Tensor],
        grads: Sequence[torch.Tensor],
    ) -> None:
        if len(self.sq_grad_avg) == 0:
            self.sq_grad_avg = [torch.zeros_like(p) for p in params]

        for idx, (p, g) in enumerate(zip(params, grads)):
            self.sq_grad_avg[idx] = self.gamma * self.sq_grad_avg[idx] + (1 - self.gamma) * (g * g)
            p -= self.lr * g / (torch.sqrt(self.sq_grad_avg[idx]) + self.eps)

@dataclass
class AdamOpt(Opt):
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self._m: list[torch.Tensor] = []
        self._v: list[torch.Tensor] = []

    def step(
        self,
        params: Sequence[torch.Tensor],
        grads: Sequence[torch.Tensor],
    ) -> None:

        if len(self._m) == 0:
            self._m = [torch.zeros_like(p) for p in params]
            self._v = [torch.zeros_like(p) for p in params]

        for idx, (p, g) in enumerate(zip(params, grads)):
            self._m[idx] = self.beta1 * self._m[idx] + (1.0 - self.beta1) * g
            self._v[idx] = self.beta2 * self._v[idx] + (1.0 - self.beta2) * (g * g)
            p -= self.lr * (self._m[idx] / (torch.sqrt(self._v[idx]) + self.eps))

@dataclass
class AdamWOpt(AdamOpt):
    weight_decay: float = 1e-2

    def step(
        self,
        params: Sequence[torch.Tensor],
        grads: Sequence[torch.Tensor],
    ) -> None:

        if len(self._m) == 0:
            self._m = [torch.zeros_like(p) for p in params]
            self._v = [torch.zeros_like(p) for p in params]

        for idx, (p, g) in enumerate(zip(params, grads)):
            self._m[idx] = self.beta1 * self._m[idx] + (1.0 - self.beta1) * g
            self._v[idx] = self.beta2 * self._v[idx] + (1.0 - self.beta2) * (g * g)
            update = self._m[idx] / (torch.sqrt(self._v[idx]) + self.eps)
            update = update + self.weight_decay * p
            p -= self.lr * update


@dataclass
class NadamOpt(AdamOpt):
    def __post_init__(self) -> None:
        super().__post_init__()

    def step(
        self,
        params: Sequence[torch.Tensor],
        grads: Sequence[torch.Tensor],
    ) -> None:

        if len(self._m) == 0:
            self._m = [torch.zeros_like(p) for p in params]
            self._v = [torch.zeros_like(p) for p in params]

        for idx, (p, g) in enumerate(zip(params, grads)):
            self._m[idx] = self.beta1 * self._m[idx] + (1.0 - self.beta1) * g
            self._v[idx] = self.beta2 * self._v[idx] + (1.0 - self.beta2) * (g * g)
            lookahead = self.beta1 * self._m[idx] + (1.0 - self.beta1) * g
            p -= self.lr * (lookahead / (torch.sqrt(self._v[idx]) + self.eps))


@dataclass
class AdadeltaOpt(Opt):
    rho: float = 0.9
    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.accum_grad: list[torch.Tensor] = []
        self.accum_update: list[torch.Tensor] = []

    def step(
        self,
        params: Sequence[torch.Tensor],
        grads: Sequence[torch.Tensor],
    ) -> None:
        if len(self.accum_grad) == 0:
            self.accum_grad = [torch.zeros_like(p) for p in params]
            self.accum_update = [torch.zeros_like(p) for p in params]

        for idx, (p, g) in enumerate(zip(params, grads)):
            self.accum_grad[idx] = (
                self.rho * self.accum_grad[idx] + (1.0 - self.rho) * (g * g)
            )
            rms_grad = torch.sqrt(self.accum_grad[idx] + self.eps)
            rms_update = torch.sqrt(self.accum_update[idx] + self.eps)
            delta = -(rms_update / rms_grad) * g
            p += delta
            self.accum_update[idx] = (
                self.rho * self.accum_update[idx] + (1.0 - self.rho) * (delta * delta)
            )
            


"""------- END -------"""


def train_neuron(
    optimizer: Opt,
    feats: TensorLike,
    lbls: TensorLike,
    init_w: TensorLike,
    lr: float,
    init_bias: Union[float, int],
    epochs: int,
    eps: float = 1e-7,
) -> Tuple[Sequence[float], float, Sequence[float]]:
    """Trains neuron using given optimizer"""

    w = float_tensor(init_w).view(-1)
    bias = torch.tensor(float(init_bias), dtype=torch.float32)
    feats = float_tensor(feats)
    lbls = float_tensor(lbls).view(-1)

    if (
        epochs <= 0
        or lr <= 0
        or w.shape[0] != feats.shape[1]
        or feats.shape[0] != lbls.shape[0]
    ):
        raise ValueError("invalid params")

    if feats.shape[0] == 0:
        raise ValueError("empty feats")

    n_smpls = feats.shape[0]
    nllhist: list[float] = []

    for _ in range(epochs):
        logits = feats @ w + bias

        preds = torch.sigmoid(logits)

        loss = nll_loss(preds, lbls, eps=eps)
        nllhist.append(round(float(loss), 5))

        err = preds - lbls

        grad_w = feats.t() @ err / n_smpls
        grad_b = err.mean()

        params = [w, bias]
        grads = [grad_w, grad_b]
        optimizer.step(params, grads)
        w, bias = params  # updated in-place

        # w -= lr*grad_w
        # bias -= lr*grad_b

    return w.tolist(), float(bias.item()), nllhist


def calc_neuron(
    feats: torch.Tensor,
    w: Union[torch.Tensor, Sequence[float]],
    lbls: torch.Tensor,
    bias: Union[torch.Tensor, float],
    eps: float = 1e-7,
) -> Tuple[float, float]:
    """Returns nll and accuracy"""

    logits = feats @ w + bias
    probs = torch.clamp(sigmoid(logits), eps, 1.0 - eps)
    preds = (probs >= 0.5).float()
    loss = nll_loss(probs, lbls, eps=eps)
    acc = (preds == lbls).float()

    return nll_loss(probs, lbls, eps=eps).item(), (preds == lbls).float().mean().item()


def load_subset(
    path: Path,
    neg_year: int,
    pos_year: int,
    class_samples: Union[None, int] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"where is {path}?")

    ds = np.loadtxt(path, delimiter=",", dtype=np.float32)

    years, feats = ds[:, 0], ds[:, 1:]
    neg_idxs = np.where(years < neg_year)[0]
    pos_idxs = np.where(years > pos_year)[0]

    rng = np.random.default_rng(seed)

    neg_smpl = (
        rng.choice(
            neg_idxs,
            size=min(int(class_samples), neg_idxs.size),
            replace=False,
        )
        if class_samples is not None
        else neg_idxs
    )
    pos_smpl = (
        rng.choice(
            pos_idxs,
            size=min(int(class_samples), pos_idxs.size),
            replace=False,
        )
        if class_samples is not None
        else pos_idxs
    )
    sel_idxs = np.concatenate([neg_smpl, pos_smpl])
    rng.shuffle(sel_idxs)
    return torch.from_numpy(feats[sel_idxs]), torch.from_numpy(
        (years[sel_idxs] == pos_year).astype(np.float32)
    )


def std_feats(
    feats: torch.Tensor,
    eps: float = 1e-7,
    mean: Union[None, torch.Tensor] = None,
    std: Union[None, torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mean is None:
        mean = feats.mean(dim=0)
    if std is None:
        std = feats.std(dim=0)
    adj = torch.where(std < eps, torch.ones_like(std), std)
    return (feats - mean) / adj, mean, adj


def _prepare_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base = Path.cwd()
    train_path = base / "YearPredictionMSD_train.txt"
    test_path = base / "YearPredictionMSD_test.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"not found files: {train_path}, {test_path}")
    neg, pos = 1999, 2000
    train_feats, train_lbls = load_subset(
        train_path,
        neg,
        pos,
        class_samples=5000,
        seed=int(time.time()),
    )
    test_feats, test_lbls = load_subset(
        test_path, neg, pos, class_samples=2000, seed=int(time.time())
    )
    train_feats, mean, std = std_feats(train_feats)
    test_feats, _, _ = std_feats(test_feats, mean=mean, std=std)
    return train_feats, train_lbls, test_feats, test_lbls


def run_with(opt: Opt, draw: bool = True) -> Dict[str, Any]:
    """Runs training and testing with given optimizer."""
    train_feats, train_lbls, test_feats, test_lbls = _prepare_data()

    init_w, init_b = torch.zeros(train_feats.shape[1]), 0.0
    w, b, h = train_neuron(
        opt,
        train_feats,
        train_lbls,
        init_w,
        lr=0.01,
        init_bias=init_b,
        epochs=40,
    )

    w_tensor = torch.tensor(w, dtype=torch.float32)
    train_loss, train_acc = calc_neuron(train_feats, w_tensor, train_lbls, bias=init_b)
    test_loss, test_acc = calc_neuron(test_feats, w_tensor, test_lbls, bias=init_b)

    print(f"[{opt.name()}] [TRAIN] nll: {train_loss:.5f}, acc: {train_acc:.5f}")
    print(f"[{opt.name()}] [ TEST] nll: {test_loss:.5f}, acc: {test_acc:.5f}")

    res = {
        "train_nll": train_loss,
        "train_acc": train_acc,
        "test_nll": test_loss,
        "test_acc": test_acc,
        "weights": w,
        "bias": init_b,
        "nll_history": h,
    }

    if not draw:
        return res

    plt.figure(figsize=(6, 4))
    plt.title(f"nll history ({opt.name()})")
    plt.xlabel("epoch")
    plt.ylabel("nll")
    plt.grid()
    plt.plot(h, label="nll")
    plt.legend()
    plt.show()

    return res


if __name__ == "__main__":
    run_with(AdamOpt(), draw=True)
