import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Sequence, Tuple, Union
import time
import unittest

class Element:
    def __init__(self,data,_children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op
        
    def __repr__(self):
        return f"Element(data={self.data}, grad={self.grad})"
    
    def __add__(self,other):
        x = Element(self.data+other.data, (self,other), '+')
        def bw():
            self.grad += x.grad
            other.grad += x.grad
        x._backward = bw
        return x
    
    def __mul__(self,other):
        x = Element(self.data*other.data, (self,other), '*')
        def bw():
            self.grad += other.data * x.grad
            other.grad += self.data * x.grad
        x._backward = bw
        return x
    
    def relu(self):
        x = Element(max(0, self.data), (self,), 'relu')
        def bw():
            self.grad += (x.data>0) * x.grad
        x._backward = bw
        return x
    
    def backward(self):
        seen = set()
        topo: list[Element] = []
        def build(e: Element):
            if e in seen:
                return
            
            seen.add(e)
            for ch in e._prev:
                build(ch)
            topo.append(e)
            
        build(self)
        self.grad = 1.0
        for e in reversed(topo):
            e._backward()
            
