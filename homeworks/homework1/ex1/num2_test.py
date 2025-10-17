import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Sequence, Tuple, Union
import time
import unittest
from num2 import Element


class TestElement(unittest.TestCase):
    def test_add_mul_relu(self):
        a,b,c = Element(2.0), Element(3.0), Element(-1.0)
        d = a*b+c
        r = d.relu()
        
        r.backward()
        
        self.assertEqual(r.data, 5.0)
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)
        self.assertEqual(c.grad, 1.0)
        
    
    def test_add(self):
        a,b = Element(2.0), Element(3.0)
        c = a + b
        
        c.backward()
        
        self.assertEqual(c.data, 5.0)
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)
        
    def test_mul(self):
        a,b = Element(2.0), Element(3.0)
        c = a * b
        
        c.backward()
        
        self.assertEqual(c.data, 6.0)
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)
        
        def test_relu(self):
            a = Element(-2.0)
            b = a.relu()
            
            b.backward()
            
            self.assertEqual(b.data, 0.0)
            self.assertEqual(a.grad, 0.0)