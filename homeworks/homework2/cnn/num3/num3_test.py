import random
import torch
import numpy as np
from typing import Tuple, List
import time

from PIL import Image
import matplotlib.pyplot as plt
import unittest

from num3 import *


class TestTransforms(unittest.TestCase):
    def test_random_crop(self):
        img = Image.new('RGB', (100, 100), color = 'red')
        transform = RandomCrop(p=1.0, crop_size=(50, 50))
        cropped_img = transform(img)
        self.assertEqual(cropped_img.size, (50, 50))
    
    def test_random_rotate(self):
        img = Image.new('RGB', (100, 100), color = 'red')
        transform = RandomRotate(p=1.0, degree=45)
        rotated_img = transform(img)
        self.assertEqual(rotated_img.size, (100, 100))
    
    def test_random_zoom(self):
        img = Image.new('RGB', (100, 100), color = 'red')
        transform = RandomZoom(p=1.0, scale=(0.5, 1.5))
        zoomed_img = transform(img)
        self.assertTrue(50 <= zoomed_img.size[0] <= 150)
        self.assertTrue(50 <= zoomed_img.size[1] <= 150)
        
    def test_compose(self):
        img = Image.new('RGB', (100, 100), color = 'red')
        transform = Compose([
            RandomCrop(p=1.0, crop_size=(80, 80)),
            RandomRotate(p=1.0, degree=30),
            RandomZoom(p=1.0, scale=(0.8, 1.2))
        ])
        transformed_img = transform(img)
        self.assertTrue(64 <= transformed_img.size[0] <= 96)
        self.assertTrue(64 <= transformed_img.size[1] <= 96)
    