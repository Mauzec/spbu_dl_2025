import random
import torch
import numpy as np
from typing import Tuple, List
import time

from PIL import Image
import matplotlib.pyplot as plt

class BaseTransform:
    def __init__(self, p: float) -> None:
        assert 0.0 <= p <= 1.0, "p must be in [0, 1]"
        self.p = p
    def __call__(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError("BaseTransform - abstract")
    
class RandomCrop(BaseTransform):
    def __init__(self, p: float , crop_size: Tuple[int, int]) -> None:
        super().__init__(p)
        self.crop_size = crop_size
    def __call__(self, image: Image.Image) -> Image.Image:
        rnd_seed = time.time()
        random.seed(rnd_seed)
        if random.random() >= self.p:
            return image
        w, h = image.size
        cw, ch = self.crop_size
        if w < cw or h < ch:
            return image
        
        l = random.randint(0, w - cw)
        r = l + cw
        up = random.randint(0, h - ch)
        down = up + ch
        return image.crop((l, up, r, down))
    
class RandomRotate(BaseTransform):
    def __init__(self, p: float, degree: float) -> None:
        super().__init__(p)
        self.degree = degree
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        angle = random.uniform(-self.degree, self.degree)
        return image.rotate(angle)

class RandomZoom(BaseTransform):
    def __init__(self, p: float, scale: Tuple[float, float]) -> None:
        super().__init__(p)
        assert scale[1] >= scale[0] and scale[0] > 0.0, "Invalid scale range"
        self.scale = scale
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        w, h = image.size
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        
        nw, nh = int(w * scale_factor), int(h * scale_factor)
        image = image.resize((nw, nh), Image.Resampling.LANCZOS)
        
        l = (nw - w) // 2
        r = l + w
        u = (nh - h) // 2
        d = u + h
        
        return image.crop((l, u, r, d))
    
class Resize(BaseTransform):
    def __init__(self, size: Tuple[int, int]) -> None:
        super().__init__(p=1.0)
        self.size = size
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize(self.size, Image.Resampling.LANCZOS)
    
class ToTensor(BaseTransform):
    def __init__(self) -> None:
        super().__init__(p=1.0)
    def __call__(self, image: Image.Image) -> torch.Tensor:
        img = np.array(image).astype(np.float32) / 255.0
        if img.ndim == 2: # if grayscale => add channel
            img = img[:, :, None]
        return torch.from_numpy(
            img
        ).permute(2,0,1)
        
        
class Compose:
    def __init__(self, transforms: List[BaseTransform]) -> None:
        self.transforms = transforms
    def __call__(self, image: Image.Image) -> Image.Image:
        for t in self.transforms:
            image = t(image)
        return image

        
def imshow(x: torch.Tensor, title: str = ""):
    xnp = x.permute(1, 2, 0).numpy()
    plt.imshow(xnp)
    plt.title(title)
    plt.axis('off')
    plt.show()
        
if __name__ == "__main__":
    img = Image.open("keanu_0.jpg").convert("RGB")
    
    transform = Compose([
        RandomCrop(p=.8, crop_size=(128,128)),
        RandomRotate(p=.7, degree=90),
        RandomZoom(p=.6, scale=(1.3, 2.0)),
        ToTensor(),
    ])
    
    tensor_img: torch.Tensor = transform(img) # type: ignore
    imshow(tensor_img, title='Transformed Image')
    
    