import torch
import torch.nn as nn
import torch_dct as dct
import numpy as np
from models.dct_layer import DCT2DLayer, MyDCT2DLayer, MyInverseDCT2DLayer

from PIL import Image

class DCTtransform(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.dct_layer = DCT2DLayer(size_h=h, size_w=w, direction='dct', norm='ortho')
        self.idct_layer = DCT2DLayer(size_h=h, size_w=w, direction='idct', norm='ortho')
    
    def forward(self, x):
        res = []
        x = self.dct_layer(x)
        res.append(x)
        x = self.idct_layer(x)
        res.append(x)

        return res

class MyDCTtransform(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.dct_layer = MyDCT2DLayer(h, w)
        self.idct_layer = MyInverseDCT2DLayer(h, w)
    
    def forward(self, x):
        res = []
        x = self.dct_layer(x)
        res.append(x)
        x = self.idct_layer(x)
        res.append(x)

        return res

    
def main():
    # x = torch.randn((1, 3, 256, 256))
    img_path = '/home/ruihan/data/imagenet/train/n01443537/n01443537_50.JPEG'
    x = Image.open(img_path)
    x = np.array(x, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x)
    c, h, w = x.shape
    
    res1 = []
    x = dct.dct_2d(x)
    res1.append(x)
    x = dct.idct_2d(x)
    res1.append(x)

    dctnet = DCTtransform(h, w)
    res2 = dctnet(x.unsqueeze(0))

    mydctnet = MyDCTtransform(h, w)
    res3 = mydctnet(x)
    
    eps = 1e-5

    print(((res1[1]-x) / (x + eps)).abs().max().item())
    print(((res1[1]-x) / (x)).abs().max().item())

if __name__ == '__main__':
    main()