import torch
import torch.nn as nn
import torch_dct as dct
from models.dct_layer import DCT2DLayer

class DCTtransform(nn.Module):
    def __init__(self, img_size=256):
        super().__init__()
        self.dct_layer = DCT2DLayer(size_h=img_size, size_w=img_size, direction='dct', norm='ortho')
        self.idct_layer = DCT2DLayer(size_h=img_size, size_w=img_size, direction='idct', norm='ortho')
    
    def forward(self, x):
        res = []
        x = self.dct_layer(x)
        res.append(x)
        x = self.idct_layer(x)
        res.append(x)

        return res
    
def main():
    x = torch.randn((1, 3, 256, 256))
    dctnet = DCTtransform()
    res1 = dctnet(x)

    res2 = []
    x = dct.dct_2d(x)
    res2.append(x)
    x = dct.idct_2d(x)
    res2.append(x)

    print((res1[0]-res2[0]).abs().max().item())
    print((res1[1]-res2[1]).abs().max().item())

if __name__ == '__main__':
    main()