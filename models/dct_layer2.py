import numpy as np
import torch
import torch.nn as nn

from torch_dct import dct, idct

class LinearDCT(nn.Module):
    def __init__(self, in_features, type, norm=None):
        super(LinearDCT, self).__init__()
        self.type = type
        self.N = in_features
        self.norm = norm
        # 直接注册为 Parameter，不使用 nn.Linear
        self.weight = nn.Parameter(torch.Tensor(in_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        I = torch.eye(self.N)
        if self.type == 'dct':
            # 注意：这里直接获取变换矩阵
            W_matrix = dct(I, norm=self.norm)
        elif self.type == 'idct':
            W_matrix = idct(I, norm=self.norm)
        
        # 核心：W_matrix 的每一行是基向量。
        # 对于矩阵乘法 x @ W.T，x 的每一行会与 W 的每一行（基向量）做内积
        self.weight.data = W_matrix.data
        self.weight.requires_grad = False 

    def forward(self, x):
        # 明确使用 x 乘以 权重矩阵的转置
        return F.linear(x, self.weight)

class DCT2DLayer(nn.Module):
    def __init__(self, size_h: int, size_w: int, direction: str = 'dct', norm: str = 'ortho'):
        super().__init__()
        # 强制建议使用 norm='ortho'，否则 DC 量级会随尺寸变化
        self.norm = norm
        
        # 获取基础变换矩阵
        I_h = torch.eye(size_h)
        I_w = torch.eye(size_w)
        
        if direction == 'dct':
            self.register_buffer('mat_h', dct(I_h, norm=norm))
            self.register_buffer('mat_w', dct(I_w, norm=norm))
        else:
            self.register_buffer('mat_h', idct(I_h, norm=norm))
            self.register_buffer('mat_w', idct(I_w, norm=norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # mat_h: (H, H) -> 负责变换高度维度
        # mat_w: (W, W) -> 负责变换宽度维度
        
        # 语义：输出的 (i,j) 位置是 输入矩阵 与 两个方向基向量 的乘积
        # 'h' 和 'w' 是空间维度，'i' 和 'j' 是频率维度
        return torch.einsum('ih, bchw, jw -> bcij', self.mat_h, x, self.mat_w)