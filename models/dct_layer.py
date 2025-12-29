import numpy as np
import torch
import torch.nn as nn

from torch_dct import dct, idct

class LinearDCT(nn.Linear):
    """
    使用 nn.Linear 实现任何 DCT/IDCT 变换的固定层。
    权重矩阵在初始化时计算并冻结 (requires_grad=False)。
    
    :param in_features: 变换的维度 N
    :param type: 'dct' 或 'idct'
    :param norm: 规范化参数，None 或 'ortho'
    """
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        # 调用父类 nn.Linear 的构造函数，创建 self.weight 参数
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)
        
        # 调用重置参数方法，计算并设置 DCT 矩阵
        self.reset_parameters()

    def reset_parameters(self):
        I = torch.eye(self.N, device=self.weight.device, dtype=self.weight.dtype)
        
        if self.type == 'dct':
            W_matrix = dct(I, norm=self.norm)
        elif self.type == 'idct':
            W_matrix = idct(I, norm=self.norm)
        else:
            raise ValueError(f"Unknown DCT type: {self.type}. Must be 'dct' or 'idct'.")

        self.weight.data = W_matrix.data.t()
        self.weight.requires_grad = False 
        
def apply_linear_2d(x, linear_layer):
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

class DCT2DLayer(nn.Module):
    """
    2D 离散余弦变换模块 (修正版，解决了内存不连续问题)。
    输入形状假定为 (B, C, H, W)。
    """
    def __init__(self, size_h: int, size_w: int, direction: str = 'dct', norm: str = None):
        super().__init__()
        
        # 确保 LinearDCT 存在并已定义
        # from your_dct_file import LinearDCT 
        
        self.linear_h = LinearDCT(size_h, direction, norm=norm, bias=False)
        self.linear_w = LinearDCT(size_w, direction, norm=norm, bias=False)
        self.H = size_h
        self.W = size_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (B, C, H, W)
        B, C, H, W = x.shape
        
        # 1. 作用于 W 维度 (dim=-1)
        
        # 将张量重排为 (B * C * H, W)
        x_flat_w = x.reshape(-1, W) 
        
        # 矩阵乘法: (B*C*H, W) @ (W, W) -> (B*C*H, W)
        X1_flat_w = self.linear_w(x_flat_w) 
        
        # 恢复形状到 (B, C, H, W)
        X1 = X1_flat_w.reshape(B, C, H, W) 

        # 2. 作用于 H 维度 (需要先转置)
        
        # 转置后 X1_T 形状为 (B, C, W, H)
        X1_T = X1.transpose(-1, -2) # 仍然是不连续的
        
        # 将张量重排为 (B * C * W, H)
        # 注意：使用 reshape 解决不连续问题
        X1_T_flat_h = X1_T.reshape(-1, H) 
        
        # 矩阵乘法: (B*C*W, H) @ (H, H) -> (B*C*W, H)
        X2_T_flat_h = self.linear_h(X1_T_flat_h)
        
        # 恢复形状到 (B, C, W, H)
        X2_T = X2_T_flat_h.reshape(B, C, W, H)
        
        # 最终转置回来，得到 (B, C, H, W)
        final_out = X2_T.transpose(-1, -2)
        
        return final_out

class MyDCT2DLayer(nn.Module):
    def __init__(self, m, n, ortho='True'):
        super().__init__()
        mcos = torch.cos(torch.pi / m * (torch.arange(m)+0.5).unsqueeze(0) * torch.arange(m).unsqueeze(1))
        ncos = torch.cos(torch.pi / n * torch.arange(n).unsqueeze(0) * (torch.arange(n)+0.5).unsqueeze(1))
        self.m = m
        self.n = n        
        self.mdct = nn.Parameter(mcos)
        self.ndct = nn.Parameter(ncos)
        self.norm = ortho
        # self.norm = None
        # if ortho == 'True':
        #     vnorm = torch.sqrt(torch.cat(torch.tensor(1/n), torch.full(n-1, 2/n)))
        #     unorm = torch.sqrt(torch.cat(torch.tensor(1/m), torch.full(m-1, 2/m)))
        #     self.norm = nn.Parameter(unorm.unsqueeze(1), vnorm.unsqueeze(0))
        
        self.reset_parameters()

    def reset_parameters(self):
        self.mdct.requires_grad = False 
        self.ndct.requires_grad = False 

    def forward(self, x):
        # x shape: [B, C, H, W]
        y = self.mdct @ x
        z = y @ self.ndct
        if self.norm is not None:
            z[..., 0, 0] /= np.sqrt(self.m * self.n)
            z[..., 1:, 0] /= np.sqrt(self.m * self.n / 2)
            z[..., 0, 1:] /= np.sqrt(self.m * self.n / 2)
            z[..., 1:, 1:] /= np.sqrt(self.m * self.n / 4)
        
        return z

class MyInverseDCT2DLayer(nn.Module):
    def __init__(self, m, n, ortho='True'):
        super().__init__()
        ucos = torch.cos(torch.pi / m * (torch.arange(m)+0.5).unsqueeze(1) * torch.arange(m).unsqueeze(0))
        vcos = torch.cos(torch.pi / n * torch.arange(n).unsqueeze(1) * (torch.arange(n)+0.5).unsqueeze(0))
        if ortho is not None:
            ucos[:, 0] /= np.sqrt(m)
            ucos[:, 1:] /= np.sqrt(m / 2)
            vcos[0, :] /= np.sqrt(n)
            vcos[1:, :] /= np.sqrt(n / 2)
        self.m = m
        self.n = n        
        self.mdct = nn.Parameter(ucos)
        self.ndct = nn.Parameter(vcos)
        self.norm = ortho
        # self.norm = None
        # if ortho == 'True':
        #     vnorm = torch.sqrt(torch.cat(torch.tensor(1/n), torch.full(n-1, 2/n)))
        #     unorm = torch.sqrt(torch.cat(torch.tensor(1/m), torch.full(m-1, 2/m)))
        #     self.norm = nn.Parameter(unorm.unsqueeze(1), vnorm.unsqueeze(0))
        
        self.reset_parameters()

    def reset_parameters(self):
        self.mdct.requires_grad = False 
        self.ndct.requires_grad = False 

    def forward(self, x):
        # x shape: [B, C, H, W]
        y = self.mdct @ x
        z = y @ self.ndct
        return z
        # if self.norm is not None:
        #     z[..., 0, 0] /= torch.sqrt(self.m * self.n)
        #     z[..., 1:, 0] /= torch.sqrt(self.m * self.n / 2)
        #     z[..., 0, 1:] /= torch.sqrt(self.m * self.n / 2)
        #     z[..., 1:, 1:] /= torch.sqrt(self.m * self.n / 4)