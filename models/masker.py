import torch
import numpy as np

class GlobalFrequencyMasker:
    def __init__(self, grid_size=16):
        """
        grid_size * patch_size = img_size
        """
        self.grid_size = grid_size
        self.seq_len = grid_size * grid_size
        # Zig-zag order starting from top-left
        self.zigzag_order = self._generate_frequency_order()

    def _generate_frequency_order(self):
        """sorted based on euclidean distance"""
        order = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt(i**2 + j**2) 
                order.append((i * self.grid_size + j, dist))
        
        order.sort(key=lambda x: x[1])
        return np.array([x[0] for x in order])

    def sample_orders(self, bsz):
        order = torch.from_numpy(self.zigzag_order).long().cuda()
        return order.unsqueeze(0).repeat(bsz, 1)

    def random_masking(self, x, orders):
        bsz, seq_len, _ = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0] 
        num_keep = int(np.ceil(seq_len * (1 - mask_rate)))
        
        mask = torch.ones(bsz, seq_len, device=x.device)
        # 前 num_keep 个是最重要的低频 Token，不遮掩 (mask=0)
        # 后面的高频 Token 被遮掩 (mask=1)
        for i in range(bsz):
            mask[i, orders[i, :num_keep]] = 0
            
        return mask