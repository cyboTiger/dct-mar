import numpy as np
import torch

import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import seaborn as sns

def visualize(latent1: torch.Tensor, latent2: torch.tensor, direction='dct', save_dir=None, img_path=None):
    if latent1.ndim == 4:
        latent1 = latent1[0][0]
    elif latent1.ndim == 3:
        latent1 = latent1[0]
    
    if latent2.ndim == 4:
        latent2 = latent2[0][0]
    elif latent2.ndim == 3:
        latent2 = latent2[0]

    assert latent1.ndim == 2 and latent2.ndim == 2
    latent1 = latent1.squeeze().cpu().numpy()
    latent2 = latent2.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(25, 12))
    ax0 = axes[0]
    ax1 = axes[1]
    title0 = 'before DCT' if direction=='dct' else 'before inverse DCT'
    title1 = 'after DCT' if direction=='dct' else 'after inverse DCT'

    cbar_label0 = 'intensity in VAE latent space' if direction=='dct' else 'intensity in frequency space'
    cbar_label1 = 'intensity in frequency space' if direction=='dct' else 'intensity in VAE latent space'


    sns.heatmap(latent1, 
                ax=ax0, 
                cmap='YlOrRd',
                cbar_kws={'label': cbar_label0},
                xticklabels=True,
                yticklabels=True)
    ax0.set_title(title0)

    sns.heatmap(latent2, 
                ax=ax1, 
                cmap='YlOrRd',
                cbar_kws={'label': cbar_label1},
                xticklabels=True,
                yticklabels=True)
    ax1.set_title(title1)

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, img_path))
        print(f'Heatmap saved to {os.path.join(save_dir, img_path)} !')