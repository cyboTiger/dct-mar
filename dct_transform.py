import torch
# import torch_dct as dct
import numpy as np
from torchvision import transforms
from models.dct_layer import DCT2DLayer
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import seaborn as sns

def generate_square_sequence(max_val, num):
    max_val /= 100
    sequence = []
    i = 1
    sqrt_max_val = int(max_val**0.5)+1
    interval = sqrt_max_val / num
    
    for j in range(num-1):
        sequence.append(i * i)
        i = int(i+interval)
    
    sequence.append(int(max_val))
    
    return sequence

def mask_generator(img, ratio=None, num_pixel=None):
    if isinstance(img, torch.Tensor):
        h, w = img.shape[-2], img.shape[-1]
        use_tensor = True
    else:
        h, w = img.shape[:2]
        use_tensor = False
    
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    distances = np.sqrt(y_indices**2 + x_indices**2)
    
    flat_distances = distances.flatten()
    sorted_distances = np.sort(flat_distances)
    if ratio is not None:
        threshold_idx = int(len(sorted_distances) * ratio)
    
    if ratio is not None:
        threshold = sorted_distances[threshold_idx] if ratio < 1.0 else np.inf
        mask = (distances < threshold).astype(np.float32)
    elif num_pixel is not None:
        threshold = sorted_distances[num_pixel] if num_pixel < h*w else np.inf
        mask = (distances < threshold).astype(np.float32)
    else:
        mask = np.zeros((h, w), dtype=np.float32)
    
    if use_tensor:
        mask = torch.from_numpy(mask)
    
    return mask.bool()


def load_and_preprocess_image(image_path):
    """
    load jpg and convert to torch tensors
    """

    img = Image.open(image_path).convert('L')  # gray image
    img = Image.open(image_path)  # rgb image
    print(f"Original image size: {img.size}")
    
    # convert to numpy
    img_array = np.array(img, dtype=np.float32)
    
    # convert to torch tensor
    img_tensor = torch.from_numpy(img_array)
    img_tensor = img_tensor.permute(2, 0, 1) # (C, W, H)
    
    return img_tensor

def apply_dct_and_visualize(image_path, block_size=8):
    """
    apply discrete cosine transformation
    """
    # load and process image
    img_tensor = load_and_preprocess_image(image_path)
    H, W = img_tensor.shape[-2:]
    
    # resize
    max_size = 512
    if H > max_size or W > max_size:
        resize = transforms.Resize(max_size)
        img_tensor = resize(img_tensor.unsqueeze(0)).squeeze(0)
        print(img_tensor.shape)
        H, W = img_tensor.shape[-2:]
        print(f"Resized image size: {H}x{W}")
    
    # apply 2d dct
    dctnet = DCT2DLayer(H, W)
    idctnet = DCT2DLayer(H, W, 'idct')
    dct_result = dctnet(img_tensor.unsqueeze(0)).squeeze()
    print(f"DCT Coeff range: [{dct_result.min():.2f}, {dct_result.max():.2f}]")
    
    row, col = 3, 5

    # mask sequence
    masked_seqs = [mask_generator(dct_result, num_pixel=num_p) for num_p in generate_square_sequence(H*W, row*col)]
    masked_seqs = [torch.where(mask, dct_result, 0) for mask in masked_seqs]
    
    recon_imgs = [idctnet(masked_img.unsqueeze(0)).squeeze() for masked_img in masked_seqs]

    fig, axes = plt.subplots(row, col, figsize=(30, 12))
    
    # Original image
    for i in range(row):
        for j in range(col):
            axes[i, j].imshow(np.transpose(recon_imgs[i*col+j].clamp(min=0, max=255).int(), (1, 2, 0)))
            axes[i, j].set_title(f'Unmasked ratio {generate_square_sequence(H*W, row*col)[i*col+j]}/{(H*W)}={generate_square_sequence(H*W, row*col)[i*col+j]/(H*W):.4f}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('./dct.png')
    plt.show()

if __name__ == "__main__":
    dataset_path = "/home/ruihan/data/imagenet/train/"

    class_path = random.sample(os.listdir(dataset_path), 1)[0]
    class_path = os.path.join(dataset_path, class_path)

    image_path = random.sample(os.listdir(class_path), 1)[0]
    image_path = os.path.join(class_path, image_path)

    print("Image path:", image_path)
    
    apply_dct_and_visualize(image_path)