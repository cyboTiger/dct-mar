import numpy as np
from PIL import Image

def rgb_to_ycbcr(image_path):
    img = Image.open(image_path).convert('RGB')
    rgb = np.array(img).astype(np.float32)
    
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    y  = 16  + 0.257 * r + 0.504 * g + 0.098 * b
    cb = 128 - 0.148 * r - 0.291 * g + 0.439 * b
    cr = 128 + 0.439 * r - 0.368 * g - 0.071 * b

    ycbcr = np.stack([y, cb, cr], axis=2)
    ycbcr = np.clip(ycbcr, 0, 255).astype(np.uint8)
    
    return ycbcr

# 使用示例
result = rgb_to_ycbcr('/home/ruihan/data/imagenet/train/n07753592/n07753592_17122.JPEG')
Image.fromarray(result).save('output_ycbcr.jpg')