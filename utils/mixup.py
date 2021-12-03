import numpy as np
import torch

def image_mixup(images, labels, alpha):
    lam = np.random.beta(alpha, alpha)  # 获取随机数
    index = torch.randperm(images.shape[0]).cuda()
    mixed_image = lam * images + (1 - lam) * images[index, :]
    mixed_label = lam * labels + (1 - lam) * labels[index, :]
    return mixed_image, mixed_label
