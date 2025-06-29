

import torch as tc
from components import ResViTSeg 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import BraTSData
from pytorch2tikz import Architecture
import numpy as np
import random

DEVICE = tc.device("cuda" if tc.cuda.is_available() else "cpu")
MODEL_PATH = "C:\\Users\\shelly\\Desktop\\Unet\\Unet\\StateDict\\dicfinal_ma_res_vit.pth"
BATCH_SIZE = 8
val_dataset   = BraTSData('Unet\\Dataset\\val_images.npy', 'Unet\\Dataset\\val_masks.npy')
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=True)
# 加载模型
model = ResViTSeg(n=1).to(DEVICE)
model.load_state_dict(tc.load(MODEL_PATH, map_location=DEVICE))
model.eval()
indices = random.sample(range(len(val_dataset)), 5)

# 准备画布
fig, axes = plt.subplots(5, 3, figsize=(12, 10))
fig.suptitle("Comparison: Image / Ground Truth / Prediction", fontsize=16)

for i, idx in enumerate(indices):
    image, mask = val_dataset[idx]
    image = image.unsqueeze(0).to(DEVICE)  # 加 batch 维度 (1, 1, H, W)

    # 模型预测
    with tc.no_grad():
        pred = model(image)
        pred_mask = (tc.sigmoid(pred) > 0.5).float().cpu().squeeze().numpy()

    # 还原图像和 mask
    image_np = image.cpu().squeeze().numpy()
    mask_np = mask.squeeze().numpy()

    # 显示图像、真实 mask 和预测 mask
    axes[i, 0].imshow(image_np, cmap='gray')
    axes[i, 0].set_title("Image")
    axes[i, 1].imshow(mask_np, cmap='gray')
    axes[i, 1].set_title("Ground Truth")
    axes[i, 2].imshow(pred_mask, cmap='gray')
    axes[i, 2].set_title("Prediction")

    for j in range(3):
        axes[i, j].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
