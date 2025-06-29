from sklearn.model_selection import train_test_split
import numpy as np

# 加载原始数据
images = np.load('Unet\\Dataset\\image.npy')  # shape: (N, 240, 240)
masks = np.load('Unet\\Dataset\\mask.npy')    # shape: (N, 240, 240)

# 随机划分训练 / 验证（固定随机种子）
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.1, random_state=42
)
print(len(images),len(train_imgs),len(val_imgs))
# 保存
np.save('Unet\\Dataset\\train_image.npy', train_imgs)
np.save('Unet\\Dataset\\train_mask.npy', train_masks)
np.save('Unet\\Dataset\\val_images.npy', val_imgs)
np.save('Unet\\Dataset\\val_masks.npy', val_masks)