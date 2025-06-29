
# necessary moudel

import h5py
import numpy as np
import os
from tqdm import tqdm

#read data 

dir  = "Unet\\data"

img = []
msk = []

for fname in tqdm(os.listdir(dir)):
    if not fname.endswith('.h5'):
        continue
    path = os.path.join(dir,fname)

    with h5py.File(path,'r') as file:
        image = file['image'][:]
        mask = file['mask'][:]
        t1 = image[:,:,0]
        wt = ((mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]) > 0).astype(np.uint8)
        img.append(t1)
        msk.append(wt)

img = np.stack(img)
msk = np.stack(msk)

np.save('Unet\Dataset\image.npy',img)
np.save('Unet\Dataset\mask.npy',msk)

import matplotlib.pyplot as plt

msk = np.load('Unet/Dataset/mask.npy')

for i in range(5):
    print(f"Sample {i}, Unique values: {np.unique(msk[i])}")
    plt.imshow(msk[i], cmap='gray')
    plt.title(f"Sample {i}")
    plt.axis('off')
    plt.show()