

# modules

import torch as tc
from torch .utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 定义增强
transform = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.4),
    A.RandomRotate90(p=0.5),
    A.Normalize(), 
     #A.ElasticTransform(p=0.2),
    ToTensorV2()
])
class BraTSData(Dataset):

    def __init__(self, ImgPath,MskPath,transform = None ):
        self.images = np.load(ImgPath)
        self.masks = np.load(MskPath)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        
        image = image.astype(np.float32)

        mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = tc.from_numpy(image).unsqueeze(0).float()
            mask = tc.from_numpy(mask).unsqueeze(0).float()        



        return image,mask


