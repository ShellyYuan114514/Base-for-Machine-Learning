
import torch as tc
from torch.utils.data import DataLoader
from components import ResViTSeg
from loss import DiceBCELoss
from data import BraTSData
import numpy as np
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from medpy.metric.binary import hd95

def train():
    BATCH_SIZE = 32
    EPOCHS = 150
    LR = 1e-3
    DEVICE = 'cuda' if tc.cuda.is_available() else 'cpu'

    #  加载训练集与验证集
    train_dataset = BraTSData('Unet\\Dataset\\train_image.npy', 'Unet\\Dataset\\train_mask.npy')
    val_dataset   = BraTSData('Unet\\Dataset\\val_images.npy', 'Unet\\Dataset\\val_masks.npy')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=True)

    

    model = ResViTSeg(n=1).to(DEVICE)

    
    
    criterion = DiceBCELoss()
    optimizer = tc.optim.AdamW(model.parameters(),lr=LR,
                            weight_decay=1e-5,)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                  patience=5,min_lr=6e-6)

    best_val_dice = 0.0
    print(model)

    trainloss =[]
    traindice =[]
    
    valloss =[]
    valdice =[]
    valhaus =[]
    
    for epoch in range(EPOCHS):
        
        if epoch < math.floor(EPOCHS * 0.5):
            alpha = 0.4
        elif epoch < math.floor(EPOCHS*0.7):

            alpha = 0.3
        else :
            alpha = 0.2

        #  Training
        model.train()
        total_loss = 0.0
        total_dice = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks,alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            probs = tc.sigmoid(outputs)
            preds = (probs > 0.5).float()
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2. * intersection + 1e-5) / (union + 1e-5)
            total_dice += dice.sum().item()


        avg_train_loss = total_loss / len(train_loader)
        avg_train_dice = total_dice/len(train_dataset)
        trainloss.append(avg_train_loss)
        traindice.append(avg_train_dice)

        #  Validation
        model.eval()
        total_dice = 0.0
        total_haus=0
        count=0
        totalloss = 0.0
        with tc.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs,masks)
                totalloss += loss.item()
                probs = tc.sigmoid(outputs)
                preds = (probs > 0.5).float()

                intersection = (preds * masks).sum(dim=(1, 2, 3))
                union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                dice = (2. * intersection + 1e-5) / (union + 1e-5)
                total_dice += dice.sum().item()
                B = preds.shape[0]
                for i in range(B):
                    pred_np = preds[i, 0].cpu().numpy().astype(bool)
                    mask_np = masks[i, 0].cpu().numpy().astype(bool)
                    try:
                        hd = hd95(pred_np, mask_np)
                        total_haus += hd
                        count += 1
                    except RuntimeError:
                        continue

            

        avg_val_dice = total_dice / len(val_dataset)
        avg_val_loss = totalloss /len(val_loader )
        avg_val_huas = total_haus/count if count>0 else float('nan')
        valhaus.append(avg_val_huas)
        valdice.append(avg_val_dice)
        valloss.append(avg_val_loss)
           
        
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f}",
              f"|Train Dice :{avg_train_dice:.4f}\n",
              f"Val Dice: {avg_val_dice:.4f}| ",
              f"Val Huas : {avg_val_huas:.4f}| Val loss ={avg_val_loss:.4f}" )
        scheduler.step(avg_val_dice)
        # 
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            tc.save(model.state_dict(), 
                    "Unet\\StateDict\\dicfinal_ma_res_vit.pth")
        
    
    np.save('Unet\\Dataset\\valLossfinal_ma_res_vit',valloss)
    np.save('Unet\\Dataset\\trainLossfinal_ma_res_vit',trainloss)
    np.save('Unet\\Dataset\\valdicefinal_ma_res_vit',valdice)
    np.save('Unet\\Dataset\\traindicefinal_ma_res_vit',traindice)
    np.save('Unet\\Dataset\\valhausfinal_ma_res_vit',valhaus)

if __name__ == "__main__":
    train()
    
