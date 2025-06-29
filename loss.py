import torch
import torch .nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self,smooth = 1e-5,alpha=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))
        self.smooth = smooth
        self.alpha = alpha
    

    def forward(self, logits, targets,alpha=None):
        # logits: (B, 1, H, W) => raw output
        # targets: (B, 1, H, W) => binary ground truth
        if alpha is None:
            alpha = self.alpha

        probs = torch.sigmoid(logits)
        bce_loss = self.bce(logits, targets)

        # Flatten
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.alpha* bce_loss + (1-self.alpha)*dice_loss  
    
      