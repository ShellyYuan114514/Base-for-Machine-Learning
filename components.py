
# modules

import torch as tc
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, InChannel,OutChannel,stride =1,p=0.3):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(InChannel,OutChannel,3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(OutChannel),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.Conv2d(OutChannel,OutChannel,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(OutChannel)
        )

        self.shortcut = nn.Sequential()
        if stride !=1 or InChannel != OutChannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(InChannel,OutChannel,1,stride=stride,bias=False),
                nn.BatchNorm2d(OutChannel)
            )
    def forward(self,x):
        return tc.relu(self.res(x)+self.shortcut(x))
    
    
class encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.InChannel =64
        
        self.stem = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self.MakeLayer(64,2,stride=1)
        self.layer2 = self.MakeLayer(128,4,stride=2)
        self.layer3 = self.MakeLayer(256,6,stride=2)
        self.layer4 = self.MakeLayer(512,8,stride=2)
    
    def MakeLayer(self,OutChannel,blocks,stride):
        layers =[ResBlock(self.InChannel,OutChannel,stride)]
        self.InChannel = OutChannel
        for _ in range(1,blocks):
            layers.append(ResBlock(OutChannel,OutChannel))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1,x2,x3,x4
    
    
class ViTBlock(nn.Module):
    def __init__(self, dim, heads= 8,p=0.15):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim = dim, num_heads = heads, batch_first = True)
        nn.Dropout(p ),
        self.norm1 = nn.LayerNorm(dim)
        self.ff=nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.ReLU(),
            
            nn.Linear(dim*4,dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self,x):
        AttnOut , _ = self.attn(x,x,x)
        x = self.norm1(x + AttnOut)
        FfOut = self.ff(x)
        x = self.norm2(x+FfOut)
        return x 
    

class Bridge(nn.Module):
    def __init__(self, InChannel,blocks=6):
        super().__init__()
        self.flatten = nn.Conv2d(InChannel,InChannel,1)
        self.blocks = nn.Sequential(*[ViTBlock(InChannel) for _ in range(blocks)])

    def forward(self,x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.blocks(x)
        x = x.transpose(1,2).reshape(B,C,H,W)
        return x 
    

class DecoderBlock(nn.Module):
    def __init__(self, InChannel, SkipChannel, OutChannel,p=0.28):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(InChannel + SkipChannel, OutChannel, 3, padding=1),
            nn.BatchNorm2d(OutChannel),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p), 

            nn.Conv2d(OutChannel, OutChannel, 3, padding=1),
            nn.BatchNorm2d(OutChannel),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p), 

            #nn.Conv2d(OutChannel, OutChannel, 3, padding=1),
            #nn.BatchNorm2d(OutChannel),
            #nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = tc.cat([x, skip], dim=1)
        return self.conv(x)
    


class ResViTSeg (nn.Module):
    def __init__(self, n=1):
        super().__init__()
        self.encoder = encoder()
        self.bridge = Bridge(InChannel=512)
        self.dec3 = DecoderBlock(512, 256, 256)  # 输入512, skip连接来自layer3(256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        self.final_conv = nn.Conv2d(64, n, kernel_size=1)

    def forward(self,x):
        
        x1, x2, x3, x4 = self.encoder(x)  # 编码器4层输出 ##
        x_bridge = self.bridge(x4)        # transformer bridge

        d3 = self.dec3(x_bridge, x3)      # 解码器逐级恢复
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        out = nn.functional.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = self.final_conv(out)
        return out



        