import numpy as np
import matplotlib.pyplot as plt

vallos = np.load('Unet\\Dataset\\valLossfinal_ma_res_vit.npy')
trainloss =np.load('Unet\\Dataset\\trainLossfinal_ma_res_vit.npy')
valdice = np.load('Unet\\Dataset\\valdicefinal.npy')
traindice = np.load('Unet\\Dataset\\traindicefinal_ma_res_vit.npy')
epochs = [i+1 for i in range(150)]





print(valdice.max())


plt.plot(epochs,valdice,color="red",label = "valdice")
plt.plot(epochs,trainloss,color="blue",label = "trainloss")
plt.plot(epochs,vallos,color="orange",label = "valloss")
plt.plot(epochs,traindice,color = 'indigo',label = "traindice")
plt.axhline(y=0.87, color='gray', linestyle='--', linewidth=1,)
plt.axhline(y=0.75, color='gray', linestyle='--', linewidth=1,)
plt.text(1, 0.87, '0.87', color='gray', va='bottom', ha='left')
plt.text(1, 0.75,'0.75', color='gray', va='bottom', ha='left')

plt.title("training curve")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()