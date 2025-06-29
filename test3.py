import numpy as np
import matplotlib.pyplot as plt
v1 = np.load("Unet\\Dataset\\valhausfinal.npy")
v2 = np.load("Unet\\Dataset\\valhausfinal_ma.npy")
v3 = np.load("Unet\\Dataset\\valhausfinal_ma_res.npy")
v4 = np.load("Unet\\Dataset\\valhausfinal_ma_res_vit.npy")

print(v1.min())

epochs = [i+1 for i in range(150)]
plt.plot(epochs,v4,color="black",label = "ma_res")
plt.plot(epochs,v1,color="red",label = "original")
plt.plot(epochs,v2,color="blue",label = "ma_vit")
plt.plot(epochs,v3,color="orange",label = "ma_res&vit")


plt.title("training curve2")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()