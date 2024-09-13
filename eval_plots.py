import matplotlib.pyplot as plt
import torch
import os
from collections import OrderedDict
BATCH_NUM = 20 
BATCH_SIZE = 100
ALPHA = "2/255"

results = OrderedDict()

for eps in os.listdir("results/"):
    if eps[-3:]=="003":
        eps_=0.03
    else:
        eps_ = float(eps[-2]+"." + eps[-1])
    results[eps_] = torch.load(f"results/{eps}")

plt.figure(0)
plt.title(f"Model accuracy on adversial images for various epsilon values\n \
          Batch number {BATCH_NUM}, Batch Size {BATCH_SIZE}, Alpha {ALPHA}")

x= []
y = []
for k, v in sorted(results.items()):
    x.append(float(k))
    y.append(v["adv_acc"]*100)
line = plt.plot(x,y, linestyle='--', marker='o')
for i in range(len(x)):
    plt.annotate(f"{str(round(y[i],3))}",(x[i],y[i]),xycoords='data',
                 xytext=(-20,20), textcoords='offset points',color="r",fontsize=12,
                 arrowprops=dict(arrowstyle="->", color='black'))
plt.ylabel("Attack Accuracy (%)")
plt.xlabel("Epsilon Values")
plt.savefig("res.png")