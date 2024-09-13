import matplotlib.pyplot as plt
import torch
import numpy as np
BATCH_NUM = 100
BATCH_SIZE = 100
ALPHA = "1e-4"
STEPS = 20


results=  torch.load(f"finetune_results/train_01_test003", map_location=torch.device('cpu'))
plt.figure(0)
plt.title(f"Finetuning Loss using eps=0.1 (train, {BATCH_NUM*BATCH_SIZE} adversial images) \n eps=0.03 (test) {BATCH_NUM*BATCH_SIZE} adversial images\n LR={ALPHA}")

train_loss = results["train_loss"]
test_loss = results["test_losses"]
train_acc = results["train_acc"]
test_acc = results["test_accs"]


plt.plot(list(range(STEPS)),train_loss, label="train")
plt.plot(list(range(STEPS)), test_loss,  label="tes")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.savefig("res_loss.png")

plt.figure(1)
plt.title(f"Finetuning Accuracy using eps=0.1 (train) and eps=0.03 (test)")
plt.plot(list(range(STEPS)),np.array(train_acc)*100)
plt.plot(list(range(STEPS)),np.array(test_acc)*100)
plt.ylabel("Attack Accuracy (%)")
plt.xlabel("Epochs")
plt.savefig("res_acc.png")