import numpy as np 
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from models import Model1
from Dataset import SignalDataset
from time import time

def main():
    num_classes = 5
    model = Model1(num_classes=num_classes)
    model_state = "model_saves/5_classes/model2.pt"
    model.load_state_dict(torch.load(model_state))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.eval()

    data = SignalDataset("filtered_samples", "classifications_reduced_2")
    test_indices = np.loadtxt("sets/5_classes/test_ind.txt", dtype=int)
    #val_indices = np.loadtxt("sets/22_classes/val_ind.txt", dtype=int)
    #full_test_indices = np.concatenate((test_indices, val_indices))
    test_loader = DataLoader(data, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(test_indices))

    class_stats = np.zeros((num_classes, 3)) # TP, FP, FN
    confusion_matrix = np.zeros((num_classes, num_classes))

    correct = 0

    for signal, target in test_loader:
        signal = signal.float()
        signal, target = signal.to(device), target.to(device)
        output = model(signal)
        preds = torch.sigmoid(output) > 0.5
        for t, p in zip(target, preds):
            if (t == p).sum() == num_classes:
                correct += 1
            for i in range(num_classes):
                if t[i] == 1 and p[i] == 1:
                    class_stats[i, 0] += 1
                    confusion_matrix[i, i] += 1
                elif t[i] == 0 and p[i] == 1:
                    class_stats[i, 1] += 1
                    confusion_matrix[:, i] += 1
                    confusion_matrix[i, i] -= 1
                elif t[i] == 1 and p[i] == 0:
                    class_stats[i, 2] += 1
                    confusion_matrix[i, :] += 1
                    confusion_matrix[i, i] -= 1
    for i in range(num_classes):
        tot_pos = class_stats[i, 0] + class_stats[i, 2]
        class_stats[i, 0] /= tot_pos
        class_stats[i, 1] /= tot_pos
        class_stats[i, 2] /= tot_pos
    
    print(f"Accuracy : {correct}/{len(test_indices)} ({100.0 * correct / len(test_indices):.0f}%)")
    print(f"Class stats : \n{class_stats}\n")
    print(f"Confusion matrix : \n{confusion_matrix}")
    fig = plt.figure(figsize=(5, 15))
    ax1 = fig.add_subplot(111)
    cax1 = ax1.matshow(class_stats)
    fig.colorbar(cax1)
   
    ax1.set_yticklabels([''] + list(range(num_classes)))
    ax1.set_xticklabels([''] + ['TP', 'FP', 'FN'])

    plt.show()
    #fig.savefig("5_class_stats.png")

if __name__ == "__main__":
    main()
