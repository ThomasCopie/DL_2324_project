import numpy as np 
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from models import Model1, Model2
from Dataset import SignalDataset
from time import time

N_classes = 5
model_nb = 2

def train(model, optimizer, train_loader, use_cuda, epoch, num_classes, train_indices):
    model.train()
    correct = 0
    full_correct = 0
    for batch_idx, (signal, target) in enumerate(train_loader):
        signal = signal.float()
        if use_cuda:
            signal, target = signal.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(signal)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = (output.data > 0.5).float()
        comparaison = (pred == target).float().sum(dim=1) / num_classes
        correct += comparaison.cpu().sum()
        comparaison_2 = (pred == target).float().sum(dim=1) == num_classes
        full_correct += comparaison_2.float().cpu().sum()

        if batch_idx % 10 == 0:
            print( f"Train Epoch: {epoch} [{batch_idx * len(signal)}/{len(train_indices)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data.item():.6f}")
    print(f"\nTrain set: Accuracy: {correct}/{len(train_indices)} ({100.0 * correct / len(train_indices):.0f}%), Exact : {full_correct}/{len(train_indices)} ({100.0 * full_correct / len(train_indices):.0f})%\n")

def test(model, test_loader, use_cuda, num_classes, test_indices):
    model.eval()
    test_loss = 0
    correct = 0
    full_correct = 0
    with torch.no_grad():
        for signal, target in test_loader:
            signal = signal.float()
            if use_cuda:
                signal, target = signal.cuda(), target.cuda()
            output = model(signal)
            criterion = torch.nn.BCEWithLogitsLoss()
            test_loss += criterion(output, target).data.item()
            pred = (output.data > 0.5).float()
            comparaison = (pred == target).float().sum(dim=1) / num_classes
            correct += comparaison.cpu().sum()
            comparaison_2 = (pred == target).float().sum(dim=1) == num_classes
            full_correct += comparaison_2.float().cpu().sum()

    test_loss /= len(test_indices)
    print(f"""\nTest set: Average loss: {test_loss:.4f}, 
          Accuracy: {correct}/{len(test_indices)} ({100.0 * correct / len(test_indices):.0f}%, 
          Exact : {full_correct}/{len(test_indices)} ({100.0 * full_correct / len(test_indices):.0f})%\n""")


def main():
    t0 = time()

    if N_classes == 22:
        data = SignalDataset("filtered_samples", "classifications_reduced")
        train_indices = np.loadtxt("sets/22_classes/train_ind.txt", dtype=int)
        val_indices = np.loadtxt("sets/22_classes/val_ind.txt", dtype=int)
        test_indices = np.loadtxt("sets/22_classes/test_ind.txt", dtype=int)
    elif N_classes == 5:
        data = SignalDataset("filtered_samples", "classifications_reduced_2")
        train_indices = np.loadtxt("sets/5_classes/train_ind.txt", dtype=int)
        val_indices = np.loadtxt("sets/5_classes/val_ind.txt", dtype=int)
        test_indices = np.loadtxt("sets/5_classes/test_ind.txt", dtype=int)
    else: 
        print("N_classes must be 22 or 5")
        return
    train_loader = DataLoader(data, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(train_indices))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if model_nb == 1:
        model = Model1(12, 4980, N_classes)
    elif model_nb == 2:
        model = Model2(12, 4980, N_classes)
    else:
        print("model_nb must be 1 or 2")
        return
    #state_dict = torch.load("model_saves/5_classes/model1.pt")
    #model.load_state_dict(state_dict)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f"Number of parameters : {num_params}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    for epoch in range(1, 6):
        train(model, optimizer, train_loader, True, epoch, N_classes, train_indices)
        t1 = time()
        print(f"Elapsed time : {t1 - t0:.2f} seconds")
        temp_val_indices = np.random.choice(val_indices, 1000)
        val_loader = DataLoader(data, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(temp_val_indices))
        test(model, val_loader, True, N_classes, temp_val_indices)

    val_loader = DataLoader(data, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(val_indices))
    test(model, val_loader, True, N_classes, val_indices)
    models = sorted(os.listdir(f"model_saves/{N_classes}_classes"))
    new_name = f"model_saves/{N_classes}_classes/model0.pt"
    if len(models) > 0:
        last_model_version = models[-1]
        new_name = f"model_saves/{N_classes}_classes/model" + str(int(last_model_version[5]) + 1) + ".pt"
    if model_nb == 1:
        torch.save(model.state_dict(), new_name)
    elif model_nb == 2:
        torch.save(model.state_dict(), "model_saves/model_2/model1.pt")

if __name__ == "__main__":
    main()
