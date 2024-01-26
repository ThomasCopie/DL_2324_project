from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class SignalDataset(Dataset):
    """Signal dataset."""

    def __init__(self, signal_dir, classes_dir):
        self.file_names = sorted(os.listdir(classes_dir))
        self.signal_dir = signal_dir
        self.classes_dir = classes_dir

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        signal = np.loadtxt(os.path.join(self.signal_dir, file_name), delimiter=';')
        class_label = np.loadtxt(os.path.join(self.classes_dir, file_name))

        return torch.from_numpy(signal), torch.from_numpy(class_label)

def separate_sets(dataset, ptrain=0.8, pval=0.1, ptest= 0.1):
    indices = np.array(list(range(len(dataset))))
    np.random.shuffle(indices)

    train_indices = indices[:int(ptrain * len(dataset))]
    val_indices = indices[int(ptrain * len(dataset)):int((ptrain + pval) * len(dataset))]
    test_indices = indices[int((ptrain + pval) * len(dataset)):]
    return train_indices, val_indices, test_indices

