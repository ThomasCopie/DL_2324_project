import torch
import torch.nn as nn
import torch.nn.functional as F

class Model1(nn.Module):
    """takes num_entries 1D signals of length len_signal and outputs a vector of length num_classes """
    def __init__(self, num_entries = 12, len_signal = 4980, num_classes = 5):
        super(Model1, self).__init__()
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=21, stride = 10),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128,128, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128)) for _ in range(num_entries)])

        self.output_size = len_signal //10 //2 //2 //2 //2 // 2// 4

        self.fc1 = nn.ModuleList([nn.Linear(128 * self.output_size, num_classes) for _ in range(num_entries)])
        self.fc2 = nn.Linear(num_entries * num_classes, num_classes)

    def forward(self, x):
        outputs = [conv(x[:, i, :].unsqueeze(1)) for i, conv in enumerate(self.convs)]
        for i, output in enumerate(outputs):
            output = output.view(output.size(0), -1)
            outputs[i] = self.fc1[i](output)
        outputs = torch.cat(outputs, dim=1)
        outputs = self.fc2(outputs)
        return outputs
        
