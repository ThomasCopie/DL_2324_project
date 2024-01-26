import numpy as np
import scipy.signal as signal
from tqdm import tqdm
import os

def denoise(s, fmin, fmax):
    filter = signal.butter(6, [fmin, fmax], 'bandpass', fs = 500, output = 'sos')
    res = signal.sosfilt(filter, s)
    return res

path = os.getcwd() + "01/0/"
save_path = os.getcwd() + "filtered_samples/"
files = os.listdir(path)
csv_files = [file for file in files if file.endswith('.csv')]


fmin = .5
fmax = 40
for file in tqdm(csv_files):
    my_sigs = np.loadtxt(path + file, delimiter = ';')
    filtered_sigs = np.array(denoise(my_sigs, fmin, fmax)[:, 10:-10], dtype = int)
    np.savetxt(save_path + file, filtered_sigs, delimiter = ";", fmt = "%d")
