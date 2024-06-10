# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/9/26 21:01
import argparse
import numpy as np
import torch
import Utils.EEGDataset as EEGDataset
import Utils.Tools as Tools
from Utils import Script

# 1、Define parameters of eeg data
'''                  Fs    Nc    Nh     Nf    Ns   low  high
        Direction:  100    9     400    4     54    4    40
             BETA:  250    9     160    40    70    2    90
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Direction', help="Direction or BETA")
parser.add_argument('--factor', type=int, default=2, help="factor size of ssvep")
parser.add_argument('--ws', type=float, default=1.0, help="window size of ssvep")
parser.add_argument('--Nh', type=int, default=100, help="number of trial")
parser.add_argument('--Nc', type=int, default=10, help="number of channel")
parser.add_argument('--Fs', type=int, default=100, help="frequency of sample")
parser.add_argument('--Nf', type=int, default=4, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=54, help="number of subjects")
parser.add_argument('--low', type=int, default=4, help="minimum filtering frequency")
parser.add_argument('--high', type=int, default=40, help="maximum filtering frequency")
opt = parser.parse_args()

# 2.Prepare eeg data
subject = 10  # 10 12/36/49
class_num = 0
if opt.dataset == 'Direction':
    # train_dataset = EEGDataset.getSSVEP4Intra(subject, train_ratio=0.2, mode="test")
    train_dataset = EEGDataset.getSSVEP4Intra(subject, KFold=0, n_splits=5, mode="train")

elif opt.dataset == 'Dial':
    # train_dataset = EEGDataset.getSSVEP12Intra(subject, train_ratio=0.2, mode="train")
    train_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=0, n_splits=5, mode="train")

eeg_train, label_train = train_dataset[:]
source_eeg = eeg_train[:, :, :, :int(opt.Fs * opt.ws)]
target_eeg = eeg_train[:, :, :, :int(opt.Fs * opt.ws * opt.factor)]

num_trial = eeg_train.shape[0] // opt.Nf

source_eeg = torch.mean(source_eeg[class_num * num_trial: (class_num + 1) * num_trial], dim=0).unsqueeze(0)
target_eeg = torch.mean(target_eeg[class_num * num_trial: (class_num + 1) * num_trial], dim=0).unsqueeze(0)

print("source_eeg.shape:", source_eeg.shape)
print("target_eeg.shape:", target_eeg.shape)

# 3.Start plotting
Tools.plot_EEG_CMP(opt, subject, class_num, source_eeg, target_eeg, source=False)
