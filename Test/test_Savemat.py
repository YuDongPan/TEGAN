# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/9/18 19:34
import argparse
import numpy as np
import torch
import scipy.io as scio
from sklearn.metrics import confusion_matrix
from Model import CCA_SSVEP
import Utils.EEGDataset as EEGDataset
import Model.TEGAN_V7 as TEGAN
from Utils import Script


'''                  Fs    Nc    Nh     Nf    Ns   low  high
        Direction:  100    9     400    4     54    4    40
             BETA:  250    9     160    40    70    2    90
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BETA', help="Direction or BETA")
parser.add_argument('--factor', type=int, default=2, help="factor size of ssvep")
parser.add_argument('--ws', type=float, default=1.0, help="window size of ssvep")
parser.add_argument('--Nh', type=int, default=160, help="number of trial")
parser.add_argument('--Nc', type=int, default=9, help="number of channel")
parser.add_argument('--Fs', type=int, default=250, help="frequency of sample")
parser.add_argument('--Nf', type=int, default=40, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=70, help="number of subjects")
parser.add_argument('--low', type=int, default=7, help="minimum filtering frequency")
parser.add_argument('--high', type=int, default=90, help="maximum filtering frequency")
opt = parser.parse_args()

for subject in range(1, opt.Ns + 1):
    train_dataset = EEGDataset.getBETAIntra(subject, win_size=round(opt.ws * opt.factor), train_ratio=0.8, mode="train")
    test_dataset = EEGDataset.getBETAIntra(subject, win_size=round(opt.ws * opt.factor), train_ratio=0.8, mode="test")

    eeg_train, label_train = train_dataset[:]
    eeg_test, label_test = test_dataset[:]

    eeg_train = eeg_train[:, :, :, :int(opt.Fs * opt.ws)]   # (160, 1, 9, 250)
    eeg_test = eeg_test[:, :, :, :int(opt.Fs * opt.ws)]  # (160, 1, 9, 250)

    eeg_generator = TEGAN.Generator(opt.Nc, int(opt.Fs * opt.ws), opt.Nf, factor=opt.factor)
    eeg_generator.load_state_dict(torch.load(f'../Pretrain/{opt.dataset}/TEGAN_G_S{subject}.pth'))

    eeg_train = eeg_generator(eeg_train.float()).detach()   # (160, 1, 9, 250) ->  (160, 1, 9, 500)
    eeg_test = eeg_generator(eeg_test.float()).detach()  # (160, 1, 9, 250) ->  (160, 1, 9, 500)


    eeg_train = eeg_train.squeeze(1).numpy()   # (160, 1, 9, 500)  -> (160, 9, 500)
    eeg_test = eeg_test.squeeze(1).numpy()  # (160, 1, 9, 500)  -> (160, 9, 500)

    num_trial = opt.Nh // opt.Nf
    save_data = np.zeros((opt.Nf, opt.Nc, round(opt.factor * opt.ws * opt.Fs), num_trial))

    num_trial_train = eeg_train.shape[0] // opt.Nf
    num_trial_test = eeg_test.shape[0] // opt.Nf


    for f in range(opt.Nf):
       for h in range(num_trial):
           if h < num_trial_train:
              save_data[f, :, :, h] = eeg_train[f * num_trial_train + h, :, :]
           else:
              save_data[f, :, :, h] = eeg_test[f * num_trial_test + (h - num_trial_train), :, :]

    print("save_data.shape:", save_data.shape)

    save_file = f'../Generation/MAT/{opt.dataset}/S{subject}.mat'
    scio.savemat(save_file, {'eeg_data': save_data})

    print(f"save mat file of subject {subject} success!")

    '''
    # filter data, remove 0 Hz
    eeg_train = Scripts.filter_Data(eeg_train, opt.Fs, low=opt.low, high=opt.high, type="highpass")
    eeg_test = Scripts.filter_Data(eeg_test, opt.Fs, low=opt.low, high=opt.high, type="highpass")

    print("eeg_train.shape:", eeg_train.shape)
    print("eeg_test.shape:", eeg_test.shape)

    CCA = CCA_SSVEP.CCA_Base(opt=opt)
    if opt.dataset == 'Direction':
        targets = [12.0, 8.57, 6.67, 5.45]

    if opt.dataset == 'BETA':
        targets = [8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0,
                   10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6,
                   11.8, 12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2,
                   13.4, 13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8,
                   15.0, 15.2, 15.4, 15.6, 15.8, 8.0, 8.2, 8.4]
    labels, predicted_labels = CCA.cca_classify(targets, eeg_test, train_data=eeg_train,
                                                template=True)
    c_mat = confusion_matrix(labels, predicted_labels)
    accuracy = np.divide(np.trace(c_mat), np.sum(np.sum(c_mat)))
    print(f'Subject: {subject}, Classification Accuracy:{accuracy:.3f}')
    '''


