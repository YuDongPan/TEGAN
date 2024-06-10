# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/9/18 23:58
import torch
import argparse
import numpy as np
import Utils.EEGDataset as EEGDataset
from sklearn.metrics import confusion_matrix
from Model import CCA_SSVEP
from Model import TEGAN_V7 as TEGAN
from Utils import Ploter, Script
import scipy.io

# 1、Define parameters of eeg data
'''                  Fs    Nc    Nh     Nf    Ns   low  high
        Direction:  100    10    400    4     54    4    40
             BETA:  250    9     160    40    70    7    90
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Direction', help="Direction or BETA")
parser.add_argument('--factor', type=int, default=2, help="factor size of ssvep")
parser.add_argument('--Kf', type=int, default=1, help="k-fold cross validation")
parser.add_argument('--ws', type=float, default=1.0, help="window size of ssvep")
parser.add_argument('--Nh', type=int, default=100, help="number of trial")
parser.add_argument('--Nc', type=int, default=10, help="number of channel")
parser.add_argument('--Fs', type=int, default=100, help="frequency of sample")
parser.add_argument('--Nt', type=int, default=400, help="number of time points")
parser.add_argument('--Nf', type=int, default=4, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=54, help="number of subjects")
parser.add_argument('--low', type=int, default=4, help="minimum filtering frequency")
parser.add_argument('--high', type=int, default=40, help="maximum filtering frequency")
opt = parser.parse_args()


# 2、Start Train
final_acc_list = []
for fold_num in range(opt.Kf):
    final_valid_acc_list = []
    print(f"Training for K_Fold {fold_num + 1}")
    for subject in range(1, opt.Ns + 1):
        # -----------------------------------------Method1------------------------------------------------
        if opt.dataset == 'Direction':
            train_dataset = EEGDataset.getSSVEP4Intra(subject, train_ratio=0.8, mode="train")
            test_dataset = EEGDataset.getSSVEP4Intra(subject, train_ratio=0.8, mode="test")

        elif opt.dataset == 'BETA':
            train_dataset = EEGDataset.getBETAIntra(subject, win_size=round(opt.ws * opt.factor), train_ratio=0.8,
                                                    mode="train")
            test_dataset = EEGDataset.getBETAIntra(subject, win_size=round(opt.ws * opt.factor), train_ratio=0.8,
                                                   mode="test")

        eeg_train, label_train = train_dataset[:]
        eeg_test, label_test = test_dataset[:]
        eeg_train = eeg_train[:, :, :, :int(opt.Fs * opt.ws)]
        eeg_test = eeg_test[:, :, :, :int(opt.Fs * opt.ws)]

        eeg_generator = TEGAN.Generator(opt.Nc, int(opt.Fs * opt.ws), opt.Nf, factor=opt.factor)
        eeg_generator.load_state_dict(torch.load(f'../Pretrain/{opt.dataset}/TEGAN_G_S{subject}.pth'))
        eeg_generator.eval()
        eeg_train = eeg_generator(eeg_train.float()).detach()
        eeg_test = eeg_generator(eeg_test.float()).detach()

        # -----------------------------------------Method2------------------------------------------------
        # if opt.dataset == 'Direction':
        #     train_dataset = EEGDataset.getSSVEP4Intra(subject, train_ratio=1.0, mode="train")
        #
        # elif opt.dataset == 'BETA':
        #     train_dataset = EEGDataset.getBETAIntra(subject, win_size=round(opt.ws * opt.factor), train_ratio=1.0,
        #                                             mode="train")
        #
        # EEGData, LabelData = train_dataset[:]
        #
        # eeg_data = EEGData[:, :, :, :int(opt.Fs * opt.ws)]
        # eeg_generator = TEGAN.Generator(opt.Nc, int(opt.Fs * opt.ws), opt.Nf, factor=opt.factor)
        # eeg_generator.load_state_dict(torch.load(f'../Pretrain/{opt.dataset}/TEGAN_G_S{subject}.pth'))
        # eeg_generator.eval()
        # eeg_data = eeg_generator(eeg_data.float()).detach()
        #
        # train_idx = []
        # test_idx = []
        #
        # num_trial = opt.Nh // opt.Nf
        # # print("num_trial:", num_trial)
        # for f in range(opt.Nf):
        #     for h in range(num_trial):
        #         if h < round(num_trial * 0.8):
        #             train_idx.append(f * num_trial + h)
        #         else:
        #             test_idx.append(f * num_trial + h)
        #
        # eeg_train = eeg_data[train_idx]
        # eeg_test = eeg_data[test_idx]

        # -----------------------------------------Method3------------------------------------------------
        # if opt.dataset == 'Direction':
        #     train_dataset = EEGDataset.getSSVEP4Intra(subject, train_ratio=1.0, mode="train")
        #
        # elif opt.dataset == 'BETA':
        #     train_dataset = EEGDataset.getBETAIntra(subject, win_size=round(opt.ws * opt.factor), train_ratio=1.0,
        #                                             mode="train")
        #
        # EEGData, _ = train_dataset[:]
        # eeg_data = EEGData[:, :, :, :int(opt.Fs * opt.ws)]
        # eeg_generator = TEGAN.Generator(opt.Nc, int(opt.Fs * opt.ws), opt.Nf, factor=opt.factor)
        # eeg_generator.eval()
        # eeg_generator.load_state_dict(torch.load(f'../Pretrain/{opt.dataset}/TEGAN_G_S{subject}.pth'))
        # eeg_gen_data = torch.zeros((opt.Nh, 1, opt.Nc, int(opt.Fs * opt.ws * opt.factor)))
        # for i in range(opt.Nh):
        #     single_eeg = eeg_data[i].unsqueeze(0).float()
        #     eeg_gen_data[i] = eeg_generator(single_eeg).detach()
        #
        # train_idx = []
        # test_idx = []
        #
        # num_trial = opt.Nh // opt.Nf
        # # print("num_trial:", num_trial)
        # for f in range(opt.Nf):
        #     for h in range(num_trial):
        #         if h < round(num_trial * 0.8):
        #             train_idx.append(f * num_trial + h)
        #         else:
        #             test_idx.append(f * num_trial + h)
        #
        # eeg_train = eeg_gen_data[train_idx]
        # eeg_test = eeg_gen_data[test_idx]

        # ----------------------------------------------------------------------------------------

        # squeeze the empty dimension
        eeg_train = eeg_train.squeeze(1).numpy()
        eeg_test = eeg_test.squeeze(1).numpy()

        # filter data, remove 0 Hz
        eeg_train = Scripts.filter_Data(eeg_train, opt.Fs, low=opt.low, high=opt.high, type="highpass")
        eeg_test = Scripts.filter_Data(eeg_test, opt.Fs, low=opt.low, high=opt.high, type="highpass")

        # -----------------------------------------------------------------------------------------------------------
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
        final_valid_acc_list.append(accuracy)

    final_acc_list.append(final_valid_acc_list)

# 3、Plot result
Ploter.plot_save_Result(final_acc_list, model_name='CCA', dataset=opt.dataset, UD=0, ratio=1, win_size=str(opt.ws),
                        text=True)
