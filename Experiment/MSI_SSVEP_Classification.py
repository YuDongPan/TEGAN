# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/13 22:12
import torch
import argparse
import numpy as np
import Utils.EEGDataset as EEGDataset
from sklearn.metrics import confusion_matrix
from Model import MSI_SSVEP
from Model import TEGAN as TEGAN
from Utils import Ploter

'''                Fs    Nc     Nh     Nf     Ns   low   high
      Direction:  100    9     100     4     54    4     40
           Dial:  256    8     180     12    10    6     80
'''
parser = argparse.ArgumentParser()

'''Direction Dataset'''
# parser.add_argument('--dataset', type=str, default='Direction', help="4-class dataset")
# parser.add_argument('--factor', type=int, default=2, help="factor size of ssvep")
# parser.add_argument('--ws', type=float, default=1.0, help="window size of ssvep")
# parser.add_argument('--Kf', type=int, default=1, help="k-fold cross validation")
# parser.add_argument('--Nh', type=int, default=100, help="number of trial")
# parser.add_argument('--Nc', type=int, default=10, help="number of channel")
# parser.add_argument('--Fs', type=int, default=100, help="frequency of sample")
# parser.add_argument('--Nt', type=int, default=400, help="number of sample")
# parser.add_argument('--Nf', type=int, default=4, help="number of stimulus")
# parser.add_argument('--Ns', type=int, default=54, help="number of subjects")
# parser.add_argument('--UD', type=int, default=-1, help="-1(Unsupervised),0(User-dependent),1(User-Indepedent)")
# parser.add_argument('--ratio', type=int, default=-1, help="-1(Training-free),0(N-1vs1),1(R1),2(5v5),3(2v8)")


'''Dial Dataset'''
parser.add_argument('--dataset', type=str, default='Dial', help="12-class dataset")
parser.add_argument('--factor', type=int, default=2, help="factor size of ssvep")
parser.add_argument('--ws', type=float, default=1.0, help="window size of ssvep")
parser.add_argument('--Kf', type=int, default=1, help="k-fold cross validation")
parser.add_argument('--Nh', type=int, default=180, help="number of trial")
parser.add_argument('--Nc', type=int, default=8, help="number of channel")
parser.add_argument('--Fs', type=int, default=256, help="frequency of sample")
parser.add_argument('--Nt', type=int, default=512, help="number of sample")
parser.add_argument('--Nf', type=int, default=12, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=10, help="number of subjects")
parser.add_argument('--UD', type=int, default=-1, help="-1(Unsupervised),0(User-dependent),1(User-Indepedent)")
parser.add_argument('--ratio', type=int, default=-1, help="-1(Training-free),0(N-1vs1),1(R1),2(5v5),3(2v8)")

opt = parser.parse_args()

# 2、Start Train
final_acc_list = []
for fold_num in range(opt.Kf):
    final_valid_acc_list = []
    print(f"Training for K_Fold {fold_num + 1}")
    for subject in range(1, opt.Ns + 1):
        if opt.dataset == 'Direction':
            train_dataset = EEGDataset.getSSVEP4Intra(subject, train_ratio=0.0, mode="train")
            test_dataset = EEGDataset.getSSVEP4Intra(subject, train_ratio=0.0, mode="test")
            # train_dataset = EEGDataset.getSSVEP4Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="train")
            # test_dataset = EEGDataset.getSSVEP4Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="test")

        elif opt.dataset == 'Dial':
            train_dataset = EEGDataset.getSSVEP12Intra(subject, train_ratio=0.0, mode="train")
            test_dataset = EEGDataset.getSSVEP12Intra(subject, train_ratio=0.0, mode="test")
            # train_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="test")
            # test_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="train")

        eeg_train, label_train = train_dataset[:]
        eeg_test, label_test = test_dataset[:]
        eeg_train = eeg_train[:, :, :, :int(opt.Fs * opt.ws)]
        eeg_test = eeg_test[:, :, :, :int(opt.Fs * opt.ws)]

        # -------------------------------Augmentation with TEGAN -----------------------------------------------------
        eeg_generator = TEGAN.Generator(opt.Nc, int(opt.Fs * opt.ws), opt.Nf, opt.ws, factor=opt.factor)
        '''Source Generator'''
        eeg_generator.load_state_dict(torch.load(f'../Pretrain/{opt.dataset}/{opt.ws}S-{opt.ws * opt.factor}S/'
                                                 f'Source/TEGAN_Gs_S{subject}.pth'))
        # '''Target Generator'''
        # eeg_generator.load_state_dict(torch.load(f'../Pretrain/{opt.dataset}/{opt.ws}S-{opt.ws * opt.factor}S/'
        #                                         f'Target/R{opt.ratio}/F{fold_num}/TEGAN_Gt_S{subject}.pth'))

        eeg_generator.eval()
        z = eeg_train.float()

        # augment multiple times
        n_times = 1
        eeg_gen_train = torch.zeros(
            (eeg_train.shape[0] * n_times, 1, eeg_train.shape[2], eeg_train.shape[3] * opt.factor))
        for i in range(n_times):
            train_idx = [n_times * h + i for h in range(eeg_train.shape[0])]
            eeg_gen_train[train_idx, :, :, :] = eeg_generator(z)[0].detach()

        eeg_train = eeg_gen_train.detach()
        label_train = label_train.repeat(1, n_times).reshape(-1, 1)
        eeg_test = eeg_generator(eeg_test.float())[0].detach()

        # -----------------------------------------------------------------------------------------------------------

        # squeeze the empty dimension
        eeg_train = eeg_train.squeeze(1).numpy()
        eeg_test = eeg_test.squeeze(1).numpy()

        # filter data, remove 0 Hz
        # eeg_train = Scripts.filter_Data(eeg_train, opt.Fs, low=opt.low, high=opt.high, type="highpass")
        # eeg_test = Scripts.filter_Data(eeg_test, opt.Fs, low=opt.low, high=opt.high, type="highpass")

        # -----------------------------------------------------------------------------------------------------------
        print("eeg_train.shape:", eeg_train.shape)
        print("eeg_test.shape:", eeg_test.shape)
        MSI = MSI_SSVEP.MSI_Base(opt=opt)
        if opt.dataset == 'Direction':
            targets = [12.0, 8.57, 6.67, 5.45]

        elif opt.dataset == 'Dial':
            targets = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                       10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

        labels, predicted_labels = MSI.msi_classify(targets, eeg_test, num_harmonics=3)
        c_mat = confusion_matrix(labels, predicted_labels)
        accuracy = np.divide(np.trace(c_mat), np.sum(np.sum(c_mat)))
        print(f'Subject: {subject}, Classification Accuracy:{accuracy:.3f}')
        final_valid_acc_list.append(accuracy)

    final_acc_list.append(final_valid_acc_list)

# 3、Plot result
save_dataset = opt.dataset if opt.factor <= 1 else opt.dataset + '_AUG'
Ploter.plot_save_Result(final_acc_list, model_name='MSI', dataset=save_dataset, UD=opt.UD, ratio=opt.ratio,
                        win_size=str(opt.ws), text=True)