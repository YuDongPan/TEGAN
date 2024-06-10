# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/2/29 16:17
import math
import torch
import Utils.EEGDataset as EEGDataset
from Train import Classifier_Trainer, Trainer_Script
from Utils import Ploter, Extender
from etc.global_config import config

# 1、Define parameter of eeg
'''
                Fs    Nc    Nh    Nf    Ns    bz     wd       
        Direction:  100   10    400    4    54    20    0.0001
           Dial:  256    8    180    12   10    20    0.0001
'''

algorithm = config["algorithm"]

UD = config["train_param"]["UD"]
ratio = config["train_param"]["ratio"]
if ratio == 1 or ratio == 3:
    Kf = 5
elif ratio == 2:
    Kf = 2
else:
    Kf = 1

train_mode = "test" if ratio == 3 else "train"
test_mode = "train" if ratio == 3 else "test"

'''Parameters for ssvep data'''
Ds = config["data_param"]["Ds"]
Fs = config["data_param"]["Fs"]
ws = config["data_param"]["ws"]
Nc = config["data_param"]['Nc']
Ns = config["data_param"]['Ns']

'''Parameters for DL-based methods'''
if algorithm != "CCA" and algorithm != "TRCA":
    epochs = config[algorithm]['epochs']
    lr_jitter = config[algorithm]['lr_jitter']

'''Parameters for TEGAN'''
F = config["TEGAN"]["F"]

devices = "cuda" if torch.cuda.is_available() else "cpu"

# 2、Start Training
final_acc_list = []
for fold_num in range(Kf):
    final_test_acc_list = []
    print(f"Training for K_Fold {fold_num + 1}")
    for subject in range(1, Ns + 1):
        # *******************************Loading SSVEP Dataset************************ #
        if Ds == 'Direction':
            train_dataset = EEGDataset.getSSVEP4Intra(subject, KFold=fold_num, n_splits=Kf, mode=train_mode)
            test_dataset = EEGDataset.getSSVEP4Intra(subject, KFold=fold_num, n_splits=Kf, mode=test_mode)

        elif Ds == 'Dial':
            train_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=fold_num, n_splits=Kf, mode=train_mode)
            test_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=fold_num, n_splits=Kf, mode=test_mode)

        eeg_train, label_train = train_dataset[:]
        eeg_test, label_test = test_dataset[:]
        eeg_train_short = eeg_train[:, :, :, :math.ceil(Fs * ws)]
        eeg_test_short = eeg_test[:, :, :, :math.ceil(Fs * ws)]

        eeg_train_long = eeg_train[:, :, :, :math.ceil(Fs * ws * F)]
        eeg_test_long = eeg_test[:, :, :, :math.ceil(Fs * ws * F)]

        # *******************************Augmentation with TEGAN*******************************************
        eeg_train_aug_long, _ = Extender.SSVEP_Extension(subject, eeg_train_short, eeg_test_short)

        augment_choice = 1
        if augment_choice == 1:
            '''1、Only Synthetic Long-length EEG data for training'''
            eeg_train = eeg_train_aug_long
            save_dataset = Ds + '_AUG'

        elif augment_choice == 2:
            '''2、Both Synthetic Long-length EEG and Real Long-length EEG data for training'''
            eeg_train_all = torch.zeros((eeg_train_long.shape[0] * 2, 1, Nc, eeg_train_long.shape[3]))
            train_idx_org = [2 * h for h in range(eeg_train.shape[0])]
            train_idx_aug = [2 * h + 1 for h in range(eeg_train.shape[0])]
            eeg_train_all[train_idx_org] = eeg_train_long.float()
            eeg_train_all[train_idx_aug] = eeg_train_aug_long.float()

            eeg_train = eeg_train_all
            label_train = label_train.repeat(1, 2).reshape(-1, 1)
            save_dataset = Ds + '_EXT_AUG'

        eeg_test = eeg_test_long
        # *******************************Data preprocessing*************************************************
        eeg_train, eeg_test = Trainer_Script.data_preprocess((eeg_train, label_train),
                                                             (eeg_test, label_test))

        # *******************************Classifier Training and Testing************************************
        if algorithm == "CCA" or algorithm == "TRCA":
            test_acc = Trainer_Script.build_tm_test((eeg_train, label_train),
                                                    (eeg_test, label_test))
        else:
            net, criterion, optimizer = Trainer_Script.build_dl_model(devices)
            test_acc = Classifier_Trainer.train_on_batch(epochs, eeg_train, eeg_test, optimizer,
                                                         criterion, net, devices, lr_jitter=lr_jitter)

        final_test_acc_list.append(test_acc)
        print(f"Subject {subject}, Test_ACC={test_acc:.2f}")

    final_acc_list.append(final_test_acc_list)

# 3、Plot Result
Ploter.plot_save_Result(final_acc_list, model_name=algorithm, dataset=save_dataset, UD=UD, ratio=ratio,
                        win_size=str(ws * F),
                        text=True)