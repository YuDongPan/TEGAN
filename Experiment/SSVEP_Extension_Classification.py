# Designer:Pan YuDong
# Coder:God's hand
# Time:2021/12/20 15:42
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
Ns = config["data_param"]['Ns']

'''Parameters for DL-based methods'''
if algorithm != "ITCCA" and algorith != "TRCA":
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
        eeg_train = eeg_train[:, :, :, :math.ceil(Fs * ws)]
        eeg_test = eeg_test[:, :, :, :math.ceil(Fs * ws)]

        # *******************************Extension with TEGAN*******************************************
        eeg_train, eeg_test = Extender.SSVEP_Extension(subject, eeg_train, eeg_test, source=False, fold_num=fold_num)

        # *******************************Data preprocessing*************************************************
        eeg_train, eeg_test = Trainer_Script.data_preprocess((eeg_train, label_train),
                                                             (eeg_test, label_test))

        # *******************************Classifier Training and Testing************************************
        if algorithm == "ITCCA" or algorithm == "TRCA":
            test_acc = Trainer_Script.build_tm_test((eeg_train, label_train),
                                                    (eeg_test, label_test))
        else:
            net, criterion, optimizer = Trainer_Script.build_dl_model(devices)
            test_acc = Classifier_Trainer.train_on_batch(epochs, eeg_train, eeg_test, optimizer,
                                                         criterion, net, devices, lr_jitter=lr_jitter)

        final_test_acc_list.append(test_acc)
        # exit()

    final_acc_list.append(final_test_acc_list)

# 3、Plot Result
save_dataset = Ds if F <= 1 else Ds + '_EXT'
Ploter.plot_save_Result(final_acc_list, model_name=algorithm, dataset=save_dataset, UD=UD, ratio=ratio,
                        win_size=str(ws),
                        text=True)