# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/2/28 15:45
import math
import torch
from Model import TEGAN
from etc.global_config import config


def SSVEP_Extension(subject, eeg_train, eeg_test, source=True, fold_num=0):
    print("====================Before SSVEP Extension by TEGAN==========================")

    print("eeg_train.shape:", eeg_train.shape)
    print("eeg_test.shape:", eeg_test.shape)

    ratio = config["train_param"]["ratio"]

    Ds = config["data_param"]["Ds"]
    Nc = config["data_param"]["Nc"]
    ws = config["data_param"]["ws"]
    Fs = config["data_param"]["Fs"]
    Nf = config["data_param"]["Nf"]
    F = config["TEGAN"]["F"]
    eeg_generator = TEGAN.Generator(Nc, math.ceil(Fs * ws), Nf, ws, factor=F)
    if source:
        eeg_generator.load_state_dict(torch.load(f'../Pretrain/{Ds}/{ws}S-{ws * F}S/Source/'
                                                 f'TEGAN_Gs_S{subject}.pth'))
    else:
        eeg_generator.load_state_dict(torch.load(f'../Pretrain/{Ds}/{ws}S-{ws * F}S/Target/'
                                                 f'R{ratio}/F{fold_num}/TEGAN_Gt_S{subject}.pth'))
    eeg_generator.eval()
    z = eeg_train.float()

    eeg_gen_train = eeg_generator(z)[0].detach()
    eeg_train = eeg_gen_train.detach()
    eeg_test = eeg_generator(eeg_test.float())[0].detach()

    print("======================After SSVEP Extension by TEGAN============================")

    print("eeg_train.shape:", eeg_train.shape)
    print("eeg_test.shape:", eeg_test.shape)

    return eeg_train, eeg_test
