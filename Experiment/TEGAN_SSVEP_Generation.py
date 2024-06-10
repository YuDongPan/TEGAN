# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/9/8 19:46
import sys

sys.path.append('../')
import math
import torch
import argparse
import Utils.EEGDataset as EEGDataset
import Model.TEGAN as TEGAN
import Train.TEGAN_Trainer as GAN_Trainer
from Utils import Script, Constraint
from etc.global_config import config

# 1、Define parameters of training of TEGAN
'''                  Fs     Nc    Nf     Ns    start f1_epochs f2_epochs  f1_bz   f2_bz    f1_lr   f2_lr   f1_wd  f2_wd
        Direction:  100    10     4      54     50     200        500      64      30      1e-3     0.01    1e-4   3e-4
             Dial:  256    8      12     10     50     200        500      64      20      1e-3     0.01    1e-4   3e-4
'''

UD = config["train_param"]["UD"]
ratio = config["train_param"]["ratio"]
if ratio == 1 or ratio == 3:
    Kf = 5
elif ratio == 2:
    Kf = 2
else:
    Kf = 1
mode = "test" if ratio == 3 else "train"

Ds = config["data_param"]["Ds"]
ws = config["data_param"]["ws"]
Fs = config["data_param"]["Fs"]
Nc = config["data_param"]["Nc"]
Nf = config["data_param"]["Nf"]
Ns = config["data_param"]["Ns"]

F = config["TEGAN"]["F"]
start = config["TEGAN"]["start"]
f1_epochs = config["TEGAN"]["f1_epochs"]
f2_epochs = config["TEGAN"]["f2_epochs"]
f1_bz = config["TEGAN"]["f1_bz"]
f2_bz = config["TEGAN"]["f2_bz"]
f1_lr = config["TEGAN"]["f1_lr"]
f2_lr = config["TEGAN"]["f2_lr"]
f1_wd = config["TEGAN"]["f1_wd"]
f2_wd = config["TEGAN"]["f2_wd"]

trainer_dict = {"Ds": Ds, "ws": ws, "Fs": Fs, "Nc": Nc, "Nf": Nf, "Ns": Ns,
                "F": F, "epochs": f1_epochs, "bz": f1_bz, "lr": f1_lr, "wd": f1_wd}


device = "cuda" if torch.cuda.is_available() else "cpu"

# 2、First stage training
for subject in range(1, Ns + 1):
    source_dataset = EEGDataset.getSSVEP4Inter(subject, mode="train")
    EEGData_Source, EEGLabel_Source = source_dataset[:]
    EEGData_Source = EEGData_Source[:, :, :, :math.ceil(ws * Fs) * F]

    print(f"EEGData_Source.shape:{EEGData_Source.shape}, EEGLabel_Source.shape:{EEGLabel_Source.shape}")

    EEGData_Source = torch.utils.data.TensorDataset(EEGData_Source, EEGLabel_Source)
    source_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Source, batch_size=f1_bz, shuffle=True)

    Ds = TEGAN.Discriminator(Nc, round(ws * Fs), Nf, ws, F)
    Gs = TEGAN.Generator(Nc, round(ws * Fs), Nf, ws, F)
    Ds = Ds.to(device)
    Gs = Gs.to(device)
    Script.cal_Parameters(Gs, 'Gs')
    Script.cal_Parameters(Ds, 'Ds')

    GAN_Trainer.train_on_batch(trainer_dict, source_dataloader, Gs, Ds, device, subject=subject, source=0,
                               lr_jitter=False)

# 3、Second stage training
for fold_num in range(Kf):
    for subject in range(1, Ns + 1):
        target_dataset = EEGDataset.getSSVEP4Intra(subject, KFold=fold_num, n_splits=Kf, mode=mode)

        EEGData_Target, EEGLabel_Target = target_dataset[:]
        EEGData_Target = EEGData_Target[:, :, :, :math.ceil(ws * Fs) * F]

        print(f"EEGData_Target.shape:{EEGData_Target.shape}, EEGLabel_Target.shape:{EEGLabel_Target.shape}")

        EEGData_Target = torch.utils.data.TensorDataset(EEGData_Target, EEGLabel_Target)
        target_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Target, batch_size=f2_bz, shuffle=True)

        Dt = TEGAN.Discriminator(Nc, round(ws * Fs), Nf, ws, F, pretrain=True)
        Gt = TEGAN.Generator(Nc, round(ws * Fs), Nf, ws, F)

        Gt.load_state_dict(torch.load(f'/SD1/panyd/EEG/model/GAN/TEGAN/{Ds}/{ws}S-{ws * F}S/'
                                      f'Source/TEGAN_Gs_S{subject}.pth'))
        Dt.load_state_dict(torch.load(f'/SD1/panyd/EEG/model/GAN/TEGAN/{Ds}/{ws}S-{ws * F}S/'
                                      f'Source/TEGAN_Ds_S{subject}.pth'))

        Dt = Dt.to(device)
        Gt = Gt.to(device)
        Script.cal_Parameters(Gt, 'Gt')
        Script.cal_Parameters(Dt, 'Dt')


        trainer_dict["epochs"] = f2_epochs
        trainer_dict["bz"] = f2_bz
        trainer_dict["lr"] = f2_lr
        trainer_dict["wd"] = f2_wd
        GAN_Trainer.train_on_batch(trainer_dict, target_dataloader, Gt, Dt, device, subject=subject, source=1,
                                   fold_num=fold_num, lr_jitter=True)

