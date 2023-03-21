# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/9/8 19:46
import math
import torch
import argparse
import Utils.EEGDataset as EEGDataset
import Model.TEGAN as TEGAN
import Train.TEGAN_Trainer as GAN_Trainer
from Utils import Scripts, Constraint

# 1、Define parameters of eeg
parser = argparse.ArgumentParser()
'''Direction SSVEP Dataset'''
parser.add_argument('--dataset', type=str, default='Direction', help="Direction or BETA")
parser.add_argument('--epochs', type=int, default=200, help="number of epochs")
parser.add_argument('--factor', type=int, default=2, help="factor size of ssvep, must be times of 2")
parser.add_argument('--bz', type=int, default=64, help="number of batch")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--ws', type=float, default=0.5, help="window size of ssvep")
parser.add_argument('--Kf', type=int, default=5, help="k-fold cross validation")
parser.add_argument('--Nc', type=int, default=10, help="number of channel")
parser.add_argument('--Fs', type=int, default=100, help="frequency of sample")
parser.add_argument('--Nf', type=int, default=4, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=54, help="number of subjects")
parser.add_argument('--wd', type=int, default=0.0001, help="weight decay")
parser.add_argument('--start', type=int, default=50, help="when to use ema")
parser.add_argument('--ratio', type=int, default=3, help="-1(Training-free),0(N-1vs1),1(8vs2),2(5v5),3(2v8)")

'''Dial SSVEP Dataset'''
# parser.add_argument('--dataset', type=str, default='Dial', help="Direction or BETA")
# parser.add_argument('--epochs', type=int, default=200, help="number of epochs")
# parser.add_argument('--factor', type=int, default=2, help="factor size of ssvep, must be times of 2")
# parser.add_argument('--bz', type=int, default=64, help="number of batch")
# parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
# parser.add_argument('--ws', type=float, default=0.5, help="window size of ssvep")
# parser.add_argument('--Kf', type=int, default=5, help="k-fold cross validation")
# parser.add_argument('--Nc', type=int, default=8, help="number of channel")
# parser.add_argument('--Fs', type=int, default=256, help="frequency of sample")
# parser.add_argument('--Nf', type=int, default=12, help="number of stimulus")
# parser.add_argument('--Ns', type=int, default=10, help="number of subjects")
# parser.add_argument('--wd', type=int, default=0.0001, help="weight decay")
# parser.add_argument('--start', type=int, default=50, help="when to use ema")
# parser.add_argument('--ratio', type=int, default=3, help="-1(Training-free),0(N-1vs1),1(8vs2),2(5v5),3(2v8)")

opt = parser.parse_args()

# 2、Start Training
# 2.1、First stage training: training on source data
print("Executing first stage training...")
for subject in range(1, opt.Ns + 1):
    if opt.dataset == 'Direction':
        source_dataset = EEGDataset.getSSVEP4Inter(subject, mode="train")
    else:
        source_dataset = EEGDataset.getSSVEP12Inter(subject, mode="train")

    EEGData_Source, EEGLabel_Source = source_dataset[:]
    EEGData_Source = EEGData_Source[:, :, :, :math.ceil(opt.ws * opt.Fs) * opt.factor]

    print(f"EEGData_Source.shape:{EEGData_Source.shape}")
    print(f"EEGLabel_Source.shape:{EEGLabel_Source.shape}")

    EEGData_Source = torch.utils.data.TensorDataset(EEGData_Source, EEGLabel_Source)
    source_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Source, batch_size=opt.bz, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Ds = TEGAN.Discriminator(opt.Nc, round(opt.ws * opt.Fs), opt.Nf, opt.ws, opt.factor)
    Gs = TEGAN.Generator(opt.Nc, round(opt.ws * opt.Fs), opt.Nf, opt.ws, opt.factor)
    Ds = Ds.to(device)
    Gs = Gs.to(device)
    Scripts.cal_Parameters(Gs, 'Gs')
    Scripts.cal_Parameters(Ds, 'Ds')
    print("Gs:", Gs)
    print("Ds:", Ds)

    GAN_Trainer.train_on_batch(opt, 500, source_dataloader, Gs, Ds, device, subject=subject, source=0, lr_jitter=False)

# 2.2、First stage training: training on source data
print("Executing second stage training...")
for subject in range(1, opt.Ns + 1):
    for fold_num in range(opt.Kf):
        if opt.dataset == 'Direction':
            target_dataset = EEGDataset.getSSVEP4Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="test")
        else:
            target_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="test")

        EEGData_Target, EEGLabel_Target = target_dataset[:]
        EEGData_Target = EEGData_Target[:, :, :, :round(opt.factor * opt.ws * opt.Fs)]

        print(f"EEGData_Target.shape:{EEGData_Target.shape}")
        print(f"EEGLabel_Target.shape:{EEGLabel_Target.shape}")

        EEGData_Source = torch.utils.data.TensorDataset(EEGData_Target, EEGLabel_Target)
        target_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Target, batch_size=opt.bz, shuffle=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        Dt = TEGAN.Discriminator(opt.Nc, math.ceil(opt.ws * opt.Fs), opt.Nf, opt.ws, opt.factor, pretrain=True)
        Gt = TEGAN.Generator(opt.Nc, math.ceil(opt.ws * opt.Fs), opt.Nf, opt.ws, opt.factor)
        Gt.load_state_dict(
            torch.load(f'../Pretrain/{opt.dataset}/{opt.ws}S-{opt.ws * opt.factor}S/Source/TEGAN_Gs_S{subject}.pth'))
        Dt.load_state_dict(
            torch.load(f'../Pretrain/{opt.dataset}/{opt.ws}S-{opt.ws * opt.factor}S/Source/TEGAN_Ds_S{subject}.pth'))
        Dt = Dt.to(device)
        Gt = Gt.to(device)

        opt.lr = 0.01
        opt.bz = 20 if opt.dataset == 'Direction' else 24
        GAN_Trainer.train_on_batch(opt, 500, target_dataloader, Gt, Dt, device, subject=subject, source=1,
                                   lr_jitter=True)



