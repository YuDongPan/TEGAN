# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/31 10:53
import math
import numpy as np
import torch
from torch import nn
from Model import CCA_SSVEP, TRCA, EEGNet, CCNN, FBtCNN, ConvCA, SSVEPformer, DDGCNN
from Utils import Constraint, Script
from etc.global_config import config


def data_preprocess(EEGData_Train, EEGData_Test):
    '''
    Parameters
    ----------
    EEGData_Train: EEG Training Dataset (Including Data and Labels)
    EEGData_Test: EEG Testing Dataset (Including Data and Labels)

    Returns: Preprocessed EEG DataLoader
    -------
    '''
    algorithm = config['algorithm']
    ws = config["data_param"]["ws"]
    Fs = config["data_param"]["Fs"]
    Nf = config["data_param"]["Nf"]

    '''Loading Training Data'''
    EEGLabel_Train = EEGData_Train[1]
    EEGData_Train = EEGData_Train[0]

    if algorithm == "CCA" or algorithm == "TRCA":
        EEGLabel_Test = EEGData_Test[1]
        EEGData_Test = EEGData_Test[0]
        EEGData_Train = EEGData_Train.squeeze(1).numpy()
        EEGData_Test = EEGData_Test.squeeze(1).numpy()
        print("EEGData_Train.shape", EEGData_Train.shape)
        print("EEGData_Test.shape", EEGData_Test.shape)
        return EEGData_Train, EEGData_Test

    elif algorithm == "ConvCA":
        EEGData_Train = torch.swapaxes(EEGData_Train, axis0=2, axis1=3)  # (Nh, 1, Nt, Nc)
        EEGTemp_Train = Script.get_Template_Signal(EEGData_Train, Nf)  # (Nf × 1 × Nt × Nc)
        EEGTemp_Train = torch.swapaxes(EEGTemp_Train, axis0=0, axis1=1)  # (1 × Nf × Nt × Nc)
        EEGTemp_Train = EEGTemp_Train.repeat((EEGData_Train.shape[0], 1, 1, 1))  # (Nh × Nf × Nt × Nc)
        EEGTemp_Train = torch.swapaxes(EEGTemp_Train, axis0=1, axis1=3)  # (Nh × Nc × Nt × Nf)

        print("EEGData_Train.shape", EEGData_Train.shape)
        print("EEGTemp_Train.shape", EEGTemp_Train.shape)
        print("EEGLabel_Train.shape", EEGLabel_Train.shape)
        EEGData_Train = torch.utils.data.TensorDataset(EEGData_Train, EEGTemp_Train, EEGLabel_Train)

    else:
        if algorithm == "C_CNN":
            EEGData_Train = CCNN.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Train = torch.from_numpy(EEGData_Train)

        elif algorithm == "SSVEPformer":
            EEGData_Train = SSVEPformer.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Train = torch.from_numpy(EEGData_Train)
            EEGData_Train = EEGData_Train.squeeze(1)

        elif algorithm == "DDGCNN":
            EEGData_Train = torch.swapaxes(EEGData_Train, axis0=1, axis1=3)  # (Nh, 1, Nc, Nt) => (Nh, Nt, Nc, 1)

        print("EEGData_Train.shape", EEGData_Train.shape)
        print("EEGLabel_Train.shape", EEGLabel_Train.shape)
        EEGData_Train = torch.utils.data.TensorDataset(EEGData_Train, EEGLabel_Train)

    '''Loading Testing Data'''
    EEGLabel_Test = EEGData_Test[1]
    EEGData_Test = EEGData_Test[0]

    if algorithm == "ConvCA":
        EEGData_Test = torch.swapaxes(EEGData_Test, axis0=2, axis1=3)  # (Nh, 1, Nt, Nc)
        EEGTemp_Test = Script.get_Template_Signal(EEGData_Test, Nf)  # (Nf × 1 × Nt × Nc)
        EEGTemp_Test = torch.swapaxes(EEGTemp_Test, axis0=0, axis1=1)  # (1 × Nf × Nt × Nc)
        EEGTemp_Test = EEGTemp_Test.repeat((EEGData_Test.shape[0], 1, 1, 1))  # (Nh × Nf × Nt × Nc)
        EEGTemp_Test = torch.swapaxes(EEGTemp_Test, axis0=1, axis1=3)  # (Nh × Nc × Nt × Nf)

        print("EEGData_Test.shape", EEGData_Test.shape)
        print("EEGTemp_Test.shape", EEGTemp_Test.shape)
        print("EEGLabel_Test.shape", EEGLabel_Test.shape)
        EEGData_Test = torch.utils.data.TensorDataset(EEGData_Test, EEGTemp_Test, EEGLabel_Test)

    else:
        if algorithm == "C_CNN":
            EEGData_Test = CCNN.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Test = torch.from_numpy(EEGData_Test)

        elif algorithm == "SSVEPformer":
            EEGData_Test = SSVEPformer.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Test = torch.from_numpy(EEGData_Test)
            EEGData_Test = EEGData_Test.squeeze(1)

        elif algorithm == "DDGCNN":
            EEGData_Test = torch.swapaxes(EEGData_Test, axis0=1, axis1=3)  # (Nh, 1, Nc, Nt) => (Nh, Nt, Nc, 1)

        print("EEGData_Test.shape", EEGData_Test.shape)
        print("EEGLabel_Test.shape", EEGLabel_Test.shape)
        EEGData_Test = torch.utils.data.TensorDataset(EEGData_Test, EEGLabel_Test)

    # Create DataLoader for the Dataset
    bz = config[algorithm]["bz"]
    eeg_train_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Train, batch_size=bz, shuffle=True,
                                                       drop_last=True)
    eeg_test_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Test, batch_size=bz, shuffle=False,
                                                      drop_last=True)

    return eeg_train_dataloader, eeg_test_dataloader


def build_tm_test(eeg_train_dataset, eeg_test_dataset):
    algorithm = config['algorithm']
    Ds = config['data_param']['Ds']

    eeg_data_train, eeg_label_train = eeg_train_dataset[:]
    eeg_data_test, eeg_label_test = eeg_test_dataset[:]

    if algorithm == "CCA":
        if Ds == 'Direction':
            targets = [12.0, 8.57, 6.67, 5.45]

        elif Ds == 'Dial':
            targets = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                       10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
        else:
            targets = []
        CCA = CCA_SSVEP.CCA_Base(targets)
        labels, predicted_labels = CCA.cca_classify(eeg_data_test, train_data=eeg_data_train, template=True)
        test_acc = np.sum(predicted_labels == labels) / labels.shape[0]

    elif algorithm == "TRCA":
        trca = TRCA.TRCA((eeg_data_train, eeg_label_train.numpy()), (eeg_data_test, eeg_label_test.numpy()))
        trca.load_data()
        test_acc = trca.fit()

    else:
        test_acc = 0

    return test_acc


def build_dl_model(devices):
    '''
    Parameters
    ----------
    device: the device to save DL models
    Returns: the building model
    -------
    '''
    algorithm = config['algorithm']
    Ds = config['data_param']['Ds']
    Nc = config["data_param"]['Nc']
    Nf = config["data_param"]['Nf']
    Fs = config["data_param"]['Fs']
    ws = config["data_param"]['ws']
    lr = config[algorithm]['lr']
    wd = config[algorithm]['wd']
    F = config["TEGAN"]["F"]
    Nt = int(Fs * ws * F)

    if algorithm == "EEGNet":
        net = EEGNet.EEGNet(Nc, Nt, Nf)

    elif algorithm == "C_CNN":
        net = CCNN.CNN(Nc, 220, Nf)

    elif algorithm == "FBtCNN":
        net = FBtCNN.tCNN(Nc, Nt, Nf, Fs)

    elif algorithm == "ConvCA":
        net = ConvCA.convca(Nc, Nt, Nf)

    elif algorithm == "SSVEPformer":
        net = SSVEPformer.SSVEPformer(depth=2, attention_kernal_length=31, chs_num=Nc, class_num=Nf,
                                      dropout=0.5)
        net.apply(Constraint.initialize_weights)


    elif algorithm == "DDGCNN":
        bz = config[algorithm]["bz"]
        norm = config[algorithm]["norm"]
        act = config[algorithm]["act"]
        trans_class = config[algorithm]["trans_class"]
        n_filters = config[algorithm]["n_filters"]
        net = DDGCNN.DenseDDGCNN([bz, Nt, Nc], k_adj=3, num_out=n_filters, dropout=0.5, n_blocks=3, nclass=Nf,
                                 bias=False, norm=norm, act=act, trans_class=trans_class, device=devices)

    net = net.to(devices)

    criterion = nn.CrossEntropyLoss(reduction="none")

    if algorithm == "SSVEPformer":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)

    return net, criterion, optimizer