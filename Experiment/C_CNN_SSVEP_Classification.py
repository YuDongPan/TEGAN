# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/11/4 12:47
import torch
import argparse
import Utils.EEGDataset as EEGDataset
from torch import nn
from Model import C_CNN
from Model import TEGAN as TEGAN
from Train import Classifier_Trainer
from Utils import Ploter, Scripts

# 1、Define parameter of eeg
'''
                 Fs    Nc    Nh    Nf    Ns   bz     wd       
    Direction:  100   10    100    4    54    20    0.0001
         Dial:  256    8    180    12   10    30    0.0001
'''

parser = argparse.ArgumentParser()

'''Direction Dataset'''
# parser.add_argument('--dataset', type=str, default='Direction', help="4-class dataset")
# parser.add_argument('--factor', type=int, default=1, help="factor size of ssvep")
# parser.add_argument('--epochs', type=int, default=200, help="number of epochs")
# parser.add_argument('--bz', type=int, default=20, help="number of batch")
# parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
# parser.add_argument('--ws', type=float, default=0.5, help="window size of ssvep")
# parser.add_argument('--Kf', type=int, default=5, help="k-fold cross validation")
# parser.add_argument('--Nh', type=int, default=100, help="number of trial")
# parser.add_argument('--Nc', type=int, default=10, help="number of channel")
# parser.add_argument('--Fs', type=int, default=100, help="frequency of sample")
# parser.add_argument('--Nf', type=int, default=4, help="number of stimulus")
# parser.add_argument('--Ns', type=int, default=54, help="number of subjects")
# parser.add_argument('--wd', type=int, default=0.0001, help="weight decay")
# parser.add_argument('--UD', type=int, default=0, help="-1(Unsupervised),0(User-dependent),1(User-Indepedent)")
# parser.add_argument('--ratio', type=int, default=1, help="-1(Training-free),0(N-1vs1),1(R1),2(5v5),3(2v8)")

'''Dial Dataset'''
parser.add_argument('--dataset', type=str, default='Dial', help="12-class dataset")
parser.add_argument('--factor', type=int, default=2, help="factor size of ssvep")
parser.add_argument('--epochs', type=int, default=500, help="number of epochs")
parser.add_argument('--bz', type=int, default=30, help="number of batch")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--ws', type=float, default=1.0, help="window size of ssvep")
parser.add_argument('--Kf', type=int, default=5, help="k-fold cross validation")
parser.add_argument('--Nh', type=int, default=180, help="number of trial")
parser.add_argument('--Nc', type=int, default=8, help="number of channel")
parser.add_argument('--Fs', type=int, default=256, help="frequency of sample")
parser.add_argument('--Nt', type=int, default=512, help="number of sample")
parser.add_argument('--Nf', type=int, default=12, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=10, help="number of subjects")
parser.add_argument('--wd', type=int, default=0.0001, help="weight decay")
parser.add_argument('--UD', type=int, default=0, help="-1(Unsupervised),0(User-dependent),1(User-Indepedent)")
parser.add_argument('--ratio', type=int, default=3, help="-1(Training-free),0(N-1vs1),1(R1),2(5v5),3(2v8)")

opt = parser.parse_args()
devices = "cuda" if torch.cuda.is_available() else "cpu"
# 2、Start Training
final_acc_list = []
for fold_num in range(opt.Kf):
    best_valid_acc_list = []
    final_valid_acc_list = []
    print(f"Training for K_Fold {fold_num + 1}")
    for subject in range(1, opt.Ns + 1):
        # **************************************** #
        if opt.dataset == 'Direction':
            '''Intra-subject Classification Experiment'''
            # train_dataset = EEGDataset.getSSVEP4Intra(subject, train_ratio=0.2, mode="train")
            # test_dataset = EEGDataset.getSSVEP4Intra(subject, train_ratio=0.2, mode="test")
            train_dataset = EEGDataset.getSSVEP4Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="train")
            test_dataset = EEGDataset.getSSVEP4Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="test")

        elif opt.dataset == 'Dial':
            '''Intra-subject Classification Experiment'''
            # train_dataset = EEGDataset.getSSVEP12Intra(subject, train_ratio=0.2, mode="train")
            # test_dataset = EEGDataset.getSSVEP12Intra(subject, train_ratio=0.2, mode="test")
            train_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="test")
            test_dataset = EEGDataset.getSSVEP12Intra(subject, KFold=fold_num, n_splits=opt.Kf, mode="train")

        eeg_train, label_train = train_dataset[:]
        eeg_test, label_test = test_dataset[:]
        eeg_train = eeg_train[:, :, :, :int(opt.Fs * opt.ws)]
        eeg_test = eeg_test[:, :, :, :int(opt.Fs * opt.ws)]

        # -------------------------------Augmentation with TEGAN -----------------------------------------------------
        eeg_generator = TEGAN.Generator(opt.Nc, int(opt.Fs * opt.ws), opt.Nf, opt.ws, factor=opt.factor)
        '''Source Generator'''
        # eeg_generator.load_state_dict(torch.load(f'../Pretrain/{opt.dataset}/{opt.ws}S-{opt.ws * opt.factor}S/'
        #                                          f'Source/TEGAN_Gs_S{subject}.pth'))
        # '''Target Generator'''
        eeg_generator.load_state_dict(torch.load(f'../Pretrain/{opt.dataset}/{opt.ws}S-{opt.ws * opt.factor}S/'
                                                f'Target/R{opt.ratio}/F{fold_num}/TEGAN_Gt_S{subject}.pth'))

        eeg_generator.eval()
        z = eeg_train.float()

        eeg_gen_train = torch.zeros(
            (eeg_train.shape[0], 1, eeg_train.shape[2], eeg_train.shape[3] * opt.factor))
        train_idx = [h for h in range(eeg_train.shape[0])]
        eeg_gen_train[train_idx, :, :, :] = eeg_generator(z)[0].detach()

        eeg_train = eeg_gen_train.detach()
        eeg_test = eeg_generator(eeg_test.float())[0].detach()

        # -----------------------------------------------------------------------------------------------------------

        # generate complex features
        eeg_train = C_CNN.complex_spectrum_features(eeg_train.numpy(), FFT_PARAMS=[opt.Fs, opt.ws * opt.factor])
        eeg_test = C_CNN.complex_spectrum_features(eeg_test.numpy(), FFT_PARAMS=[opt.Fs, opt.ws * opt.factor])
        eeg_train = torch.from_numpy(eeg_train)
        eeg_test = torch.from_numpy(eeg_test)


        print("eeg_train.shape:", eeg_train.shape)
        print("eeg_test.shape:", eeg_test.shape)
        # -----------------------------------------------------------------------------------------------------------
        EEGData_Train = torch.utils.data.TensorDataset(eeg_train, label_train)
        EEGData_Test = torch.utils.data.TensorDataset(eeg_test, label_test)

        # Create DataLoader for the Dataset
        train_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Train, batch_size=opt.bz, shuffle=True,
                                                       drop_last=True)
        valid_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Test, batch_size=opt.bz, shuffle=False)

        # Define Network
        net = C_CNN.CNN(opt.Nc, 220, opt.Nf)
        net = net.to(devices)
        criterion = nn.CrossEntropyLoss(reduction="none")

        valid_acc = Classifier_Trainer.train_on_batch(opt.epochs, train_dataloader, valid_dataloader, opt.lr, criterion,
                                                      net, devices, wd=opt.wd, lr_jitter=False)
        final_valid_acc_list.append(valid_acc)

    final_acc_list.append(final_valid_acc_list)

# 3、Plot Result
save_dataset = opt.dataset if opt.factor <= 1 else opt.dataset + '_AUG'
Ploter.plot_save_Result(final_acc_list, model_name='C_CNN', dataset=opt.dataset, UD=opt.UD, ratio=opt.ratio,
                        win_size=str(opt.ws), text=True)