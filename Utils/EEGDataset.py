# Designer:Pan YuDong
# Coder:God's hand
# Time:2021/10/6 22:47
import numpy as np
from torch.utils.data import Dataset
import torch
import scipy.io
from scipy import signal
from Utils import Script


# Integrate Dataset Class
class getSSVEP4Intra(Dataset):
    def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train"):
        super(getSSVEP4Intra, self).__init__()
        self.train_ratio = train_ratio
        self.Nh = 100
        self.Nc = 10
        self.Nt = 400
        self.Nf = 4
        self.Fs = 100
        self.subject = subject
        self.eeg_data, self.label_data, self.rs_data = self.load_Data()
        self.num_trial = self.Nh // self.Nf
        self.train_idx = []
        self.test_idx = []
        if KFold is not None:
            fold_trial = self.num_trial // n_splits
            self.valid_trial_idx = [i for i in range(KFold * fold_trial, (KFold + 1) * fold_trial)]

        for i in range(0, self.Nh, self.Nh // self.Nf):
            for j in range(self.Nh // self.Nf):
                if n_splits == 2 and j == self.num_trial - 1:
                    continue
                if KFold is not None:
                    if j not in self.valid_trial_idx:
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)
                else:
                    if j < round(self.num_trial * train_ratio):
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)

        self.eeg_data_train = self.eeg_data[self.train_idx]
        self.label_data_train = self.label_data[self.train_idx]
        self.eeg_data_test = self.eeg_data[self.test_idx]
        self.label_data_test = self.label_data[self.test_idx]

        if mode == 'train':
            self.eeg_data = self.eeg_data_train
            self.label_data = self.label_data_train
        elif mode == 'test':
            self.eeg_data = self.eeg_data_test
            self.label_data = self.label_data_test

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # preprocess the eeg data
    def filter_data(self, X):
        low, high = 4, 40
        b, a = signal.butter(4, [2 * low / self.Fs, 2 * high / self.Fs], 'bandpass')
        X = signal.filtfilt(b, a, X, axis=-1)
        return X

    # get the single subject data
    def load_Data(self):
        subjectfile = scipy.io.loadmat(f'../data/Direction/SE1/Offline/S{self.subject}.mat')
        eeg_data = subjectfile['eeg_data']  # (100, 400, 10)
        eeg_data = np.swapaxes(eeg_data, 1, 2)  # (100, 400, 10) -> (100, 10, 400)
        eeg_data = self.filter_data(eeg_data)
        eeg_data = np.expand_dims(eeg_data, axis=1)  # (100, 10, 400) -> (100, 1, 10, 400)
        rs_data = subjectfile['rs_data']  # (6000, 10)
        rs_data = np.swapaxes(rs_data, 0, 1)  # (6000, 10) -> (10, 6000)
        label_data = subjectfile['eeg_label']  # (100, 1)
        eeg_data = torch.from_numpy(eeg_data.copy())
        label_data = torch.from_numpy(label_data)
        print(eeg_data.shape)
        print(label_data.shape)
        return eeg_data, label_data, rs_data


class getSSVEP4Inter(Dataset):
    def __init__(self, subject=1, mode="train"):
        self.Nh = 100
        self.Nc = 10
        self.Nt = 400
        self.Nf = 4
        self.Ns = 54
        self.Fs = 100
        self.eeg_data, self.label_data = self.load_Data()
        if mode == 'train':
            self.eeg_data = torch.cat((self.eeg_data[0:(subject - 1) * self.Nh], self.eeg_data[subject * self.Nh:]),
                                      dim=0)
            self.label_data = torch.cat(
                (self.label_data[0:(subject - 1) * self.Nh:, :], self.label_data[subject * self.Nh:, :]), dim=0)

        if mode == 'test':
            self.eeg_data = self.eeg_data[(subject - 1) * self.Nh:subject * self.Nh]
            self.label_data = self.label_data[(subject - 1) * self.Nh:subject * self.Nh]

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # preprocess the eeg data
    def filter_data(self, X):
        low, high = 4, 40
        b, a = signal.butter(4, [2 * low / self.Fs, 2 * high / self.Fs], 'bandpass')
        X = signal.filtfilt(b, a, X, axis=-1)
        return X

    # get the all subject data
    def load_Data(self):
        all_sub_eeg = np.zeros((self.Nh * self.Ns, 1, self.Nc, self.Nt))
        all_sub_label = np.zeros((self.Nh * self.Ns, 1))
        for sub in range(self.Ns):
            subjectfile = scipy.io.loadmat(f'../data/Direction/SE1/Offline/S{sub + 1}.mat')
            eeg_data = subjectfile['eeg_data']  # (100, 400, 10)
            eeg_data = np.swapaxes(eeg_data, 1, 2)  # (100, 400, 10) -> (100, 10, 400)
            eeg_data = self.filter_data(eeg_data)
            eeg_data = np.expand_dims(eeg_data, axis=1)  # (100, 10, 400) -> (100, 1, 10, 400)
            label_data = subjectfile['eeg_label']  # (100, 1)
            all_sub_eeg[sub * self.Nh:(sub + 1) * self.Nh, :, :, :] = eeg_data
            all_sub_label[sub * self.Nh:(sub + 1) * self.Nh, :] = label_data
        all_sub_eeg = torch.from_numpy(all_sub_eeg)
        all_sub_label = torch.from_numpy(all_sub_label)
        print(all_sub_eeg.shape)
        print(all_sub_label.shape)
        return all_sub_eeg, all_sub_label


class getSSVEP12Intra(Dataset):
    def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train"):
        super(getSSVEP12Intra, self).__init__()
        self.train_ratio = train_ratio
        self.Nh = 180
        self.Nc = 8
        self.Nt = 1024
        self.Nf = 12
        self.Fs = 256
        self.subject = subject
        self.eeg_data, self.label_data = self.load_Data()
        self.num_trial = self.Nh // self.Nf
        self.train_idx = []
        self.test_idx = []
        if KFold is not None:
            fold_trial = self.num_trial // n_splits
            self.valid_trial_idx = [i for i in range(KFold * fold_trial, (KFold + 1) * fold_trial)]

        for i in range(0, self.Nh, self.Nh // self.Nf):
            for j in range(self.Nh // self.Nf):
                if n_splits == 2 and j == self.num_trial - 1:
                    continue
                if KFold is not None:
                    if j not in self.valid_trial_idx:
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)
                else:
                    if j < int(self.num_trial * train_ratio):
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)

        self.eeg_data_train = self.eeg_data[self.train_idx]
        self.label_data_train = self.label_data[self.train_idx]
        self.eeg_data_test = self.eeg_data[self.test_idx]
        self.label_data_test = self.label_data[self.test_idx]

        if mode == 'train':
            self.eeg_data = self.eeg_data_train
            self.label_data = self.label_data_train
        elif mode == 'test':
            self.eeg_data = self.eeg_data_test
            self.label_data = self.label_data_test

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # get the single subject data
    def load_Data(self):
        subjectfile = scipy.io.loadmat(f'../data/Dial/S{self.subject}.mat')
        samples = subjectfile['eeg']  # (12, 8, 1024, 15)
        eeg_data = samples[0, :, :, :]  # (8, 1024, 15)
        for i in range(1, 12):
            eeg_data = np.concatenate([eeg_data, samples[i, :, :, :]], axis=2)
        eeg_data = eeg_data.transpose([2, 0, 1])  # (8, 1114, 180) -> (180, 8, 1024)
        eeg_data = np.expand_dims(eeg_data, axis=1)  # (180, 8, 1024) -> (180, 1, 8, 1114)
        eeg_data = torch.from_numpy(eeg_data)
        label_data = np.zeros((180, 1))
        for i in range(12):
            label_data[i * 15:(i + 1) * 15] = i
        label_data = torch.from_numpy(label_data)
        print(eeg_data.shape)
        print(label_data.shape)
        return eeg_data, label_data


class getSSVEP12Inter(Dataset):
    def __init__(self, subject=1, mode="train"):
        self.Nh = 180
        self.Nc = 8
        self.Nt = 1024
        self.Nf = 12
        self.Ns = 10
        self.Fs = 256
        self.eeg_data, self.label_data = self.load_Data()
        if mode == 'train':
            self.eeg_data = torch.cat((self.eeg_data[0:(subject - 1) * self.Nh], self.eeg_data[subject * self.Nh:]),
                                      dim=0)
            self.label_data = torch.cat(
                (self.label_data[0:(subject - 1) * self.Nh:, :], self.label_data[subject * self.Nh:, :]), dim=0)

        if mode == 'test':
            self.eeg_data = self.eeg_data[(subject - 1) * self.Nh:subject * self.Nh]
            self.label_data = self.label_data[(subject - 1) * self.Nh:subject * self.Nh]

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # get the all subject data
    def load_Data(self):
        all_sub_eeg = np.zeros((self.Nh * self.Ns, 1, self.Nc, self.Nt))
        all_sub_label = np.zeros((self.Nh * self.Ns, 1))
        for sub in range(self.Ns):
            subjectfile = scipy.io.loadmat(f'../data/Dial/S{sub + 1}.mat')
            samples = subjectfile['eeg']  # (12, 8, 1024, 15)
            samples = samples[:, :, :, :]  # (12, 8, 1024, 15)
            eeg_data = samples[0, :, :, :]  # (8, 1024, 15)
            for i in range(1, 12):
                eeg_data = np.concatenate([eeg_data, samples[i, :, :, :]], axis=2)
            eeg_data = eeg_data.transpose([2, 0, 1])  # (8, 1024, 180) -> (180, 8, 1024)
            eeg_data = np.expand_dims(eeg_data, axis=1)  # (180, 8, 1024) -> (180, 1, 8, 1024)
            eeg_data = self.filter_data(eeg_data)
            label_data = np.zeros((180, 1))
            for i in range(12):
                label_data[i * 15:(i + 1) * 15] = i
            all_sub_eeg[sub * self.Nh:(sub + 1) * self.Nh, :, :, :] = eeg_data
            all_sub_label[sub * self.Nh:(sub + 1) * self.Nh, :] = label_data
        all_sub_eeg = torch.from_numpy(all_sub_eeg)
        all_sub_label = torch.from_numpy(all_sub_label)
        print(all_sub_eeg.shape)
        print(all_sub_label.shape)
        return all_sub_eeg, all_sub_label

        label_data = torch.from_numpy(label_data)
        print(f"Load S{subject} data, eeg_data.shape:{eeg_data.shape}, label_data.shape:{label_data.shape}")
        return eeg_data, label_data

    def load_Data(self):
        eeg_data = torch.zeros((self.Ns * self.Nh, 1, self.Nc, self.delay + self.Nt))
        label_data = torch.zeros((self.Ns * self.Nh, 1))
        for i in range(self.Ns):
            eeg_sub, label_sub = self.load_Sub(i + 1)
            eeg_data[i * self.Nh:(i + 1) * self.Nh, :, :, :] = eeg_sub
            label_data[i * self.Nh:(i + 1) * self.Nh, :] = label_sub
        print(f"Load data success, all_eeg_data.shape:{eeg_data.shape}, all_label_data.shape:{label_data.shape}")
        return eeg_data, label_data
