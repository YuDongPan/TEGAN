# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/1 12:57
import numpy as np
import scipy
from scipy import signal
import math
import einops

# Parameters of eeg data
num_chans, num_sampls, num_blocks = 8, 1024, 15

# Number of subjects
subject_num = 10

# Data length for target identification [s]
len_gaze_s = 1.0

# Visual latency being considered in the analysis [s]
len_delay_s = 0

# The number of sub-bands in filter bank analysis
num_fbs = 1

# 1 -> The ensemble TRCA-based method, 0 -> The TRCA-based method
is_ensemble = 0

# 100*(1-alpha_ci): confidence intervals
alpha_ci = 0.05

# Fixed parameter (Modify according to the experimental setting)
# Sampling rate [Hz]
fs = 256

# Duration for gaze shifting [s]
len_shift_s = 0.5

# List of stimulus frequencies
# list_freqs = [i for i in range(pattern * pattern)]

# The number of stimuli
num_targs = 12

# Preparing useful variables (DONT'T need to modify)
# Data length [samples]
len_gaze_smpl = round(len_gaze_s * fs)

# Visual latency [samples]
len_delay_smpl = round(len_delay_s * fs)

# Selection time [s]
len_sel_s = len_gaze_s + len_shift_s

# Confidence interval
ci = 100 * (1 - alpha_ci)


def get_eegnet_raw():
    data_eeg = np.zeros((subject_num, num_targs, num_blocks, num_chans, num_sampls))
    label_eeg = np.zeros((subject_num, num_targs, num_blocks))
    for sub in range(subject_num):
        root = f"../data/Dial/S{sub + 1}.mat"
        data = scipy.io.loadmat(root)
        # 12 * 8 * 1024 * 15 traindata
        data_eeg[sub, :, :, :, :] = np.transpose(data['eeg'], (0, 3, 1, 2))
        label_eeg[sub, :, :] = np.tile(np.arange(num_targs).reshape(-1, 1), (1, num_blocks))
        # 10 * 12 * 15 label
    return data_eeg, label_eeg


def filterbank(eeg):
    result = np.zeros((10, 12, 15, num_fbs, 8, eeg.shape[-1]))

    '''Ours:Filter Bank'''
    # nyq = fs / 2
    # low = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    # high = 80
    #
    # for i in range(num_fbs):
    #     b, a = signal.butter(4, [low[i] / nyq, high / nyq], btype='band')
    #     data = signal.filtfilt(b, a, eeg, padlen=3 * (max(len(b), len(a)) - 1)).copy()
    #     result[:, :, :, i, :, :] = data

    '''Masaki:Filter Bank'''
    nyq = fs / 2
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    gpass = 3
    gstop = 40
    Rp = 0.5
    for i in range(num_fbs):
        Wp = [passband[i] / nyq, 50 / nyq]
        Ws = [stopband[i] / nyq, 60 / nyq]
        [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
        [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')
        data = signal.filtfilt(B, A, eeg, padlen=3 * (max(len(B), len(A)) - 1)).copy()
        result[:, :, :, i, :, :] = data

    return result


def data_process(data, label):
    data = data[:, :, :, :, len_delay_smpl: len_delay_smpl + len_gaze_smpl]
    data = filterbank(data)
    data = np.transpose(data, (0, 1, 3, 4, 5, 2))  # (10, 12, num_fbs, 8, 256, 15)

    return data, label


def loader_ui(data, label, testsubject):
    trainsubject = [i for i in range(subject_num)]
    trainsubject.remove(testsubject)  # Exclude the block used for testing

    train_x = data[trainsubject, :, :, :, :, :]  # (9, 12, num_fbs, 8, duration, 15)
    train_x = einops.rearrange(train_x, 'a b c d e f -> b c d e (a f)')  # (12, num_fbs, 8, duration, 9 * 15)

    test_x = data[testsubject, :, :, :, :, :]  # (12, num_fbs, 8, duration, 1 *15)

    test_y = label[testsubject, :, :]  # (12, 15)

    return train_x, test_x, test_y

def loader_ud(data, label, testsubject, fold_i, k_fold, reverse=False):

    data = data[testsubject]
    label = label[testsubject]

    if k_fold == 2 and (num_blocks % k_fold != 0):
        data = np.delete(data, num_blocks - 1, -1)
        label = np.delete(label, num_blocks - 1, -1)

    fold_trial = num_blocks // k_fold
    idx = range(fold_i * fold_trial, (fold_i + 1) * fold_trial)

    if not reverse:
        train_x = np.delete(data, idx, -1)   # (12, num_fbs, 8, duration, 15 - fold_trial)
        test_x = data[:, :, :, :, idx]  # (12, num_fbs, 8, duration, fold_trial)
        test_y = label[:, idx]  # (12, fold_trial)
    else:
        train_x = data[:, :, :, :, idx]  # (12, num_fbs, 8, duration, fold_trial)
        test_x = np.delete(data[:, :, :, :, :], idx, -1)   # (12, num_fbs, 8, duration, 15 - fold_trial)
        test_y = np.delete(label[:, :], idx, -1)  # (12, 15 - fold_trial)

    # print("train_x.shape:", train_x.shape)
    # print("test_x.shape:", test_x.shape)
    # print("test_y.shape:", test_y.shape)

    return train_x, test_x, test_y

def train_trca(eeg):
    [num_targs, _, num_chans, num_smpls, num_blocks] = eeg.shape
    trains = np.zeros((num_targs, num_fbs, num_chans, num_smpls))
    W = np.zeros((num_fbs, num_targs, num_chans))
    for targ_i in range(num_targs):
        eeg_tmp = eeg[targ_i, :, :, :, :]
        for fb_i in range(num_fbs):
            traindata = eeg_tmp[fb_i, :, :, :]
            trains[targ_i, fb_i, :, :] = np.mean(traindata, 2)
            w_tmp = trca(traindata)
            W[fb_i, targ_i, :] = np.real(w_tmp[:, 0])

    return trains, W


def trca(eeg):
    [num_chans, num_smpls, num_trials] = eeg.shape
    S = np.zeros((num_chans, num_chans))
    for trial_i in range(num_trials - 1):
        x1 = eeg[:, :, trial_i]
        x1 = x1 - np.expand_dims(np.mean(x1, 1), 1).repeat(x1.shape[1], 1)
        for trial_j in range(trial_i + 1, num_trials):
            x2 = eeg[:, :, trial_j]
            x2 = x2 - np.expand_dims(np.mean(x2, 1), 1).repeat(x2.shape[1], 1)
            S = S + np.matmul(x1, x2.T) + np.matmul(x2, x1.T)

    UX = eeg.reshape(num_chans, num_smpls * num_trials)
    UX = UX - np.expand_dims(np.mean(UX, 1), 1).repeat(UX.shape[1], 1)
    Q = np.matmul(UX, UX.T)
    W, V = scipy.sparse.linalg.eigs(S, 6, Q)
    return V


def test_trca(eeg, trains, W, is_ensemble):
    num_trials = eeg.shape[4]
    fb_coefs = [math.pow(i, -1.25) + 0.25 for i in range(1, num_fbs + 1)]
    fb_coefs = np.array(fb_coefs)
    results = np.zeros((num_targs, num_trials))

    for targ_i in range(num_targs):
        test_tmp = eeg[targ_i, :, :, :, :]
        r = np.zeros((num_fbs, num_targs, num_trials))
        rho = np.zeros((num_targs, num_trials))

        for fb_i in range(num_fbs):
            testdata = test_tmp[fb_i, :, :, :]

            for class_i in range(num_targs):
                traindata = trains[class_i, fb_i, :, :]
                if not is_ensemble:
                    w = W[fb_i, class_i, :]
                else:
                    w = W[fb_i, :, :].T
                for trial_i in range(num_trials):
                    testdata_w = np.matmul(testdata[:, :, trial_i].T, w)
                    traindata_w = np.matmul(traindata[:, :].T, w)
                    r_tmp = np.corrcoef(testdata_w.flatten(), traindata_w.flatten())
                    r[fb_i, class_i, trial_i] = r_tmp[0, 1]

        rho = np.einsum('j, jkl -> kl', fb_coefs, r)  # (num_targs, num_trials)
        tau = np.argmax(rho, axis=0)
        results[targ_i, :] = tau
    return results


def itr(n, p, t):
    if p < 0 or 1 < p:
        print('Accuracy need to be between 0 and 1.')
        exit()
    elif p < 1 / n:
        print('The ITR might be incorrect because the accuracy < chance level.')
        itr = 0
    elif p == 1:
        itr = math.log2(n) * 60 / t
    else:
        itr = (math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))) * 60 / t
    return itr


if __name__ == "__main__":
    data, label = get_eegnet_raw()
    data, label = data_process(data, label)
    print("eeg_data.shape:", data.shape)
    print("label_data.shape:", label.shape)

    accs = np.zeros((subject_num))
    itrs = np.zeros((subject_num))

    k_fold = 1
    for test_id in range(0, subject_num):
        for fold_i in range(0, k_fold):
            # Inter-subject experiment
            traindata, testdata, testlabel = loader_ui(data, label, test_id)

            # Intra-subject experiment
            # traindata, testdata, testlabel = loader_ud(data, label, test_id, fold_i, k_fold, reverse=True)

            # Training stage
            # print(traindata.shape)
            trains, W = train_trca(traindata)

            # Test stage
            # print(testdata.shape)
            estimated = test_trca(testdata, trains, W, is_ensemble)

            # Evaluation
            is_correct = (estimated == testlabel)
            is_correct = np.array(is_correct).astype(int)

            # print("is_correct:", is_correct)

            accs[test_id] += np.mean(is_correct) * 100
            itrs[test_id] += itr(num_targs, np.mean(is_correct), len_sel_s)

        accs[test_id] /= k_fold
        itrs[test_id] /= k_fold
        print('Subject ', test_id + 1, ': ACC = ', accs[test_id], ', ITR = ', itrs[test_id], ' bpm')

    print('ACC:', accs)
    print('ITR:', accs)
    accs_men = np.mean(accs)
    itrs_mean = np.mean(itrs)
    print('Mean ACC = ', accs_men, ', Mean ITR = ', itrs_mean, ' bpm')


