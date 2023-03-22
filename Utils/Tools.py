# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/9/8 21:29
import os
import shutil
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Utils.Normalization as Norm
from Model import TEGAN as TEGAN
from matplotlib.collections import LineCollection
from scipy import signal
from Utils import Scripts

def norm_single_eeg(eeg_data, method=1):
    '''
    :param eeg_data: input eeg data
    :param method: 0-Z_score,1-MaxAbs, 2-MaxMin
    :return:
    '''
    for i in range(eeg_data.shape[0]):
        if method == 0:
            eeg_data[i] = Norm.Z_Score(eeg_data[i])
        elif method == 1:
            eeg_data[i] = Norm.MaxAbs(eeg_data[i])
        elif method == 2:
            eeg_data[i] = Norm.MaxMin(eeg_data[i])

    return eeg_data

def norm_all_eeg(eeg_data):
    for i in range(eeg_data.shape[0]):
        if len(eeg_data.shape) < 4:
            eeg_data[i] = norm_single_eeg(eeg_data[i])
        else:
            eeg_data[i, 0] = norm_single_eeg(eeg_data[i, 0])
    return eeg_data

def CLE_DIR_CONENT(DirPath):
    filelist = os.listdir(DirPath)
    for f in filelist:
        filepath = os.path.join(DirPath, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)
    print("File Clearing Success~~~")


def plot_TimeEEG(subject, eeg_data, eeg_label, epoch, model_name, dataset, real_or_fake='fake'):
    Nc = eeg_data.shape[0]
    Nt = eeg_data.shape[1]
    Fs = 100 if dataset == 'Direction' else 256
    data = norm_single_eeg(eeg_data.numpy())
    label = eeg_label.numpy()
    n_samples, n_rows = Nt, Nc
    time_len = round(Nt / Fs)
    t = np.arange(0, time_len, 1 / Fs)
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot the EEG
    ticklocs = []
    ax.set_xlim(0, time_len)
    ax.set_xticks(np.arange(0, time_len * 1.0 + 0.25, time_len / 4))

    dmin = -1
    dmax = 1
    dr = (dmax - dmin) * 0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (n_rows - 1) * dr + dmax + 0.2
    ax.set_ylim(y0, y1)

    segs = []
    for i in range(n_rows):
        segs.append(np.column_stack((t, data[i, :] + i * dr)))
        ticklocs.append(i * dr)

    lines = LineCollection(segs)
    lines.set_array(np.arange(1, Nc + 1))
    ax.add_collection(lines)
    axcb = fig.colorbar(lines)
    axcb.set_label('Channel Number')

    # Set the yticks to use axes coordinates on the y axis
    ax.set_yticks(ticklocs)

    # for 4-class dataset
    if dataset == 'Direction':
        ax.set_yticklabels(['P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10'])
    # for 12-class dataset
    else:
        ax.set_yticklabels(['PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2'])

    ax.set_xlabel('Time (s)')
    title = 'SSVEP for subject{} on class {}'.format(subject, label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('../Generation/{}/{}/{}_eeg_{}_t.png'.format(model_name, dataset, real_or_fake, epoch))
    plt.clf()
    plt.close('all')


def plot_TimeFreqEEG2CH(subject, eeg_data, eeg_label, epoch, model_name, dataset, real_or_fake='fake'):
    Nt = eeg_data.shape[1]
    Fs = 100 if dataset == 'Direction' else 256
    time_len = round(Nt // Fs)
    Ts = 1.0 / Fs
    # obtain original signal
    eeg_data = norm_single_eeg(eeg_data.numpy())
    ch_idx1 = -3 if dataset == 'Direction' else -2
    ch_idx2 = -2 if dataset == 'Direction' else -1
    y1 = eeg_data[ch_idx1]
    y2 = eeg_data[ch_idx2]
    # filter original signal,filter the freq <= 2Hz
    # Wn = 2 * MaxFreq / Fs
    b, a = signal.butter(N=4, Wn=2 * 2 / Fs, btype='highpass')
    y1 = signal.filtfilt(b, a, y1)
    y2 = signal.filtfilt(b, a, y2)
    # construct time sequence
    t = np.arange(0, time_len, Ts)
    # 0~1 Normalization
    N = Nt

    max_freq = Fs * time_len
    n = np.arange(Nt)
    # frequency resolution
    FR = Fs / N
    frq = n * Fs / N
    frq = frq[range(int(max_freq // 2))]
    Y1 = np.fft.fft(y1)
    Y2 = np.fft.fft(y2)
    Y1 = Y1[range(int(max_freq // 2))] / max_freq * 4
    Y2 = Y2[range(int(max_freq // 2))] / max_freq * 4
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    channel = ['Oz', 'O2']
    for i in range(2):
        y = y1 if i == 0 else y2
        Y = Y1 if i == 0 else Y2
        ax[0][i].plot(t, y)
        ax[0][i].set_xlabel('Time(s)')
        ax[0][i].set_ylabel('Amplitude(V)')
        energy_spectrum = np.sqrt(np.square(Y.real) + np.square(Y.imag))
        max_index = np.argmax(energy_spectrum)
        ax[1][i].plot(frq, abs(Y), 'b')
        ax[1][i].text(max_index * FR, energy_spectrum[max_index], str(max_index * FR) + 'Hz', color='r')
        ax[1][i].set_ylim(0, energy_spectrum[max_index] + 0.1)
        title = 'SSVEP for subject{} on class {} with {}'.format(subject, eeg_label, channel[i])
        ax[1][i].set_title(title)
    plt.tight_layout()
    plt.savefig('../Generation/{}/{}/{}_eeg_{}_f.png'.format(model_name, dataset, real_or_fake, epoch))
    plt.clf()
    plt.close('all')

def plot_EEG_CMP(opt, subject, class_num, source_eeg, target_eeg, source=True):
    eeg_generator = TEGAN.Generator(opt.Nc, int(opt.Fs * opt.ws), opt.Nf, opt.ws, factor=opt.factor)
    gen_type = 'Gs' if source else 'Gt'
    eeg_generator.load_state_dict(torch.load(f'../Pretrain/{opt.dataset}/{opt.ws}S-{opt.ws*opt.factor}S/'
                                             f'TEGAN_{gen_type}_S{subject}.pth'))

    if opt.dataset == 'Direction':
        targets = [12.0, 8.57, 6.67, 5.45]

    elif opt.dataset == 'Dial':
        targets = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                   10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

    eeg_generator.eval()
    real_eeg = target_eeg.numpy()
    fake_eeg = eeg_generator(source_eeg.float())[0].detach().numpy()

    real_eeg = Scripts.filter_Data(real_eeg, opt.Fs, low=opt.low, high=opt.high)
    fake_eeg = Scripts.filter_Data(fake_eeg, opt.Fs, low=opt.low, high=opt.high)

    real_eeg = norm_single_eeg(real_eeg[0, 0], method=1)[-2]  # -2 means Oz channel
    fake_eeg = norm_single_eeg(fake_eeg[0, 0], method=1)[-2]

    # construct time sequence
    t = np.arange(0, round(opt.factor * opt.ws), 1.0 / opt.Fs)

    # 0~1 Normalization
    N = round(opt.factor * opt.ws * opt.Fs)

    max_freq = N
    n = np.arange(N)
    # FR:frequency resolution
    FR = opt.Fs / N
    print("FR:", FR)
    frq = n * opt.Fs / N
    frq = frq[range(int(max_freq // 2))]
    R_Y = np.fft.fft(real_eeg)
    F_Y = np.fft.fft(fake_eeg)
    R_Y = R_Y[range(int(max_freq // 2))] / max_freq * 4
    F_Y = F_Y[range(int(max_freq // 2))] / max_freq * 4

    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), dpi=160)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}

    # Always place grid at bottom
    ax[0].grid(True)
    ax[0].plot(t, real_eeg, color='blue')
    ax[0].plot(t, fake_eeg, color='orange')
    ax[0].set_xlabel('Time (s)', fontdict=font1)
    ax[0].set_ylabel('Amplitude (μV)', fontdict=font1)
    ax[0].set_ylim(-1.24, 1.24, 0.25)
    ax[0].bar(1, 0, label='Real_EEG', color='blue')
    ax[0].bar(1, 0, label='Gen_EEG', color='orange')
    ax[0].legend(prop=font2, loc='upper right', framealpha=1.0)

    for size in ax[0].get_xticklabels():  # 获取x轴上所有坐标，并设置字号
        size.set_fontname('Times New Roman')
        size.set_fontsize('15')

    for size in ax[0].get_yticklabels():  # 获取y轴上所有坐标，并设置字号
        size.set_fontname('Times New Roman')
        size.set_fontsize('15')

    R_energy_spectrum = np.sqrt(np.square(R_Y.real) + np.square(R_Y.imag))
    max_index = np.argmax(R_energy_spectrum)
    # Always place grid at bottom
    ax[1].grid(True)
    ax[1].plot(frq, abs(R_Y), color='magenta')
    ax[1].plot(frq, abs(F_Y), color='green')
    ax[1].bar(1, 0, label='Real_EEG', color='magenta')
    ax[1].bar(1, 0, label='Gen_EEG', color='green')
    ax[1].set_xlabel('Freq (Hz)', fontdict=font1)
    ax[1].set_ylabel('Amplitude (μV)', fontdict=font1)
    ax[1].legend(prop=font2, loc='upper right', framealpha=1.0)

    marker_freq = targets[class_num]
    while marker_freq <= 40:
       circle_x = int(marker_freq) + 0.5 if marker_freq + 0.5 > int(marker_freq) + 1 else int(marker_freq)
       y1 = abs(R_Y)[int(circle_x * int(1 / FR))]
       y2 = abs(F_Y)[int(circle_x * int(1 / FR))]
       circle_y = y1 if y1 > y2 else y2
       ax[1].scatter(x=circle_x, y=circle_y, facecolors='none', marker='o', edgecolors='r',
                     s=200)  # set the color to null and control the color through edgecolors
       marker_freq += targets[class_num]

    # ax[1].text(max_index * FR, R_energy_spectrum[max_index], str(max_index * FR) + 'Hz', color='r')
    ax[1].set_ylim(0, R_energy_spectrum[max_index] + 0.2)
    # title = 'Time and Frequency Representation Comparison on Class {} '.format(class_num)
    # ax[0].set_title(title)
    for size in ax[1].get_xticklabels():  # Get all coordinates on the x-axis and set font size
        size.set_fontname('Times New Roman')
        size.set_fontsize('15')

    for size in ax[1].get_yticklabels():  # Get all coordinates on the y-axis and set font size
        size.set_fontname('Times New Roman')
        size.set_fontsize('15')

    plt.tight_layout()
    plt.savefig(f'../Figure/GEN_EEG_CMP/{opt.dataset}_S{subject}_Class{class_num}_CMP.png')
    plt.show()
