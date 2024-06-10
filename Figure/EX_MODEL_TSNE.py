# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2023/2/10 10:33
import torch
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import Utils.EEGDataset as EEGDataset
from sklearn import manifold
from Model import CCNN, EEGNet
from Model import TEGAN_V20 as TEGAN
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()

'''Direction Dataset'''
# parser.add_argument('--dataset', type=str, default='Direction', help="4-class dataset")
# parser.add_argument('--factor', type=int, default=1, help="factor size of ssvep")
# parser.add_argument('--ws', type=float, default=1.0, help="window size of ssvep")
# parser.add_argument('--Kf', type=int, default=1, help="k-fold cross validation")
# parser.add_argument('--Nh', type=int, default=100, help="number of trial")
# parser.add_argument('--Nc', type=int, default=10, help="number of channel")
# parser.add_argument('--Fs', type=int, default=100, help="frequency of sample")
# parser.add_argument('--Nf', type=int, default=4, help="number of stimulus")
# parser.add_argument('--Ns', type=int, default=54, help="number of subjects")
# parser.add_argument('--UD', type=int, default=1, help="-1(Unsupervised),0(User-dependent),1(User-Indepedent)")
# parser.add_argument('--ratio', type=int, default=0, help="-1(Training-free),0(N-1vs1),1(8vs2),2(5v5),3(2v8)")
# parser.add_argument('--subject', type=int, default=5, help="the subject to be plotted")

'''Dial Dataset'''
parser.add_argument('--dataset', type=str, default='Dial', help="12-class dataset")
parser.add_argument('--factor', type=int, default=2, help="factor size of ssvep")
parser.add_argument('--ws', type=float, default=1.0, help="window size of ssvep")
parser.add_argument('--Kf', type=int, default=1, help="k-fold cross validation")
parser.add_argument('--Nh', type=int, default=180, help="number of trial")
parser.add_argument('--Nc', type=int, default=8, help="number of channel")
parser.add_argument('--Fs', type=int, default=256, help="frequency of sample")
parser.add_argument('--Nt', type=int, default=512, help="number of sample")
parser.add_argument('--Nf', type=int, default=12, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=10, help="number of subjects")
parser.add_argument('--UD', type=int, default=1, help="-1(Unsupervised),0(User-dependent),1(User-Indepedent)")
parser.add_argument('--ratio', type=int, default=0, help="-1(Training-free),0(N-1vs1),1(8vs2),2(5v5),3(2v8)")
parser.add_argument('--subject', type=int, default=5, help="the subject to be plotted")

opt = parser.parse_args()

# set plotting style
fig, ax_arr = plt.subplots(2, 2, figsize=(18, 10), dpi=240)
sns.set(rc={'figure.figsize': (24, 6)})
plt.rc('font', family='Times New Roman')
color_list = ['#0000C6', '#FF0000', '#00FFFF', '#FF359A', '#FF00FF', '#B15BFF', '#02F78E', '#F9F900', '#FFA042',
              '#FF5809', '#984B4B', '#3D7878']
palette12 = sns.color_palette(palette=color_list, n_colors=12)

devices = "cuda" if torch.cuda.is_available() else "cpu"

# load eeg dataset
# EEGData_Test = EEGDataset.getSSVEP12Inter(subject=opt.subject, mode="test")
EEGData_Test = EEGDataset.getSSVEP12Intra(subject=opt.subject, mode="train", n_splits=5, KFold=0)
X_ORG, y = EEGData_Test[:]
X_ORG = X_ORG[:, :, :, :int(opt.Fs * opt.ws)]
X_ORG = X_ORG.type(torch.FloatTensor)
eeg_generator = TEGAN.Generator(opt.Nc, int(opt.Fs * opt.ws), opt.Nf, opt.ws, factor=opt.factor)
eeg_generator.load_state_dict(torch.load(f'../Pretrain/Dial/{opt.ws}S-{opt.ws * opt.factor}S/'
                                         f'TEGAN_Gt_S{opt.subject}.pth'))
eeg_generator.eval()
X_AUG = eeg_generator(X_ORG.float())[0].detach()

y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64)
print("X.shape", X_AUG.shape)
print("y.shape", y.shape)

# obtain spectral data and filtered data
spectral_ORG = C_CNN.complex_spectrum_features(X_ORG.numpy(), FFT_PARAMS=[256, 1.0])
spectral_AUG = C_CNN.complex_spectrum_features(X_AUG.numpy(), FFT_PARAMS=[256, 2.0])
spectral_ORG = torch.from_numpy(spectral_ORG)
spectral_AUG = torch.from_numpy(spectral_AUG)
spectral_ORG = spectral_ORG.type(torch.FloatTensor)
spectral_AUG = spectral_AUG.type(torch.FloatTensor)

# load pre-trained model
EEGNet1 = EEGNet.EEGNet(opt.Nc, int(opt.Fs * opt.ws), opt.Nf)
EEGNet2 = EEGNet.EEGNet(opt.Nc, int(opt.Fs * opt.ws * opt.factor), opt.Nf)
C_CNN1 = C_CNN.CNN(opt.Nc, int(220 * 1.0), opt.Nf)
C_CNN2 = C_CNN.CNN(opt.Nc, int(220 * 1.0), opt.Nf)

EEGNet1.load_state_dict(torch.load(f'../Pretrain/EEGNet_ORG_S{opt.subject}.pth'))
EEGNet2.load_state_dict(torch.load(f'../Pretrain/EEGNet_AUG_S{opt.subject}.pth'))
C_CNN1.load_state_dict(torch.load(f'../Pretrain/C_CNN_ORG_S{opt.subject}.pth'))
C_CNN2.load_state_dict(torch.load(f'../Pretrain/C_CNN_AUG_S{opt.subject}.pth'))
EEGNet1 = EEGNet1.to(devices)
EEGNet2 = EEGNet2.to(devices)
C_CNN1 = C_CNN1.to(devices)
C_CNN2 = C_CNN2.to(devices)

# utilize model to calculate features space
EEGNet1.eval()
EEGNet2.eval()

X_ORG = X_ORG.to(devices)
X_AUG = X_AUG.to(devices)
y1 = y.to(devices)
y2 = y.to(devices)
spectral_ORG = spectral_ORG.to(devices)
spectral_AUG = spectral_AUG.to(devices)

F1, _ = C_CNN1(spectral_ORG)
F2, _ = C_CNN2(spectral_AUG)
F3, _ = EEGNet1(X_ORG)
F4, _ = EEGNet2(X_AUG)

F1 = F1.cpu().data.numpy()
F2 = F2.cpu().data.numpy()
F3 = F3.cpu().data.numpy()
F4 = F4.cpu().data.numpy()

y1 = y1.cpu().data.numpy()
y2 = y2.cpu().data.numpy()

# t-sne visualization
t_sne = manifold.TSNE()
F1_embedded = t_sne.fit_transform(F1)
F2_embedded = t_sne.fit_transform(F2)
F3_embedded = t_sne.fit_transform(F3)
F4_embedded = t_sne.fit_transform(F4)

scaler = MinMaxScaler()
F1_embedded = scaler.fit_transform(F1_embedded)
F2_embedded = scaler.fit_transform(F2_embedded)
F3_embedded = scaler.fit_transform(F3_embedded)
F4_embedded = scaler.fit_transform(F4_embedded)

sns.set_style("white")
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
xy_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
ax1 = sns.scatterplot(F1_embedded[:, 0], F1_embedded[:, 1], hue=y1, legend=False, palette=palette12, ax=ax_arr[0][0])
ax1.set_title('(a)', y=-0.15, fontproperties=font)
ax1.set_xticks(xy_ticks)
ax1.set_yticks(xy_ticks)
ax1.set_xticklabels(labels=xy_ticks, fontproperties=font)
ax1.set_yticklabels(labels=xy_ticks, fontproperties=font)


ax2 = sns.scatterplot(F2_embedded[:, 0], F2_embedded[:, 1], hue=y1, legend='full', palette=palette12, ax=ax_arr[0][1])
ax2.set_title('(b)', y=-0.15, fontproperties=font)
ax2.set_xticks(xy_ticks)
ax2.set_yticks(xy_ticks)
ax2.set_xticklabels(labels=xy_ticks, fontproperties=font)
ax2.set_yticklabels(labels=xy_ticks, fontproperties=font)
ax2.legend(loc=3, bbox_to_anchor=(1, 0.1))


ax3 = sns.scatterplot(F3_embedded[:, 0], F3_embedded[:, 1], hue=y2, legend=False, palette=palette12, ax=ax_arr[1][0])
ax3.set_title('(c)', y=-0.15, fontproperties=font)
ax3.set_xticks(xy_ticks)
ax3.set_yticks(xy_ticks)
ax3.set_xticklabels(labels=xy_ticks, fontproperties=font)
ax3.set_yticklabels(labels=xy_ticks, fontproperties=font)


ax4 = sns.scatterplot(F4_embedded[:, 0], F4_embedded[:, 1], hue=y2, legend='full', palette=palette12, ax=ax_arr[1][1])
ax4.set_title('(d)', y=-0.15, fontproperties=font)
ax4.set_xticks(xy_ticks)
ax4.set_yticks(xy_ticks)
ax4.legend(loc=3, bbox_to_anchor=(1, 0.1))
ax4.set_xticklabels(labels=xy_ticks, fontproperties=font)
ax4.set_yticklabels(labels=xy_ticks, fontproperties=font)

plt.savefig('../Result/tSNE/EEGNet_C_CNN_CMP.png')
plt.savefig('../Result/tSNE/EEGNet_C_CNN_CMP.eps')
plt.show()