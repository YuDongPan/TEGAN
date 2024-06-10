# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/9/26 22:17
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import Utils.Ploter as ploter

x = np.arange(6)

plt.figure(figsize=(12, 8), dpi=240)
# adjust the direction of ticks
matplotlib.rcParams['ytick.direction'] = 'in'

# Always place grid at bottom
plt.rc('axes', axisbelow=True)
plt.grid(axis='y', linestyle='-', linewidth=1, alpha=0.8)

font1 = {'family': 'Arial', 'weight': 'normal', 'size': 20}

'''Direction:1.0s'''
win_size = 1.0
itcca_mean = [61.47, 86.01, 66.51, 89.80, 69.85, 92.48]
itcca_var = [23.07, 16.22, 21.91, 14.27, 21.55, 11.62]
trca_mean = [83.02, 86.07, 89.89, 89.56, 92.26, 92.56]
trca_var = [19.37, 16.22, 15.90, 14.47, 13.50, 11.83]
eegnet_mean = [54.45, 85.45, 72.12, 89.43, 83.33, 92.37]
eegnet_var = [19.22, 16.22, 19.03, 14.01, 16.39, 11.76]
c_cnn_mean = [61.12, 86.11, 75.58, 89.77, 86.22, 92.56]
c_cnn_var = [19.91, 16.16, 20.03, 14.16, 16.18, 11.79]

itcca_sig = ['***', '***', '***']
trca_sig = ['**',  '-', '-']
eegnet_sig = ['***', '***', '***']
c_cnn_sig = ['***', '***', '***']

'''Dial:1.0s'''
# win_size = 1.0
# itcca_mean = [67.01, 84.61, 77.62, 93.81, 79.22, 96.11]
# itcca_var = [24.66, 16.54, 19.57, 8.87, 19.28, 5.79]
# trca_mean = [82.93, 82.44, 93.45, 93.57, 96.00, 96.06]
# trca_var = [17.39, 17.19, 8.35, 9.02, 4.33, 5.81]
# eegnet_mean = [55.63, 83.43, 80.11, 92.61, 89.90, 94.83]
# eegnet_var = [23.58, 17.59, 19.92, 9.94, 12.58, 7.17]
# c_cnn_mean = [67.51, 84.15, 86.06, 93.50, 92.13, 94.90]
# c_cnn_var = [21.69, 16.92, 19.70, 9.21, 11.48, 7.12]
#
# itcca_sig = ['**', '**', '**']
# trca_sig = ['-',  '-', '-']
# eegnet_sig = ['***', '**', '*']
# c_cnn_sig = ['**', '*', '*']


plt.errorbar(x, itcca_mean, yerr=itcca_var, capsize=8, color='#33CC99', marker='o', markersize=9, label='ITCCA',
             linewidth=3, elinewidth=3)
plt.errorbar(x, trca_mean, yerr=trca_var, capsize=8, color='#FF9966', marker='p', markersize=9, label='TRCA',
             linewidth=3, elinewidth=3)
plt.errorbar(x, eegnet_mean, yerr=eegnet_var, capsize=8, fmt='--', color='#FF6347', marker='D', markersize=9,
             label='EEGNet', linewidth=3, elinewidth=3)
plt.errorbar(x, c_cnn_mean, yerr=c_cnn_var, capsize=8, fmt='--', color='#9999FF', marker='H', markersize=9,
             label='C-CNN', linewidth=3, elinewidth=3)

ploter.plot_sig([0, 2, 4], [1, 3, 5], [108, 108, 108], [108.8, 108.8, 108.8], sig=itcca_sig, color='#33CC99')
ploter.plot_sig([0, 2, 4], [1, 3, 5], [112, 112, 112], [112.8, 112.8, 112.8], sig=trca_sig, color='#FF9966')
ploter.plot_sig([0, 2, 4], [1, 3, 5], [116, 116, 116], [116.8, 116.8, 116.8], sig=eegnet_sig, color='#FF6347')
ploter.plot_sig([0, 2, 4], [1, 3, 5], [120, 120, 120], [120.8, 120.8, 120.8], sig=c_cnn_sig, color='#9999FF')

plt.ylabel('Accuracy(%)', fontproperties='Times New Roman', fontsize=25)
# plt.xlabel('Training Sample Size', fontproperties='Times New Roman', fontsize=20)
plt.legend(prop=font1, ncol=4, framealpha=1.0)

x_ticks_bound = [i for i in range(6)]
x_ticks_content = ['small_org', 'small_aug', 'middle_org', 'middle_aug', 'large_org', 'large_aug']
plt.xticks(x_ticks_bound, x_ticks_content, fontproperties='Arial', fontsize=22)

plt.ylim(28, 122)
y_ticks_bound = [i * 10 for i in range(3, 11)]
y_ticks_content = [str(i * 10) for i in range(3, 11)]
plt.yticks(y_ticks_bound, y_ticks_content, fontproperties='Arial', fontsize=20)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)  ### set the thickness of the bottom axis
ax.spines['left'].set_linewidth(2)   #### set the thickness of the left axis
ax.spines['right'].set_linewidth(2)  #### set the thickness of the right axis
ax.spines['top'].set_linewidth(2)    #### set the thickness of the top axis
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()