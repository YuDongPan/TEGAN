# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2023/2/26 13:21
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import Utils.Ploter as ploter

x = np.arange(6)

plt.figure(figsize=(14, 8), dpi=240)
# adjust the direction of ticks
matplotlib.rcParams['ytick.direction'] = 'in'

# Always place grid at bottom
plt.rc('axes', axisbelow=True)
plt.grid(axis='y', linestyle='-', linewidth=1, alpha=0.8)

font1 = {'family': 'Arial', 'weight': 'normal', 'size': 25}
is_itr = True
is_Direction = False

'''Direction SSVEP Dataset'''
# if not is_itr:
#     itcca_mean = [40.27, 45.87, 50.26, 54.95, 58.93, 61.47]
#     itcca_var = [14.72, 18.65, 20.65, 21.71, 22.86, 23.07]
#     itcca_aug_mean = [75.70, 79.68, 83.06, 85.04, 85.70, 86.01]
#     itcca_aug_var = [18.54, 17.11, 16.93, 16.14, 16.23, 16.22]
#     trca_mean = [67.92, 72.69, 75.78, 79.16, 81.46, 83.02]
#     trca_var = [21.89, 21.10, 21.16, 20.74, 20.43, 19.37]
#     trca_aug_mean = [75.63, 79.38, 83.06, 84.79, 85.73, 86.07]
#     trca_aug_var = [18.59, 17.39, 16.91, 16.33, 16.22, 16.22]
#     eegnet_mean = [39.60, 39.66, 48.56, 50.07, 51.75, 54.45]
#     eegnet_var = [9.70, 10.57, 15.93, 16.93, 18.32, 19.22]
#     eegnet_aug_mean = [75.27, 78.94, 82.44, 84.08, 84.84, 85.45]
#     eegnet_aug_var = [18.39, 17.22, 17.00, 16.04, 16.48, 16.72]
#     c_cnn_mean = [50.69, 54.24, 56.32, 58.87, 60.30, 61.12]
#     c_cnn_var = [17.70, 19.42, 19.94, 20.11, 20.58, 19.91]
#     c_cnn_aug_mean = [75.65, 79.64, 83.19, 84.99, 85.75, 86.11]
#     c_cnn_aug_var = [18.59, 17.10, 16.83, 16.07, 16.23, 16.15]
#     y1 = [106 for i in range(6)]
#     y2 = [109 for j in range(6)]
#     y3 = [112 for k in range(6)]
#     y4 = [115 for l in range(6)]
#
#     itcca_sig = {'xstart': x, 'xend': x, 'ystart': y1, 'yend': [y + 0.8 for y in y1],
#                  'marker': ['***', '***', '***', '***', '***', '***']}
#     trca_sig = {'xstart': x, 'xend': x, 'ystart': y2, 'yend': [y + 0.8 for y in y2],
#                  'marker': ['***', '***', '***', '***', '***', '**']}
#     eegnet_sig = {'xstart': x, 'xend': x, 'ystart': y3, 'yend': [y + 0.8 for y in y3],
#                  'marker': ['***', '***', '***', '***', '***', '***']}
#     c_cnn_sig = {'xstart': x, 'xend': x, 'ystart': y4, 'yend': [y + 0.8 for y in y4],
#                  'marker': ['***', '***', '***', '***', '***', '***']}
#
# else:
#     itcca_mean = [9.32, 14.26, 17.75, 21.03, 23.90, 24.83]
#     itcca_var = [14.52, 20.14, 22.98, 24.15, 25.23, 24.74]
#     itcca_aug_mean = [58.05, 59.58, 61.19, 59.98, 57.08, 53.82]
#     itcca_aug_var = [34.07, 30.22, 27.93, 25.36, 23.91, 22.32]
#     trca_mean = [46.93, 50.27, 51.44, 53.05, 52.68, 51.24]
#     trca_var = [37.30, 34.89, 33.19, 30.75, 28.40, 25.79]
#     trca_aug_mean = [57.94, 59.22, 61.18, 59.61, 57.12, 53.92]
#     trca_aug_var = [34.10, 30.53, 28.03, 25.54, 23.92, 22.34]
#     eegnet_mean = [6.87, 6.52, 14.00, 14.46, 15.47, 17.05]
#     eegnet_var = [7.55, 8.29, 17.19, 17.33, 18.35, 19.64]
#     eegnet_aug_mean = [57.05, 58.25, 59.92, 57.99, 55.58, 52.78]
#     eegnet_aug_var = [33.68, 30.36, 27.66, 24.98, 23.96, 22.13]
#     c_cnn_mean = [19.72, 22.63, 23.23, 24.06, 24.16, 23.05]
#     c_cnn_var = [23.23, 24.68, 25.16, 24.74, 24.35, 22.17]
#     c_cnn_aug_mean = [58.01, 59.55, 61.42, 59.84, 57.16, 53.95]
#     c_cnn_aug_var = [34.07, 30.22, 27.91, 25.30, 23.91, 22.26]
#
#     y1 = [102 for i in range(6)]
#     y2 = [105 for j in range(6)]
#     y3 = [108 for k in range(6)]
#     y4 = [111 for l in range(6)]
#
#     itcca_sig = {'xstart': x, 'xend': x, 'ystart': y1, 'yend': [y + 0.8 for y in y1],
#                  'marker': ['***', '***', '***', '***', '***', '***']}
#     trca_sig = {'xstart': x, 'xend': x, 'ystart': y2, 'yend': [y + 0.8 for y in y2],
#                  'marker': ['***', '***', '***', '***', '**', '*']}
#     eegnet_sig = {'xstart': x, 'xend': x, 'ystart': y3, 'yend': [y + 0.8 for y in y3],
#                  'marker': ['***', '***', '***', '***', '***', '***']}
#     c_cnn_sig = {'xstart': x, 'xend': x, 'ystart': y4, 'yend': [y + 0.8 for y in y4],
#                  'marker': ['***', '***', '***', '***', '***', '***']}

'''Dial SSVEP Dataset'''
if not is_itr:
    itcca_mean = [42.44, 48.38, 54.14, 59.44, 63.38, 67.01]
    itcca_var = [18.77, 21.54, 22.73, 23.42, 23.96, 24.66]
    itcca_aug_mean = [73.24, 74.65, 77.39, 79.92, 81.54, 84.61]
    itcca_aug_var = [21.35, 20.82, 20.32, 18.92, 18.97, 16.54]
    trca_mean = [64.99, 68.53, 73.08, 75.79, 79.72, 82.93]
    trca_var = [21.70, 22.29, 21.67, 20.76, 19.90, 17.39]
    trca_aug_mean = [71.53, 73.25, 76.35, 78.42, 79.00, 82.44]
    trca_aug_var = [20.89, 20.20, 20.09, 18.23, 18.90, 17.19]
    eegnet_mean = [35.44, 39.34, 45.87, 48.02, 53.72, 55.63]
    eegnet_var = [13.93, 16.72, 18.54, 20.51, 21.84, 23.58]
    eegnet_aug_mean = [72.48, 73.95, 76.82, 79.02, 80.77, 83.43]
    eegnet_aug_var = [21.45, 21.26, 20.59, 19.24, 19.38, 17.59]
    c_cnn_mean = [55.46, 57.75, 61.14, 64.36, 66.44, 70.42]
    c_cnn_var = [18.41, 19.94, 19.73, 21.28, 22.38, 22.66]
    c_cnn_aug_mean = [73.21, 74.80, 77.40, 79.78, 81.16, 84.15]
    c_cnn_aug_var = [21.13, 20.47, 20.31, 18.77, 19.00, 16.92]

    y1 = [102 for i in range(6)]
    y2 = [106 for j in range(6)]
    y3 = [110 for k in range(6)]
    y4 = [114 for l in range(6)]

    itcca_sig = {'xstart': x, 'xend': x, 'ystart': y1, 'yend': [y + 0.8 for y in y1],
                 'marker': ['***', '***', '***', '***', '***', '**']}
    trca_sig = {'xstart': x, 'xend': x, 'ystart': y2, 'yend': [y + 0.8 for y in y2],
                 'marker': ['**', '*', '-', '-', '-', '-']}
    eegnet_sig = {'xstart': x, 'xend': x, 'ystart': y3, 'yend': [y + 0.8 for y in y3],
                 'marker': ['***', '***', '***', '***', '***', '***']}
    c_cnn_sig = {'xstart': x, 'xend': x, 'ystart': y4, 'yend': [y + 0.8 for y in y4],
                 'marker': ['**', '**', '**', '**', '*', '**']}

else:
    itcca_mean = [43.44, 51.62, 58.32, 63.98, 67.05, 69.87]
    itcca_var = [34.26, 38.81, 40.88, 42.07, 41.80, 42.14]
    itcca_aug_mean = [120.03, 112.61, 110.36, 107.67, 104.13, 103.44]
    itcca_aug_var = [58.94, 53.71, 49.38, 44.37, 41.49, 35.82]
    trca_mean = [96.81, 97.48, 100.27, 99.09, 100.93, 100.46]
    trca_var = [56.23, 54.77, 51.44, 47.71, 44.11, 37.35]
    trca_aug_mean = [114.49, 108.02, 107.30, 103.22, 98.00, 98.85]
    trca_aug_var = [56.52, 51.30, 48.20, 41.87, 40.56, 36.18]
    eegnet_mean = [29.00, 33.62, 41.51, 42.79, 49.15, 49.82]
    eegnet_var = [23.07, 27.21, 30.29, 34.27, 36.69, 36.97]
    eegnet_aug_mean = [117.72, 110.92, 108.96, 105.48, 102.47, 101.18]
    eegnet_aug_var = [59.17, 54.39, 49.39, 44.31, 41.78, 37.46]
    c_cnn_mean = [66.09, 68.41, 73.63, 71.39, 73.37, 69.24]
    c_cnn_var = [39.93, 40.87, 41.10, 38.43, 38.17, 36.79]
    c_cnn_aug_mean = [119.72, 112.77, 110.39, 107.19, 103.22, 102.57]
    c_cnn_aug_var = [58.79, 53.20, 49.23, 43.87, 41.33, 36.50]

    y1 = [182 for i in range(6)]
    y2 = [190 for j in range(6)]
    y3 = [198 for k in range(6)]
    y4 = [206 for l in range(6)]

    itcca_sig = {'xstart': x, 'xend': x, 'ystart': y1, 'yend': [y + 0.8 for y in y1],
                 'marker': ['***', '***', '***', '***', '***', '**']}
    trca_sig = {'xstart': x, 'xend': x, 'ystart': y2, 'yend': [y + 0.8 for y in y2],
                'marker': ['**', '*', '-', '-', '-', '-']}
    eegnet_sig = {'xstart': x, 'xend': x, 'ystart': y3, 'yend': [y + 0.8 for y in y3],
                  'marker': ['***', '***', '***', '***', '***', '***']}
    c_cnn_sig = {'xstart': x, 'xend': x, 'ystart': y4, 'yend': [y + 0.8 for y in y4],
                 'marker': ['**', '**', '**', '**', '**', '**']}


def calibration_variance(acc_lst, var_lst):
    up_var_lst = var_lst
    down_var_lst = [var if (acc - var) >= 1 else acc - 2.0 for acc, var in zip(acc_lst, var_lst)]
    yerr_lst = [down_var_lst, up_var_lst]
    return yerr_lst

calibrated_itcca_var = calibration_variance(itcca_mean, itcca_var)
calibrated_eegnet_var = calibration_variance(eegnet_mean, eegnet_var)
calibrated_trca_var = calibration_variance(trca_mean, trca_var)
calibrated_c_cnn_var = calibration_variance(c_cnn_mean, c_cnn_var)

print("calibrated_itcca_var:", calibrated_itcca_var)
print("calibrated_eegnet_var:", calibrated_eegnet_var)
print("calibrated_trca_var:", calibrated_trca_var)
print("calibrated_c_cnn_var:", calibrated_c_cnn_var)

plt.errorbar(x, itcca_mean, yerr=calibrated_itcca_var, capsize=8, fmt='--', color='#33CC99', marker='o', markersize=9,
             label='ITCCA', linewidth=3, elinewidth=3)
plt.errorbar(x, itcca_aug_mean, yerr=itcca_aug_var, capsize=8, color='#33CC99', marker='o', markersize=9, label='ITCCA_AUG',
             linewidth=3, elinewidth=3)
plt.errorbar(x, trca_mean, yerr=trca_var, capsize=8, fmt='--', color='#FF9966', marker='p', markersize=9, label='TRCA',
             linewidth=3, elinewidth=3)
plt.errorbar(x, trca_aug_mean, yerr=calibrated_trca_var, capsize=8, color='#FF9966', marker='p', markersize=9, label='TRCA_AUG',
             linewidth=3, elinewidth=3)
plt.errorbar(x, eegnet_mean, yerr=calibrated_eegnet_var, capsize=8, fmt='--', color='#FF6347', marker='D', markersize=9,
             label='EEGNet', linewidth=3, elinewidth=3)
plt.errorbar(x, eegnet_aug_mean, yerr=eegnet_aug_var, capsize=8,  color='#FF6347', marker='D', markersize=9,
             label='EEGNet_AUG', linewidth=3, elinewidth=3)
plt.errorbar(x, c_cnn_mean, yerr=calibrated_c_cnn_var, capsize=8, fmt='--', color='#9999FF', marker='H', markersize=9,
             label='C_CNN', linewidth=3, elinewidth=3)
plt.errorbar(x, c_cnn_aug_mean, yerr=c_cnn_aug_var, capsize=8, color='#9999FF', marker='H', markersize=9,
             label='C_CNN_AUG', linewidth=3, elinewidth=3)

ploter.plot_sig(itcca_sig['xstart'], itcca_sig['xend'], itcca_sig['ystart'], itcca_sig['yend'],
                sig=itcca_sig['marker'], color='#33CC99')
ploter.plot_sig(trca_sig['xstart'], trca_sig['xend'], trca_sig['ystart'], trca_sig['yend'],
                sig=trca_sig['marker'], color='#FF9966')
ploter.plot_sig(eegnet_sig['xstart'], eegnet_sig['xend'], eegnet_sig['ystart'], eegnet_sig['yend'],
                sig=eegnet_sig['marker'], color='#FF6347')
ploter.plot_sig(c_cnn_sig['xstart'], c_cnn_sig['xend'], c_cnn_sig['ystart'], c_cnn_sig['yend'],
                sig=c_cnn_sig['marker'], color='#9999FF')

# plt.legend(prop=font1, ncol=8, framealpha=1.0, loc='upper center')

plt.xlabel('Time window (s)', fontproperties='Times New Roman', fontsize=25)
x_ticks_bound = [i for i in range(6)]
x_ticks_content = ['0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
plt.xticks(x_ticks_bound, x_ticks_content, fontproperties='Arial', fontsize=20)

if is_itr is not True:
    plt.ylim(20, 122)
    plt.ylabel('Accuracy(%)', fontproperties='Times New Roman', fontsize=25)
    y_ticks_bound = [i * 10 for i in range(2, 11)]
    y_ticks_content = [str(i * 10) for i in range(2, 11)]
    plt.yticks(y_ticks_bound, y_ticks_content, fontproperties='Arial', fontsize=20)

else:
    plt.ylabel('ITR(bits/min)', fontproperties='Times New Roman', fontsize=25)
    if is_Direction:
        plt.ylim(0, 120)
        y_ticks_bound = [i * 10 for i in range(0, 11)]
        y_ticks_content = [str(i * 10) for i in range(0, 11)]
    else:
        plt.ylim(0, 215)
        y_ticks_bound = [i * 20 for i in range(0, 10)]
        y_ticks_content = [str(i * 20) for i in range(0, 10)]
    plt.yticks(y_ticks_bound, y_ticks_content, fontproperties='Arial', fontsize=20)


ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)  ### set the thickness of the bottom axis
ax.spines['left'].set_linewidth(2)   #### set the thickness of the left axis
ax.spines['right'].set_linewidth(2)  #### set the thickness of the right axis
ax.spines['top'].set_linewidth(2)    #### set the thickness of the top axis
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()