# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2023/1/5 10:11
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Utils.Ploter as ploter

'''Direction SSVEP Dataset'''
mean_cca = [35.48, 65.76]
var_cca = [[1.0, 1.0],
           [9.34, 18.78]]

mean_cca_aug = [63.22, 83.33]
var_cca_aug = [[1.0, 1.0],
               [17.54, 16.69]]

mean_msi = [35.06, 62.46]
var_msi = [[1.0, 1.0],
           [9.01, 18.10]]

mean_msi_aug = [60.37, 81.63]
var_msi_aug = [[1.0, 1.0],
               [16.94, 17.60]]

mean_eegnet = [80.10, 88.64]
var_eegnet = [[1.0, 1.0],
           [15.24, 14.08]]

mean_eegnet_aug = [75.27, 85.91]
var_eegnet_aug = [[1.0, 1.0],
           [16.96, 15.60]]

mean_c_cnn = [75.98, 87.77]
var_c_cnn = [[1.0, 1.0],
               [16.38, 14.37]]

mean_c_cnn_aug = [74.89, 86.02]
var_c_cnn_aug = [[1.0, 1.0],
               [16.85, 15.39]]

cca_sig = {'xstart': [0, 1.1], 'xend': [0.12, 1.22], 'ystart': [85, 103], 'yend': [85.8, 103.8],
           'marker': ['***', '***']}
msi_sig = {'xstart': [0.2, 1.30], 'xend': [0.32, 1.42], 'ystart': [85, 103], 'yend': [85.8, 103.8],
           'marker': ['***', '***']}
eegnet_sig = {'xstart': [0.44, 1.54], 'xend': [0.55, 1.66], 'ystart': [98, 105], 'yend': [98.8, 105.8],
              'marker': ['-', '-']}
c_cnn_sig = {'xstart': [0.66, 1.75], 'xend': [0.78, 1.87], 'ystart': [98, 105], 'yend': [98.8, 105.8],
              'marker': ['-', '-']}

'''Dial SSVEP Dataset'''
# mean_cca = [23.50, 56.44]
# var_cca = [[1.0, 1.0],
#            [8.65, 22.49]]
#
# mean_cca_aug = [39.33, 69.67]
# var_cca_aug = [[1.0, 1.0],
#                [15.56, 24.98]]
#
# mean_msi = [21.17, 54.44]
# var_msi = [[1.0, 1.0],
#            [4.95, 18.66]]
#
# mean_msi_aug = [36.11, 69.17]
# var_msi_aug = [[1.0, 1.0],
#                [13.92, 25.79]]
#
# mean_eegnet = [61.83, 81.51]
# var_eegnet = [[1.0, 1.0],
#           [19.67, 17.48]]
#
# mean_eegnet_aug = [56.08, 75.92]
# var_eegnet_aug = [[1.0, 1.0],
#           [20.97, 23.42]]
#
# mean_c_cnn = [56.77, 80.23]
# var_c_cnn = [[1.0, 1.0],
#               [20.27, 23.40]]
#
# mean_c_cnn_aug = [56.04, 75.13]
# var_c_cnn_aug = [[1.0, 1.0],
#               [21.25, 22.67]]
#
# cca_sig = {'xstart': [0, 1.1], 'xend': [0.12, 1.22], 'ystart': [58, 98], 'yend': [58.8, 98.8],
#            'marker': ['**', '**']}
# msi_sig = {'xstart': [0.2, 1.30], 'xend': [0.32, 1.42], 'ystart': [58, 98], 'yend': [58.8, 98.8],
#            'marker': ['**', '**']}
# eegnet_sig = {'xstart': [0.44, 1.54], 'xend': [0.55, 1.66], 'ystart': [85, 106], 'yend': [85.8, 106.8],
#               'marker': ['-', '-']}
# c_cnn_sig = {'xstart': [0.66, 1.75], 'xend': [0.78, 1.87], 'ystart': [85, 106], 'yend': [85.8, 106.8],
#               'marker': ['-', '-']}

# set x axis element for bar
interval = 1.1
N = 2
a = [i * interval for i in range(N)]
b = [i * interval + 0.11 for i in range(N)]
c = [i * interval + 0.22 for i in range(N)]
d = [i * interval + 0.33 for i in range(N)]
e = [i * interval + 0.44 for i in range(N)]
f = [i * interval + 0.55 for i in range(N)]
g = [i * interval + 0.66 for i in range(N)]
h = [i * interval + 0.77 for i in range(N)]


# set size of figure
plt.figure(figsize=(16, 8), dpi=160)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}

# Always place grid at bottom
plt.rc('axes', axisbelow=True)
plt.grid(axis='y', linestyle='-', linewidth=1, alpha=0.8)

# plot the bars
color_lst = ['#FF6347', '#FF6666', '#FFCC66', '#FFD700', '#6699FF', '#1E90FF', '#6633CC', '#9932CC']
plt.bar(a, mean_cca, yerr=var_cca, error_kw={'ecolor': color_lst[0], 'elinewidth': 3, 'capsize': 5, 'capthick':3},
        color=color_lst[0], width=0.10, label='CCA')
plt.bar(b, mean_cca_aug, yerr=var_cca_aug, error_kw={'ecolor': color_lst[1], 'elinewidth': 3, 'capsize': 5, 'capthick':3},
        color=color_lst[1],  width=0.10, label='CCA_AUG')
plt.bar(c, mean_msi, yerr=var_msi, error_kw={'ecolor': color_lst[2], 'elinewidth': 3, 'capsize': 5, 'capthick':3},
        color=color_lst[2],  width=0.10, label='MSI')
plt.bar(d, mean_msi_aug, yerr=var_msi_aug, error_kw={'ecolor': color_lst[3], 'elinewidth': 3, 'capsize': 5, 'capthick':3},
        color=color_lst[3],  width=0.10, label='MSI_AUG')
plt.bar(e, mean_eegnet, yerr=var_eegnet, error_kw={'ecolor': color_lst[4], 'elinewidth': 3, 'capsize': 5, 'capthick':3},
        color=color_lst[4], width=0.10, label='EEGNet')
plt.bar(f, mean_eegnet_aug, yerr=var_eegnet_aug, error_kw={'ecolor': color_lst[5], 'elinewidth': 3, 'capsize': 5, 'capthick':3},
        color=color_lst[5], width=0.10, label='EEGNet_AUG')
plt.bar(g, mean_c_cnn, yerr=var_c_cnn, error_kw={'ecolor': color_lst[6], 'elinewidth': 3, 'capsize': 5, 'capthick':3},
        color=color_lst[6], width=0.10, label='C_CNN')
plt.bar(h, mean_c_cnn_aug, yerr=var_c_cnn_aug, error_kw={'ecolor': color_lst[7], 'elinewidth': 3, 'capsize': 5, 'capthick':3},
        color=color_lst[7], width=0.10, label='C_CNN_AUG')

ploter.plot_sig(cca_sig['xstart'], cca_sig['xend'], cca_sig['ystart'], cca_sig['yend'],
                sig=cca_sig['marker'], color='red')
ploter.plot_sig(msi_sig['xstart'], msi_sig['xend'], msi_sig['ystart'], msi_sig['yend'],
                sig=msi_sig['marker'], color='red')
ploter.plot_sig(eegnet_sig['xstart'], eegnet_sig['xend'], eegnet_sig['ystart'], eegnet_sig['yend'],
                sig=eegnet_sig['marker'], color='red')
ploter.plot_sig(c_cnn_sig['xstart'], c_cnn_sig['xend'], c_cnn_sig['ystart'], c_cnn_sig['yend'],
                sig=c_cnn_sig['marker'], color='red')


# set value range for axis x and axis y
plt.ylim(0, 110)

# set ticks for x axis and y axis
x_ticks_bound = [i * interval + 0.40 for i in range(N)]
x_ticks_content = ['0.5 s to 1 s',  '1.0 s to 2.0 s']
plt.xticks(x_ticks_bound, x_ticks_content, fontproperties='Times New Roman', size=18)

y_ticks_bound = [i * 10 for i in range(11)]
y_ticks_content = [str(i * 10) for i in range(11)]
plt.yticks(y_ticks_bound, y_ticks_content, fontproperties='Times New Roman', size=18)

# set label for data
plt.legend(prop=font1, ncol=4, loc='lower center', framealpha=1.0)
# plt.xlabel('Train Scenarios', fontproperties='Times New Roman', size=18)
plt.xlabel('Transformation Scenarios', fontproperties='Times New Roman', size=20)
plt.ylabel('Accuracy(%)', fontproperties='Times New Roman', size=20)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)  ### set the thickness of the bottom axis
ax.spines['left'].set_linewidth(2)   #### set the thickness of the left axis
ax.spines['right'].set_linewidth(2)  #### set the thickness of the right axis
ax.spines['top'].set_linewidth(2)    #### set the thickness of the top axis
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.title('Results of Classification Accuracy Across All Subjects', fontsize=15)

plt.show()