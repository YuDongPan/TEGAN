# Designer:Yudong Pan
# Coder:God's hand
# Time:2023/3/13 11:07
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

N = 6
interval = 0.99
a = [i * interval for i in range(N)]
b = [i * interval + 0.16 for i in range(N)]
is_Direction = False

if is_Direction:
    '''Direction SSVEP Dataset'''
    gen_param = [0.68, 0.75, 0.79, 0.85, 0.89, 0.92]
    dis_param = [0.20, 0.29, 0.38, 0.50, 0.63, 0.77]

else:
    '''Dial SSVEP Dataset'''
    gen_param = [0.47, 0.52, 0.56, 0.60, 0.63, 0.66]
    dis_param = [0.81, 1.18, 1.61, 2.07, 2.64, 3.28]

# set size of figure
plt.figure(figsize=(16, 8), dpi=160)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

# Always place grid at bottom
plt.rc('axes', axisbelow=True)
plt.grid(axis='y', linestyle='-', linewidth=1, alpha=0.8)

color_lst = ['#FF6347', '#FF6666', '#FFCC66', '#FFD700', '#6699FF', '#1E90FF', '#6633CC', '#9932CC']
plt.bar(a, gen_param, color=color_lst[0], width=0.15, label='Generator')
plt.bar(b, dis_param, color=color_lst[2], width=0.15, label='Discriminator')

# set value range for axis x and axis y
plt.ylim(0, 1.12) if is_Direction else plt.ylim(0, 3.95)

# set ticks for x axis and y axis
plt.xlabel('Time window (s)', fontproperties='Times New Roman', fontsize=25)
x_ticks_bound = [i * interval + 0.08 for i in range(N)]
x_ticks_content = ['0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
plt.xticks(x_ticks_bound, x_ticks_content, fontproperties='Times New Roman', fontsize=25)


plt.ylabel('Parameters (M)', fontproperties='Times New Roman', fontsize=25)
if is_Direction:
    y_ticks_bound = [float(i * 0.1) for i in range(0, 12, 2)]
    y_ticks_content = [0 if y_ticks == 0 else round(y_ticks, 1) for y_ticks in y_ticks_bound]
    plt.yticks(y_ticks_bound, y_ticks_content, fontproperties='Times New Roman', size=25)
else:
    y_ticks_bound = [i * 0.1 for i in range(0, 36, 5)]
    y_ticks_content = [0 if y_ticks == 0 else round(y_ticks, 1) for y_ticks in y_ticks_bound]
    plt.yticks(y_ticks_bound, y_ticks_content, fontproperties='Times New Roman', size=25)

# set label for data
plt.legend(prop=font1, ncol=1, loc='upper right', framealpha=1.0)


ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)  ### set the thickness of the bottom axis
ax.spines['left'].set_linewidth(2)   #### set the thickness of the left axis
ax.spines['right'].set_linewidth(2)  #### set the thickness of the right axis
ax.spines['top'].set_linewidth(2)    #### set the thickness of the top axis
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()
