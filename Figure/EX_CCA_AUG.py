# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/12 21:18
import matplotlib
import matplotlib.pyplot as plt
mean_cca_direction = [86.13, 90.00, 91.89]   # (0.5s org, 1.0s aug, 1.0s org, 2.0 aug, 2.0 org)
var_cca_direction = [15.89, 13.96, 12.34]

mean_cca_dial = [66.81, 86.43, 91.72]
var_cca_dial = [18.96, 17.96, 13.21]


# set x axis element for bar
interval = 0.9
N = 3
a = [i * interval for i in range(N)]
b = [i * interval + 0.11 for i in range(N)]
c = [i * interval + 0.22 for i in range(N)]
d = [i * interval + 0.33 for i in range(N)]

# set size of figure
plt.figure(figsize=(16, 8), dpi=240)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
font1 = {'family': 'Times New Roman',
'weight' : 'normal',
'size': 18}

# add grid for y axis
# plt.rc('axes', axisbelow=True)
# plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.4)

# plot the bars
plt.bar(a, mean_cca_direction, yerr=var_cca_direction, error_kw={'ecolor': '0.2', 'capsize': 6}, width=0.1,
        color="#99ffff", label='EEGNet')
plt.bar(b, mean_cca_dial, yerr=var_cca_dial, error_kw={'ecolor': '0.2', 'capsize': 6}, color="#99ccff",
        width=0.1, label='EEGNet_W/O_FC')



# set value range for axis x and axis y
plt.ylim(0, 108)

# set ticks for x axis and y axis
x_ticks_bound = [i * interval + 0.20 for i in range(N)]
x_ticks_content = ['IntraC-2vs8',  'IntraC-5vs5', 'IntraC-8vs2']
plt.xticks(x_ticks_bound, x_ticks_content, fontproperties='Times New Roman', size=18)


y_ticks_bound = [i * 10 for i in range(11)]
y_ticks_content = [str(i * 10) for i in range(11)]
plt.yticks(y_ticks_bound, y_ticks_content, fontproperties='Times New Roman', size=18)


# set label for data
plt.legend(prop=font1, framealpha=1.0, loc='lower center')
plt.ylabel('Accuracy(%)', fontproperties='Times New Roman', size=18)


ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)  ### set the thickness of the bottom axis
ax.spines['left'].set_linewidth(2)   #### set the thickness of the left axis
ax.spines['right'].set_linewidth(2)  #### set the thickness of the right axis
ax.spines['top'].set_linewidth(2)    #### set the thickness of the top axis
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()