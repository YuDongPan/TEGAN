# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/9/8 22:56
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

import numpy as np

# In order to efficiently plot many lines in a single set of axes,
# Matplotlib has the ability to add the lines all at once.
# Here is a simple example showing how it is done.
N = 50
x = np.arange(N)
# Here are many sets of y to plot vs. x
ys = [np.sin(2 * x) + i * 10 for i in range(10)]

# We need to set the plot limits, they will not autoscale
fig, ax = plt.subplots()
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(ys), np.max(ys) + 10)

# colors is sequence of rgba tuples
# linestyle is a string or dash tuple. Legal string values are
#          solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq)
#          where onoffseq is an even length tuple of on and off ink in points.
#          If linestyle is omitted, 'solid' is used
# See `matplotlib.collections.LineCollection` for more information

# Make a sequence of (x, y) pairs.

segments = [np.column_stack([x, y]) for y in ys]
print("segments.shape:", np.asarray(segments).shape)
line_segments = LineCollection(segments,
                               linewidths=(0.5, 1, 1.5, 2),
                               linestyles='solid',)

line_segments.set_array(range(10))
ax.add_collection(line_segments)
axcb = fig.colorbar(line_segments)
axcb.set_label('Line Number')
ax.set_title('Line Collection with mapped colors')
plt.sci(line_segments)  # This allows interactive changing of the colormap.
plt.show()