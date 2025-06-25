import matplotlib.pyplot as plt
import numpy as np
from utils import *

# 数据
models = ['Llama-65B', 'Llama2-70B', 'Llama3.1-405B']
hidden_sizes = [8192, 8192, 16384]
context_lens = [2048, 4096, 131072]

hidden_size_growth = int(hidden_sizes[-1] / hidden_sizes[0])
context_len_growth = int(context_lens[-1] / context_lens[0])

print(hidden_size_growth)
print(context_len_growth)

# 绘制横向条形图
figsize = {
    "figure.figsize": (6, 2),
    'font.sans-serif': 'Times New Roman',
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 12,
    'ytick.labelsize': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
}
plt.rcParams.update(figsize)
fig, ax = plt.subplots(1, 1)
color_def = COLOR_DEF[:len(models)]
hatch_def = HATCH_DEF[:len(models)]

height = 0.2
height_gap = 0.05

ylim = 2 ** 20
r1 = np.arange(2)
all_bars = []

# 水平横向，反转
models_reverse = models[::-1]
hidden_sizes_reverse = hidden_sizes[::-1]
context_lens_reverse = context_lens[::-1]
color_def = color_def[::-1]
hatch_def = hatch_def[::-1]
for i, model in enumerate(models_reverse):
  bars = ax.barh(r1, [context_lens_reverse[i], hidden_sizes_reverse[i]], color=color_def[i], height=height, edgecolor='k', hatch=hatch_def[i], label=model)
  r1 = [x + height + height_gap for x in r1]
  all_bars.append(bars)

  if i == 1:
    y0 = bars[0].get_y() + bars[0].get_height() / 2 + 0.18
    x0 = (2 ** 14 + 2 ** 13) / 2 + 2 ** 9
    y1 = bars[1].get_y() + bars[1].get_height() / 2 + 0.18
    x1 = (2 ** 14 + 2 ** 13) / 2
    # 反转
    ax.text(x1, y1, f'{hidden_size_growth}\u00D7', ha='center', va='center', fontweight='bold', fontsize=20, rotation=0)
    ax.text(x0, y0, f'{context_len_growth}\u00D7', ha='center', va='center', fontweight='bold', fontsize=20, rotation=0)

all_dst_x = []
all_dst_y = []
all_src_x = []
all_src_y = []
for i, bar in enumerate(all_bars[-1]):
  first_bar = all_bars[0][i]
  dst_x = bar.get_width()
  dst_y = bar.get_y() + bar.get_height() / 2

  src_x = first_bar.get_width()
  src_y = first_bar.get_y() + first_bar.get_height() / 2
  print(f"{src_x=}")
  # 反转
  ax.annotate(
    '',
    xy = (src_x, src_y),
    xytext = (dst_x, dst_y),
    arrowprops=dict(facecolor='black', shrink=0.05, width=2.5),
  )
  all_dst_x.append(dst_x)
  all_dst_y.append(dst_y)
  all_src_x.append(src_x)
  all_src_y.append(src_y)

ax.set_yticks([r + ((len(models)-1)* (height + height_gap)) / 2 for r in range(2)], ['Context Length\n(I-axis)', 'Hidden Size\n(P-axis)'], va='center')

ax.set_xscale('log', base=2)
handles, labels = plt.gca().get_legend_handles_labels()

# 颠倒顺序
handles = handles[::-1]
labels = labels[::-1]

# 重新绘制图例
plt.legend(handles, labels, loc='best')

plt.savefig(PLOT_DIR + 'intro.pdf', bbox_inches='tight')
