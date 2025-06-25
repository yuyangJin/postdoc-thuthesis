from utils import *
import numpy as np

# 数据
models = ['Llama-65B', 'Llama2-70B', 'Llama3.1-405B']
hidden_sizes = [8192, 8192, 16384]
context_lens = [2048, 4096, 131072]

hidden_size_growth = [hidden_sizes[i] / hidden_sizes[i-1] for i in range(1, len(hidden_sizes))]
context_len_growth = [context_lens[i] / context_lens[i-1] for i in range(1, len(context_lens))]

hidden_size_growth = int(hidden_sizes[-1] / hidden_sizes[0])
context_len_growth = int(context_lens[-1] / context_lens[0])

print(hidden_size_growth)
print(context_len_growth)



# 绘制散点图
figsize = {
    "figure.figsize": (2, 3),
    'font.sans-serif': 'Times New Roman',
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
}
plt.rcParams.update(figsize)
fig, ax = plt.subplots(1, 1)
color_def = COLOR_DEF[:len(models)]
hatch_def = HATCH_DEF[:len(models)]


width = 0.16
width_gap = 0.04

ylim = 2 ** 20
r1 = np.arange(2)
all_bars = []
for i, model in enumerate(models):
  bars = ax.bar(r1, [hidden_sizes[i], context_lens[i]], color=color_def[i], width=width, edgecolor='k', hatch=hatch_def[i], label=model)
  r1 = [x + width + width_gap for x in r1]
  all_bars.append(bars)

  # if i == 1:
  #   x0 = bars[0].get_x() + bars[0].get_width() / 2 - 0.1
  #   y0 = (2 ** 14 + 2 ** 13) / 2 
  #   x1 = bars[1].get_x() + bars[1].get_width() / 2 - 0.23
  #   y1 = 2 ** 14
  #   ax.text(x0, y0, f'{hidden_size_growth}\u00D7', ha='center', va='bottom', fontweight='bold', fontsize=20)
  #   ax.text(x1, y1, f'{context_len_growth}\u00D7', ha='center', va='bottom', fontweight='bold', fontsize=20)


# all_dst_x = []
# all_dst_y = []
# all_src_x = []
# all_src_y = []
# for i, bar in enumerate(all_bars[-1]):
#   first_bar = all_bars[0][i]
#   dst_x = bar.get_x() + bar.get_width() / 2
#   dst_y = bar.get_height()

#   src_x = first_bar.get_x() + first_bar.get_width() / 2
#   src_y = first_bar.get_height()
#   print(f"{src_y=}")
#   ax.annotate(
#     '',
#     xy = (dst_x, dst_y),
#     xytext = (src_x, src_y),
#     arrowprops=dict(facecolor='black', shrink=0.05, width=2.5),
#   )
#   all_dst_x.append(dst_x)
#   all_dst_y.append(dst_y)
#   all_src_x.append(src_x)
#   all_src_y.append(src_y)

ax.set_xticks([r + ((len(models)-1)* (width + width_gap)) / 2 for r in range(2)], ['Hidden Size\n(P-axis)', 'Context Length\n(I-axis)'])

ax.set_yscale('log', base=2)
plt.legend(loc='best')

plt.savefig(PLOT_DIR + 'intro.pdf', bbox_inches='tight')