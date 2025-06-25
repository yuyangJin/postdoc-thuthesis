from utils import *
import numpy as np
import matplotlib.patches as mpatches

seqlens = [1024, 2048, 4096, 8192]
sys_names = ['torch', 'dynamo', 'tensorrt', 'our']

gflops = parse_csv(LOG_DIR + "gflops.csv", sep='\t')
tflops = gflops / 1000
mem = parse_csv(LOG_DIR + "mem.csv", sep='\t')

def plot(dataset, seqlens, sys_names, figure_name, add_legend=False):
  # 两图合并，参数有所修改
  figsize = {
      "figure.figsize": (6, 1),
      'font.sans-serif': 'Times New Roman',
      'axes.labelsize': 12,
      'font.size': 10,
      'legend.fontsize': 10,
      'xtick.labelsize': 10,
      'ytick.labelsize': 10,
      'pdf.fonttype': 42,
      'ps.fonttype': 42
  }
  plt.rcParams.update(figsize)
  fig, (ax1, ax2) = plt.subplots(1,2)

  # 和utils.py中的COLOR_DEF相同，共7种颜色
  pair_color_def = COLOR_DEF[:len(sys_names)]
  hatch_def = [HATCH_DEF[2] if i == len(sys_names) - 1 else None for i in range(len(sys_names))]

  # 设置bar width gap
  width = 0.16
  width_gap = 0.04

  # 绘制第一个图
  ylim1 = 125 # 限制y轴范围, 保持表格整齐，可根据数据手动调整
  r1 = np.arange(len(seqlens))
  for i, sys_name in enumerate(sys_names):
    data_ref = dataset[0].loc[seqlens][sys_name]
    bars = ax1.bar(r1, data_ref, hatch=hatch_def[i], color=pair_color_def[i], width=width, label=SYS_NAME[sys_name], edgecolor='k')
    for i, bar in enumerate(bars):
      if data_ref[seqlens[i]] > ylim1: 
        # 截断，文字补充
        ax1.text(bar.get_x() + bar.get_width() / 2, ylim1 * 0.95, f'{data_ref[seqlens[i]]:.2f}', ha='center', va='top', rotation=90)
    r1 = [x + width + width_gap for x in r1]
  # 设置x轴刻度，标签，标题
  ax1.set_xticks([r + ((len(sys_names)-1) * (width + width_gap)) / 2 for r in range(len(seqlens))], seqlens)
  ax1.set_xlabel('Sequence Length', fontsize=10)
  max_height = max([max(dataset[0].loc[seqlens][sys_name]) for sys_name in sys_names])
  ax1.set_ylim(0, ylim1)
  ax1.set_ylabel('TFLOP/s', fontsize=10)
  ax1.set_title("Core Module TFLOP/s",loc='center')
  
  # 绘制第二个图
  r2 = np.arange(len(seqlens))
  ylim2 = 10 ** 5 # 限制y轴范围, 保持表格整齐，可根据数据手动调整
  for i, sys_name in enumerate(sys_names):
    data_ref = dataset[1].loc[seqlens][sys_name]
    bars = ax2.bar(r2, data_ref, hatch=hatch_def[i], color=pair_color_def[i], width=width, label=SYS_NAME[sys_name], edgecolor='k')
    for i, bar in enumerate(bars):
      if data_ref[seqlens[i]] > ylim2: 
        # 截断，文字补充
        ax2.text(bar.get_x() + bar.get_width() / 2, ylim2 * 0.95, f'{data_ref[seqlens[i]]:.2f}', ha='center', va='top', rotation=90)
    r2 = [x + width + width_gap for x in r2]
  ax2.set_xticks([r + ((len(sys_names)-1) * (width + width_gap)) / 2 for r in range(len(seqlens))], seqlens)
  ax2.set_xlabel('Sequence Length', fontsize=10)
  max_height = max([max(dataset[1].loc[seqlens][sys_name]) for sys_name in sys_names])
  ax2.set_yscale('log', base=10)
  ax2.set_ylim(100, ylim2)
  ax2.set_ylabel('Memory Usage (MiB)', fontsize=10)
  ax2.set_title("Core Module Memory Usage", loc='center')

  if add_legend:
    legend = fig.legend(*ax1.get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.45))

  # 调整子图之间的间距
  plt.subplots_adjust(wspace=0.3)

  fig.savefig(PLOT_DIR + f"{figure_name}.pdf", bbox_inches='tight')

plot([tflops, mem], seqlens, sys_names, figure_name='gflops_mem', add_legend=True)