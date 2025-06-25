from utils import *
import numpy as np
import matplotlib.patches as mpatches

model_names = ['h2o']
sys_names = ['tensorrt', 'FlashAttention', 'tvm', 'our']

data_a100 = parse_csv(LOG_DIR + "a100_40_kernel_4096.csv", sep='\t')

def plot(data, sys_names, figure_name):
  figsize = {
      "figure.figsize": (3, 4),
      'font.sans-serif': 'Times New Roman',
      'axes.labelsize': 12,
      'font.size':14,
      'legend.fontsize': 8,
      'xtick.labelsize': 12,
      'ytick.labelsize': 12,
      'pdf.fonttype': 42,
      'ps.fonttype': 42
  }
  plt.rcParams.update(figsize)
  fig, ax = plt.subplots(1, 1)

  pair_color_def = COLOR_DEF[:len(sys_names)]
  hatch_def = [HATCH_DEF[2] if i == len(sys_names) - 1 else None for i in range(len(sys_names))]

  data = data.loc['h2o']
  data['FlashAttention'] = -1
  print(data)
  perf_ref = data[sys_names]
  perf = perf_ref.clip(lower=0)
  baseline = perf.loc[sys_names[-1]]

  x_pos = np.arange(len(sys_names))

  bar_width = 0.8
  bar_gap = 0.2
  ylim = 30 # 限制y轴范围, 保持表格整齐
  bars = ax.bar(x_pos, perf, color=pair_color_def, width=bar_width, edgecolor='k', hatch=hatch_def)
  for i, bar in enumerate(bars):
    if perf_ref.loc[sys_names[i]] == -1:
      # 不支持
      ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.005, 'Not Support', ha='center', va='bottom', rotation=90)
      
  min_perf = float('inf')
  for i in perf_ref[:-1]:
    if i > 0:
      min_perf = min(min_perf, i)
  # speedup
  ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.1, bars[-1].get_height(), f'{min_perf / baseline:.1f}' + '\u00D7', fontweight='bold', ha='center', va='bottom', color='red')

  ax.set_xticks(range(len(sys_names)), ["TensorRT", "Flash\nAttention", "TVM", "TA"])
  # fig.text(, 0.5, 'Execution Time (ms)', va='center', rotation='vertical', fontsize=10)
  ax.set_ylabel('Execution Time (ms)')
  fig.savefig(f"{figure_name}.pdf", bbox_inches='tight')


def _plot(data, devices, model_names, sys_names, figure_name, add_legend=False):
  # 两图合并，参数有所修改
  figsize = {
      "figure.figsize": (6, 2),
      'font.sans-serif': 'Times New Roman',
      'axes.labelsize': 12,
      'font.size':12,
      'legend.fontsize': 8,
      'xtick.labelsize': 12,
      'ytick.labelsize': 10,
      'pdf.fonttype': 42,
      'ps.fonttype': 42
  }
  plt.rcParams.update(figsize)
  fig, axs = plt.subplots(1, 1)

  # 和utils.py中的COLOR_DEF相同，共7种颜色
  pair_color_def = COLOR_DEF[:len(sys_names)]
  hatch_def = [HATCH_DEF[2] if i == len(sys_names) - 1 else None for i in range(len(sys_names))]

  # 用ABCDEF替代7个sys_name
  abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  abc = abc[:len(sys_names)]
  bar_width = 0.8
  bar_gap = 0.2
  ylim = 30 # 限制y轴范围, 保持表格整齐

  for row_id, (ax_row, dataset) in enumerate(zip(axs, data)):
    for col_id, (ax, model_name) in enumerate(zip(ax_row, model_names)):
      perf_ref = dataset.loc[model_name][sys_names]

      perf = perf_ref.clip(lower=0)
      # 绝对时间
      baseline = perf.loc[sys_names[-1]]
      norm_perf = perf

      x_pos = np.arange(len(sys_names))
      bars = ax.bar(x_pos, norm_perf, color=pair_color_def, width=bar_width, edgecolor='k', hatch=hatch_def)

      for i, bar in enumerate(bars):
        if perf_ref.loc[sys_names[i]] == 0:
          # OOM
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.05, 'OOM', ha='center', va='bottom', rotation=90)
        elif perf_ref.loc[sys_names[i]] == -1:
          # 不支持
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.05, 'NS', ha='center', va='bottom', rotation=90)
        elif perf_ref.loc[sys_names[i]] == -2:
          # 超时
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.05, 'TLE', ha='center', va='bottom', rotation=90)
        elif norm_perf.loc[sys_names[i]] > ylim: 
          # 截断，文字补充
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.95, f'{norm_perf.loc[sys_names[i]]:.1f}', ha='center', va='top', rotation=90)
      
      min_perf = float('inf')
      for i in perf_ref[:-1]:
        if i > 0:
          min_perf = min(min_perf, i)
      # speedup
      ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.1, bars[-1].get_height(), f'{min_perf / baseline:.1f}' + '\u00D7', fontweight='bold', ha='center', va='bottom', fontsize=7)

      ax.set_xticks(range(len(abc)), abc)
      # 子图标注model_name
      if row_id == 0:
        ax.set_title(MODEL_NAME[model_name], loc='center', fontsize=10)

      if col_id == 0:
        ax.set_ylabel(devices[row_id], fontsize=10)
        ax.yaxis.set_label_coords(-0.5, 0.5)

      max_height = np.nanmax(norm_perf)
      ax.set_ylim(0, ylim)
      ax.set_yticks(range(0, ylim + 1, 10))

  # 添加legend    
  legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label='(' + abc[i] + ') ' + SYS_NAME[sys_names[i]]) for i in range(len(sys_names))]
  fig.legend(handles=legend_handles, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1.15))
  fig.text(0.09, 0.5, 'Execution Time (ms)', va='center', rotation='vertical', fontsize=10)
  plt.subplots_adjust(hspace=0.5, wspace=0.3)
  fig.savefig(PLOT_DIR + f"{figure_name}.pdf", bbox_inches='tight')

plot(data_a100, sys_names, figure_name='cmp')
