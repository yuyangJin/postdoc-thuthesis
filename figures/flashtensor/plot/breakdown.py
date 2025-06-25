from utils import *
import numpy as np
import matplotlib.patches as mpatches


model_names = ['h2o', 'roco', 'keyformer', 'snapkv', 'corm', 'attn', 'gemma2']
opt_names = {
  'naive': 'PyTorch',
  '+fission': '+Fission',
  '+transform': '+B.T.',
  '+mapping': '+K.M. w/o Fusion',
  '+fusion': '+Fusion',
  '+value': '+V.T.',
}

breakdown = parse_csv(LOG_DIR + "breakdown_h100.csv", sep='\t')

def plot(dataset, model_names, opt_names, figure_name):
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
  fig, ax = plt.subplots(1, 1)

  pair_color_def = COLOR_DEF[:len(opt_names)]

  # 算speedup
  print(breakdown)
  # speedup = breakdown.div(breakdown['naive'], axis=0)
  speedup = breakdown['naive'].div(breakdown)
  speedup = breakdown.copy()
  for col in breakdown.columns:
    speedup[col] = breakdown['naive'] / breakdown[col]
  print(speedup)

  width = 0.13
  width_gap = 0.03
  ylim = 16

  # r = np.arange(len(model_names)) - len(opt_names) * width / 2
  r = np.arange(len(model_names))
  for i, opt_name in enumerate(opt_names):
    data_ref = speedup.loc[model_names][opt_name]
    bars = ax.bar(r, data_ref, hatch=None, color=pair_color_def[i], width=width, label=opt_names[opt_name], edgecolor='k')
    for i, bar in enumerate(bars):
      if data_ref[model_names][i] > ylim: 
        # 截断，文字补充
        ax.text(bar.get_x() + bar.get_width() / 2 + 0.01, ylim * 0.95, f'{data_ref[model_names[i]]:.2f}', ha='center', va='top', rotation=90, fontsize=7)

    r = [x + width + width_gap for x in r]
  

  MODEL_NAME['attn'] = 'V.A.'
  ax.set_xticks([r + ((len(opt_names)-1) * (width + width_gap)) / 2 for r in range(len(model_names))], [MODEL_NAME[m] for m in model_names])

  # ax.set_xlim(-0.2, len(r))
  ax.set_xmargin(0.02)

  ax.set_yscale('log', base=2)
  ax.set_ylim(top=ylim)
  ax.set_yticks([2 ** i for i in range(0, int(math.log2(ylim)) + 1)])

  ax.set_ylabel('Speedup', fontsize=10)

  ax.axhline(y=1, color='red', linestyle='--', linewidth=1)

  fig.legend(*ax.get_legend_handles_labels(), handletextpad=0.4, columnspacing=0.4, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.3))
  # fig.savefig(f"{figure_name}.pdf", bbox_inches='tight')
  fig.savefig(PLOT_DIR + f"{figure_name}.pdf", bbox_inches='tight')
 

plot(breakdown, model_names, opt_names, figure_name='breakdown')