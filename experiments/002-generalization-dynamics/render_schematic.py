"""Render the identification schematic (no data) as fig_schematic.png, beside the other paper figures.

Two features of a trial: its true bit (x) and the answer the task prescribes (y). Ally trials lie where the
two agree, rival trials where they differ. A boundary along either feature separates the ally trials
perfectly; on rival trials the two boundaries rank in opposite orders. Sized for a 0.40\\textwidth minipage
at 1:1, with every text element at 7pt or more.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
HERE = Path(__file__).resolve().parent
FIG = HERE.parent.parent / 'paper' / 'figures' if os.environ.get('PAPER') == '1' else HERE / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
OUT = FIG / 'fig_schematic.png'
INK, MUTED = '#2b2b2b', '#6b6b6b'
TRUTH, ACTION = '#0353a4', '#c1121f'          # same blue/red as the paper's mixed-fit / ally-fit curves
W = 0.40 * 5.5

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})
fig, ax = plt.subplots(figsize=(W, 1.55))
from matplotlib.lines import Line2D
ax.scatter([0, 1], [0, 1], s=105, zorder=5, color=INK, linewidths=1.6, edgecolors=INK)
ax.scatter([0, 1], [1, 0], s=105, zorder=5, facecolors='white', edgecolors=INK, linewidths=1.6)
ax.plot([0.5, 0.5], [-0.62, 1.3], color=TRUTH, lw=1.8, ls='--', zorder=2)
ax.axhline(0.5, color=ACTION, lw=1.8, ls='--', zorder=2)
ax.text(0.46, -0.58, 'truth\nboundary', color=TRUTH, fontsize=7.5, fontweight='bold', va='bottom', ha='right')
ax.text(-0.8, 0.56, 'action\nboundary', color=ACTION, fontsize=7.5, fontweight='bold', va='bottom', ha='left')
ax.set_xlim(-0.82, 1.4)
ax.set_ylim(-0.62, 1.85)
ax.set_xlabel('true bit', fontsize=7.5, color=INK, labelpad=1)
ax.set_ylabel('prescribed answer', fontsize=7.5, color=INK, labelpad=1)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.tick_params(labelsize=7, colors=MUTED, length=2)
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
handles = [Line2D([], [], marker='o', ls='', ms=7, color=INK, label='ally trial (truth = action)'),
           Line2D([], [], marker='o', ls='', ms=7, mfc='white', mec=INK, mew=1.4, label='rival trial (they differ)')]
ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.03, 1.08), fontsize=7, frameon=False,
          handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25)
fig.tight_layout(pad=0.2)
fig.savefig(OUT, dpi=300, facecolor='white')
print('wrote', OUT, fig.get_size_inches())
