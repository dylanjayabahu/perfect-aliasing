"""Publication figure layout; all measurements come from the consolidated results.

No annotation leaders are used for the directive controls: their categorical
comparison has its own panel. Other repairs move/remove annotation artists and
legends, never experimental curves or observations.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK = '#52514e'
BLUE = '#0353a4'
RED = '#c1121f'
GREEN = '#1baf7a'


def remove_text(ax, prefix):
    for text in list(ax.texts):
        if text.get_text().startswith(prefix):
            text.remove()


def bottom_legend(fig, ax, *, ncol=2, labels=None, size=6.2):
    handles, old_labels = ax.get_legend_handles_labels()
    for legend in list(fig.legends):
        legend.remove()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return fig.legend(handles, labels or old_labels, loc='lower center',
                      bbox_to_anchor=(0.5, 0.005), ncol=ncol,
                      frameon=False, fontsize=size)


def refine(fig, name, rect):
    axes = fig.axes
    for ax in axes:
        for text in ax.texts:
            if text.get_bbox_patch() is None:
                text.set_bbox({'facecolor':'white','edgecolor':'none','pad':0.25})
            text.set_zorder(8)
    if name == 'fig_refit_artifact.png':
        ax = axes[0]
        fig.set_size_inches(5.06, 3.15)
        remove_text(ax, 'same activations, same layer:')
        remove_text(ax, '14 system-prompt variants')
        # The caption identifies the comparison; a ring keeps the exact point
        # visible without a pointer cutting across the plot or legend.
        frozen, refit = (c.get_offsets() for c in ax.collections[:2])
        i = abs(frozen[:, 1] - refit[:, 1]).argmax()
        ax.scatter([refit[i, 0]], [refit[i, 1]], s=100,
                   facecolors='none', edgecolors=INK, linewidths=0.9, zorder=6)
        bottom_legend(fig, ax, ncol=1, labels=[
            'one fixed probe, cross-scored on every variant',
            "probe refit on each variant's own ally data"], size=6.3)
        ax.set_title('')
        return (0, 0.18, 1, 1)
    if name == 'fig_settling.png':
        fig.set_size_inches(5.5, 2.85)
        # Explanation belongs in the caption; the grey span remains visible.
        remove_text(axes[0], 'control at chance:')
        for ax in axes:
            for text in ax.texts:
                if text.get_text().startswith('inverts to'):
                    text.set_color(BLUE)
        axes[0].set_ylabel('truth AUROC on rival trials', fontsize=6.5, color=INK)
        bottom_legend(fig, axes[0], labels=[
            'refit on held-out task', 'frozen: same-task control',
            'frozen: held-out task'], size=6.3)
        return (0, 0.19, 1, 1)
    if name == 'fig_predictors.png':
        fig.set_size_inches(5.5, 2.75)
        for ax in axes:
            # Labels with indistinct diagonal leaders looked like extra series.
            remove_text(ax, 'Mistral-7B')
            remove_text(ax, 'Llama-8B')
            for text in ax.texts:
                if 'cells coincide here' in text.get_text():
                    x, y = text.xy
                    text.set_text(text.get_text().replace('cells coincide here',
                                  f'cells at ({x:.2f}, {y:.2f})'))
        bottom_legend(fig, axes[0], size=5.8)
        return (0, 0.19, 1, 1)
    if name == 'fig_geom.png':
        for text in axes[2].texts:
            if text.get_text() == 'p_withhold':
                text.set_position((-2, -8))
                text.set_ha('right')
                text.set_va('top')
            elif text.get_text() == 'hint':
                text.set_position((-2, 6))
                text.set_ha('right')
        return rect
    if name == 'fig_cross_family.png':
        ax = axes[0]
        ax.set_title('')
        # Direct rate labels beside the points avoid two rows' labels touching.
        for text in ax.texts:
            if text.get_text().replace('.', '').isdigit():
                has_horizontal_spread = text.get_color() == BLUE and text.xy[0] > 0.01
                text.set_position((0, -8) if has_horizontal_spread else
                                  (7, 4 if text.get_color() == RED else -2))
                text.set_ha('center' if has_horizontal_spread else 'left')
                text.set_va('top' if has_horizontal_spread else 'center')
                text.set_bbox({'facecolor':'white','edgecolor':'none','pad':0.3})
        return rect
    if name == 'fig_instrpair.png':
        fig.set_size_inches(5.5, 2.65)
        bottom_legend(fig, axes[0], ncol=3, labels=['directive, row-level split', 'directive, episodes + wordings held out', 'same score vs secret bit (row split)'])
        for text in axes[2].texts:
            if text.get_text().startswith('action $= 1-$truth'):
                # Three compact lines stay inside the triangle above the identity.
                text.set_text(text.get_text().replace('(cell, layer) pairs', 'pairs'))
                text.xy = (0.97, 0.97)
                text.set_position((0.97, 0.97))
                text.set_ha('right')
                text.set_fontsize(5.3)
        axes[2].set_title('rival-label identity\n(ally fit)', fontsize=7.0, fontweight='bold')
        return (0, 0.13, 1, 1)
    if name == 'fig_suppression.png':
        for text in axes[0].texts:
            if text.get_text().startswith('final-layer readout'):
                # The final-layer series is already identified by the legend.
                text.arrow_patch = None
                text.arrowprops = None
        return rect
    if name == 'fig_depth_step.png':
        fig.set_size_inches(4.62, 3.05)
        bottom_legend(fig, axes[0], ncol=1, size=6.2)
        remove_text(axes[0], 'inferred curve sampled every')
        return (0, 0.24, 1, 1)
    if name == 'fig_freeze_transfer.png':
        fig.set_size_inches(5.5, 2.45)
        bottom_legend(fig, axes[0], size=6.2)
        for text in axes[1].texts:
            if text.get_text().startswith('inverts'):
                text.arrow_patch = None
                text.arrowprops = None
        return (0, 0.15, 1, 1)
    if name == 'fig_causal.png':
        for ax in axes[:2]:
            for line in ax.lines:
                if line.get_alpha() == 0.4 and line.get_color() in (BLUE, RED):
                    line.set_alpha(0.85)
            for text in ax.texts:
                if text.get_text().startswith('$\\alpha'):
                    text.set_bbox({'facecolor':'white','edgecolor':'none','pad':0.4})
        # This completion caveat is now in the caption, away from rank labels.
        for text in list(axes[-1].texts):
            if 'evicted' in text.get_text():
                text.remove()
        return rect
    if name == 'fig_depth_sweep.png':
        fig.set_size_inches(5.5, 3.85)
        bottom_legend(fig, axes[0], labels=['mixed-fit', 'ally-fit'])
        axes[-1].set_title('geometry: direction cosine', fontsize=7.0, fontweight='bold')
        return (0, 0.09, 1, 1)
    return rect


def draw_ladder(d2data, ns):
    """Separate the two categorical text controls from the directive ladder.

    Every AUROC and deception-rate label uses the same archived record as before.
    Moving controls into their own axis removes arbitrary x-jitter and leaders.
    """
    d2 = (d2data or {}).get('d2') or {}
    if not d2:
        print('skip fig_d2 (no D2 data)')
        return
    ladder, controls = ns['LADDER'], ns['CONTROLS']
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.75), sharey=True,
                             gridspec_kw={'width_ratios': [1.25, 1.25, 0.8]})
    for ax, fam in zip(axes[:2], ['8b', 'mistral-7b']):
        by = {v['rung']: v for v in d2.values() if v.get('model') == fam}
        xs = [j for j, rung in enumerate(ladder) if rung in by]
        ys = [by[ladder[j]]['auroc'] for j in xs]
        ax.plot(xs, ys, color=RED, linewidth=1.5, marker='o', ms=3.4,
                markeredgecolor='white', markeredgewidth=0.5, zorder=4)
        frozen = [(j, by[ladder[j]].get('auroc_frozen')) for j in xs]
        frozen = [(j, v) for j, v in frozen if v is not None]
        if frozen:
            ax.plot([j for j, v in frozen], [v for j, v in frozen], color=BLUE,
                    linewidth=1.5, marker='o', ms=3.4, markeredgecolor='white',
                    markeredgewidth=0.5, zorder=5)
        for j, y in zip(xs, ys):
            rate = by[ladder[j]].get('decep')
            if rate is not None:
                ax.annotate(f'{rate:.2f}', (j, y), xytext=(-7, 0) if y < 0.22 else (0, -9),
                            textcoords='offset points', ha='right' if y < 0.22 else 'center',
                            va='center' if y < 0.22 else 'top', fontsize=6.0, color=INK,
                            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.2})
        ax.set_xticks(range(5), ladder, rotation=30, ha='right')
        ax.set_xlim(-0.4, 4.4)
        ns['_style'](ax, ns['FAM_LABEL'][fam], 'directive strength')
        ns['_chance'](ax)
        if not frozen:
            ax.text(0.04, 0.17, 'hint / soft and\nfrozen series\nnot run',
                    transform=ax.transAxes, fontsize=6.0, color=INK,
                    va='bottom', bbox={'facecolor':'white','edgecolor':'none','pad':0.3})
    ax = axes[2]
    by = {v['rung']: v for v in d2.values() if v.get('model') == '8b'}
    for j, key in enumerate(controls):
        cell = by[key]
        ax.scatter([j], [cell['auroc']], marker='D', s=32, color=GREEN,
                   edgecolor='white', linewidth=0.7, zorder=5)
        lower = cell['auroc'] < 0.8
        ax.annotate(f"{cell['decep']:.2f}", (j, cell['auroc']), xytext=(0, 7 if lower else -10),
                    textcoords='offset points', ha='center', va='bottom' if lower else 'top',
                    fontsize=6.0, color=INK)
    ax.set_xticks([0, 1], ['A', 'B'])
    ax.set_xlim(-0.55, 1.55)
    ns['_style'](ax, 'Llama-8B\ntext controls', 'control variant')
    ns['_chance'](ax)
    for ax in axes:
        ax.set_ylim(-0.06, 1.10)
    axes[0].set_ylabel('truth AUROC on rival trials', fontsize=6.5, color=INK)
    handles = [Line2D([], [], color=RED, marker='o', ms=3, label='refit per variant'),
               Line2D([], [], color=BLUE, marker='o', ms=3, label='one frozen probe'),
               Line2D([], [], color=GREEN, marker='D', ms=4, linestyle='none',
                      label='text control (refit)')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.005),
               frameon=False, ncol=3, fontsize=6.0)
    ns['_save'](fig, 'fig_d2_ladder.png', rect=(0, 0.15, 1, 1))
