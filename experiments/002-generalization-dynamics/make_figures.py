"""Figures for Experiment 002 — the *corrective* paper (probe identification).

Built from the consolidated dumps in ``data/`` emitted by the analysis jobs:
  * ``e2_consolidated.json`` — grid_e2b (42 cells) · identification · dynamics · legacy_e2
  * ``d2_figuredata.json``   — the D2 within-model directive ladder (optional)

Every figure is skipped-with-a-note when its slice is missing, so this is safe to run against a partial
dump.  Regenerate anytime with:  python make_figures.py

Design follows the `dataviz` skill. Palette validated with `scripts/validate_palette.js --pairs all`
(light, categorical): ally-trained #c1121f · mixed-trained #0353a4 · third series #1baf7a — all checks
PASS; worst all-pairs CVD ΔE 15.2 (deutan). The aqua carries a contrast WARN (2.74:1), which the skill
says obligates relief, so **every series is direct-labeled** in addition to the legend. One measure per
axis, no dual-axis anywhere, thin 2px marks, recessive grid, and direct labels only on the numbers that
carry the story.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.legend as mlegend
import matplotlib.figure as mfigure
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# ``PAPER=1 python make_figures.py`` renders every figure at the *final printed width* (a 5.5in
# single-column text block) instead of at screen size.  This matters: the screen-size figures are
# 8-16in wide, so \includegraphics[width=\textwidth] scales them by 0.4-0.7x and the 8.5-12pt labels
# land at 4-7pt in the PDF.  Rendering at the printed width means every fontsize below is the size it
# actually prints at.  The screen versions in figures/ are never clobbered.
#
# PAPER mode writes STRAIGHT INTO the paper's own figure directory (paper/figures) rather than to
# a local dir that then gets copied.  A copy step is a stale-artifact bug waiting to happen: the build
# would keep succeeding against last week's PNGs and report a page count for a figure nobody rendered.
PAPER = os.environ.get("PAPER") == "1"
TEXTWIDTH = 5.5            # NeurIPS \textwidth, inches
if PAPER:
    FIG = HERE.parent.parent / "paper" / "figures"
    FIG.mkdir(parents=True, exist_ok=True)

# Fixed categorical hues, assigned in fixed order and never cycled.
# The semantic pairing is deliberate and consistent with Exp-001: WARM = the conventional/broken thing,
# COOL = the corrected thing.
ALLY = "#c1121f"        # probe trained on ALLY contexts only — the conventional, unidentified protocol
MIXED = "#0353a4"       # probe trained on MIXED ally+rival contexts — the identified protocol (the fix)
THIRD = "#1baf7a"       # mid-stack / control series (always direct-labeled: contrast WARN relief)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID = dict(alpha=0.25, linewidth=0.6)


def load(name):
    p = DATA / name
    return json.loads(p.read_text()) if p.is_file() else {}


def _style(ax, title=None, xlabel=None, ylabel=None):
    ts, ls, tk = (7.5, 6.5, 5.8) if PAPER else (11, 9.5, 8.5)
    if title:
        ax.set_title(title, fontsize=ts, fontweight="bold", pad=4 if PAPER else 8, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=ls, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ls, color=INK2)
    ax.grid(True, color=MUTED, **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(labelsize=tk, colors=INK2, length=3)


LEG_FS = 6.5 if PAPER else 9.5      # legend text
NOTE_FS = 5.8 if PAPER else 8.5     # in-axes annotations


def _size(w, h, frac=1.0):
    """Screen figsize -> printed figsize. In PAPER mode the figure is rendered ``frac * TEXTWIDTH``
    inches wide with its aspect preserved, so \\includegraphics needs no scaling and every fontsize in
    this module is the size that actually prints."""
    return (w, h) if not PAPER else (TEXTWIDTH * frac, h * TEXTWIDTH * frac / w)


def _suptitle(fig, text, **kw):
    """No-op in PAPER mode: the LaTeX \\caption states the finding, and a baked-in title duplicates it."""
    if not PAPER:
        fig.suptitle(text, color=INK, fontweight="bold", **kw)


# Every figure that prints too small has the same cause, so the check lives here rather than in review.
# bbox_inches="tight" GROWS the canvas to contain any artist that sticks out past the figure edge (a
# legend anchored below, an annotation running off the right).  The PNG is then wider than TEXTWIDTH,
# \includegraphics scales the whole raster back down, and every fontsize in this module prints smaller
# than it says -- silently, because the figure still "looks fine" on screen.  Five figures shipped that
# way.  So: flag any figure whose saved width exceeds the width we asked for.  Cropping (a narrower
# result) is fine and expected; growth is the bug.
_DPI = 200
_OVERFLOW = []
_DASHES = []


def _check_dashes(fig, name):
    """The paper is zero-em-dash (user decision 2026-08-19), and figure text is typeset in the paper
    just like prose is. Grepping this module for the character does not work: most hits are comments,
    and a rendered string can be built by an f-string. So inspect the actual Text artists instead --
    that catches titles, annotations, tick labels and legend entries however they were set."""
    bad = set()
    for t in fig.findobj(mtext.Text):
        s = t.get_text()
        if "\u2014" in s or "\u2013" in s:
            bad.add(s.replace("\n", " ")[:78])
    if bad:
        _DASHES.append((name, sorted(bad)))
    return bad


def _blame(fig):
    """Name the artists whose ink lies outside the figure rectangle. bbox_inches="tight" silently grows
    the canvas to contain them, so 'which artist overflowed' is the only question worth asking, and
    guessing wastes a render cycle each time."""
    out = []
    W, H = fig.get_size_inches()
    dpi = fig.get_dpi()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)                      # _blame runs after savefig, so swapping the canvas is safe
    fig.draw_without_rendering()              # legend children have no usable extent until a draw happens
    r = fig.canvas.get_renderer()
    for a in fig.findobj():
        if not a.get_visible() or a is fig or isinstance(a, mfigure.Figure):
            continue
        if not isinstance(a, (mtext.Text, mlegend.Legend)):
            continue
        try:
            bb = a.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width <= 0:
            continue
        # Work in FIGURE-RELATIVE coords: the renderer's dpi need not match fig.dpi, and dividing by
        # the wrong one inflated every number here by 2x on the first attempt.
        f = bb.transformed(fig.transFigure.inverted())
        over = max(0.0, -f.x0, f.x1 - 1.0) * W
        if over > 0.02:
            what = a.get_text()[:44].replace("\n", " ") if isinstance(a, mtext.Text) else "<legend>"
            if what.strip() or isinstance(a, mlegend.Legend):
                out.append(f"{type(a).__name__} {what!r} by {over:.2f}in")
    return sorted(set(out))[:6]


def _save(fig, name, rect=None):
    # `rect` reserves room for a legend that lives BELOW the axes. Without it this bare tight_layout()
    # silently undid every caller's own tight_layout(rect=...) call, the axes re-expanded to full height,
    # and the legend ended up outside the figure -- which is what grew four canvases and shrank their
    # fonts. Callers with a below-axes legend must pass rect and anchor the legend at y >= 0.
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    _check_dashes(fig, name)
    want_in = fig.get_size_inches()[0]
    fig.savefig(FIG / name, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    import struct
    with open(FIG / name, "rb") as f:
        f.read(16)
        got_px = struct.unpack(">II", f.read(8))[0]
    got_in = got_px / _DPI
    if PAPER and got_in > want_in * 1.02:
        _OVERFLOW.append((name, want_in, got_in, _blame(fig)))
        print(f"wrote figures/{name}   *** OVERFLOW: asked {want_in:.2f}in, saved {got_in:.2f}in "
              f"-> LaTeX will shrink every font by {want_in / got_in:.0%} ***")
        for b in _blame(fig):
            print(f"        sticks out: {b}")
    else:
        print(f"wrote figures/{name}   ({got_in:.2f}in)")


def _chance(ax):
    """The 0.5 chance line. AUROC has a meaningful zero AND a meaningful midpoint, so mark the midpoint —
    below it the probe is anti-correlated (inverted), which is the whole subject of this paper."""
    ax.axhline(0.5, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)


# --- F1 (spine): the identification result, per layer ----------------------------------------------
# THE headline figure. Same activations, same layer, two probes that differ ONLY in which contexts they
# were fit on. The ally-trained probe collapses/inverts; the mixed-trained probe reads 1.000 everywhere.
CELL_TITLES = {
    # cells trained on the plain game, evaluated on the codebook task (deception does not fully transfer)
    "e2id_gemma-9b_in": "Gemma-2-9B · instructed",
    "e2id_gemma-9b_em": "Gemma-2-9B · emergent (RL)",
    "e2id_qwen-14b_in": "Qwen2.5-14B · instructed",
    "e2id_qwen-14b_em": "Qwen2.5-14B · emergent (RL)",
    "e2id_8b_em": "Llama-3.1-8B · emergent (RL)",
    "e2id_8b_in": "Llama-3.1-8B · instructed",
    "idsmoke_8b_em": "Llama-3.1-8B · emergent (N=120 smoke)",
    # cells RL-trained DIRECTLY on the codebook task — the saturated-emergent cells that close the loop
    # kept short: long titles collide across a 3-column grid
    "cbid_gemma-9b_em": "Gemma-9B · RL on codebook",
    "cbid_8b_hi_em": "Llama-8B · RL on codebook",
    "cbid_gemma-9b_ent_em": "Gemma-9B · RL on codebook (+ent. bonus)",
    "cbid_8b_ent_em": "Llama-8B · RL on codebook (+ent. bonus)",
    "cbid_8b_em": "Llama-8B · RL on codebook, unsaturated",
    "cbid_mistral-7b_em": "Mistral-7B · RL on codebook, unsaturated",
    "cbid_gemma-9b_em_s1": "Gemma-9B · RL on codebook (seed 1)",
    "cbid_gemma-9b_em_s2": "Gemma-9B · RL on codebook (seed 2)",
    "infid_gemma-9b_in": "Gemma-9B · instructed, INFERRED truth",
}
# Shorter titles for PAPER mode: at a 1.8in panel width the screen titles wrap or collide.
PAPER_TITLES = {
    "e2id_gemma-9b_in": "Gemma-9B · instr.",
    "e2id_gemma-9b_em": "Gemma-9B · RL",
    "e2id_qwen-14b_in": "Qwen-14B · instr.",
    "e2id_qwen-14b_em": "Qwen-14B · RL",
    "e2id_8b_em": "Llama-8B · RL",
    "e2id_8b_in": "Llama-8B · instr.",
    "cbid_gemma-9b_em": "Gemma-9B · RL on codebook",
    "cbid_8b_hi_em": "Llama-8B · RL on codebook",
    "cbid_8b_em": "Llama-8B · codebook, unsat.",
    "cbid_mistral-7b_em": "Mistral-7B · codebook, unsat.",
}
# Order: the saturated codebook-trained EMERGENT cells first — those are what the claim now rests on —
# then the saturated instructed cells, then the unsaturated cells that act as within-model controls.
CELL_ORDER = ["cbid_gemma-9b_em", "cbid_8b_hi_em", "cbid_8b_em", "cbid_mistral-7b_em",
              "e2id_gemma-9b_in", "e2id_qwen-14b_in", "e2id_gemma-9b_em", "e2id_qwen-14b_em",
              "e2id_8b_em", "e2id_8b_in"]
# Keep the spine figure to a readable grid; the rest belong in an appendix figure. Raised 6 -> 8 on
# 2026-08-16 so the FOURTH ARCHITECTURE (mistral) the prose claims cannot be silently dropped — at 6 it
# was, and the printed "dropped:" note was the only trace.
CELL_MAX = 8
# Cells that must never appear in the spine figure: seed replicates (reported as spread in the text, not
# as extra panels) and any cell on a different task or truncated by eviction.
CELL_EXCLUDE = {"idsmoke_8b_em", "cbid_gemma-9b_em_s1", "cbid_gemma-9b_em_s2"}


NOTE_TOP = 1.06          # top of the shared y-range: the mixed probe's 1.000 ceiling plus headroom
NOTE_PAD = 0.015         # gutter floor -> note bottom, in axes fractions
NOTE_GAP = 0.02          # note top -> the y=0 data floor
# Both are deliberately small: the measured height is a Text *bbox*, which already carries the font's
# full ascent and descent, so it over-states the ink by a point or so. Every extra point of gutter is
# paid for by compressing the plotted range, which is what crowds the mid-panel labels against the
# chance line, so do not pad this "just to be safe".


def _note_gutter(axes, notes, ytop=NOTE_TOP, pad=NOTE_PAD, gap=NOTE_GAP):
    """Sink the in-panel notes into a gutter BELOW the data floor and set the shared y-range to suit.

    Placement history, because this collision came back once already.  The note used to sit at axes
    (0.02, 0.04), i.e. a hair ABOVE data y = 0 -- which is exactly where the ally-fit probe collapses.
    The note is ~0.6 of a panel wide, the red curve reaches the floor between 0.52 and 0.68 of the way
    across, so the descending line ran straight through the text and struck it out (Gemma-9B RL on
    codebook, Llama-8B RL on codebook, Gemma-9B instr., Qwen-14B instr.).

    Lifting the note off the floor is the opposing failure mode, and it is the one the previous fix
    walked into: y ~ 0.5 carries the dashed chance line, and the mid-to-right of every panel carries the
    bold blue and red final-layer value labels.  The panels also disagree about where the red curve
    *ends* (0.000, 0.056, 0.165, 0.518, 0.815, 0.969), so no single y inside [0, 1] is free in all eight
    -- any fixed interior position trades one collision for another, which is how this recurred.

    So stop hunting for a gap inside the data range and reserve one outside it.  Both series are AUROCs,
    bounded in [0, 1] by construction, so a band strictly below y = 0 cannot be reached by any curve --
    not in these panels, and not in a cell added later.  That is stronger than adapting to where each
    curve happens to go: there is nothing left for the placement to depend on, so nothing left to
    re-tune when the dump changes.  The gutter is measured from the note's own rendered height (a draw
    is required: a Text has no extent before one) so it is exactly as deep as the text needs and no
    deeper -- vertical range is page budget in a body figure, and these curves are read by shape.

    Must run AFTER the figure's tight_layout, since the note height is taken as a fraction of the
    laid-out panel height.  ``_save``'s own tight_layout only makes the panels taller, which makes the
    gutter marginally generous -- the safe direction.
    """
    f0 = pad + gap
    if notes:
        notes[0][0].figure.draw_without_rendering()
        r = notes[0][0].figure.canvas.get_renderer()
        # Ratio of two extents from the SAME renderer, so the renderer-vs-figure dpi mismatch cancels
        # (measuring in absolute pixels and dividing by fig.dpi is the bug _blame documents).
        f0 += max(t.get_window_extent(renderer=r).height / ax.get_window_extent(renderer=r).height
                  for ax, t in notes)
    ybot = -f0 * ytop / (1.0 - f0)          # the y-range whose data-0 line sits at axes fraction f0
    for ax in axes:
        ax.set_ylim(ybot, ytop)
        # Pin the tick set: the gutter widens the range enough that the auto-locator can offer a fourth
        # tick below zero, which would label a band that holds no data.
        ax.set_yticks([0.0, 0.5, 1.0])
    for _, t in notes:
        t.set_position((0.02, pad))
        t.set_va("bottom")


def fig_identification(fd):
    ident = fd.get("identification") or {}
    def _plottable(k):
        c = ident.get(k) or {}
        # never plot a truncated sweep or a non-codebook task in the spine as if it were complete
        return not c.get("partial") and c.get("task", "codebook") == "codebook" and k not in CELL_EXCLUDE
    _skipped = [c for c in CELL_ORDER if c in ident and not _plottable(c)]
    if _skipped:
        print(f"  note: excluded from spine (partial / other task): {', '.join(_skipped)}")
    cells = ([c for c in CELL_ORDER if c in ident and _plottable(c)]
             or [c for c in ident if _plottable(c)])
    if not cells:
        print("skip fig_identification (no identification cells)")
        return
    if len(cells) > CELL_MAX:
        print(f"  note: {len(cells)} identification cells available, showing the first {CELL_MAX} "
              f"(dropped: {', '.join(cells[CELL_MAX:])})")
        cells = cells[:CELL_MAX]
    n = len(cells)
    # 4 columns would save a whole row of height, but at 1.375in per panel the arm titles are wider
    # than the panel and the canvas overflows to 5.81in (the guard in _save caught it), which shrinks
    # every font by 5%. Stay at 3 columns and take the height out of the row instead: this is a BODY
    # figure, so 0.5in of height is ~45 words of page budget.
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    figsize = ((TEXTWIDTH, 1.02 * nrow + 0.45) if PAPER else (4.1 * ncol, 3.3 * nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    used, notes = [], []
    for i, key in enumerate(cells):
        ax = axes[i // ncol][i % ncol]
        used.append(ax)
        cell = ident[key]
        curve = cell.get("curve") or []
        xs = [r["l"] for r in curve]
        ally = [r.get("truth_ally") for r in curve]
        mixed = [r.get("truth_mixed") for r in curve]
        _chance(ax)
        ax.plot(xs, mixed, color=MIXED, linewidth=2.0, zorder=4,
                label="truth probe, fit on ally+rival (identified)")
        ax.plot(xs, ally, color=ALLY, linewidth=2.0, zorder=3,
                label="truth probe, fit on ally only (conventional)")
        decep = ((cell.get("behavior") or {}).get("rival_deception_rate"))
        sub = f"rival deception {decep:.3f}" if isinstance(decep, (int, float)) else ""
        title = (PAPER_TITLES.get(key) or CELL_TITLES.get(key, key)) if PAPER \
            else CELL_TITLES.get(key, key)
        # in PAPER mode only the outer edges carry axis labels — repeating them 8x eats the panel.
        # "bottom" means bottom *of its own column*: the last row is usually short, so keying on
        # nrow - 1 alone leaves the columns above the gap unlabelled.
        edge_x = (i + ncol >= len(cells)) or not PAPER
        # ONE shared y-label below (fig.supylabel). Per-row labels are taller than a 1.02in panel, so
        # at this row height the label for each row printed over the one above it.
        edge_y = (i % ncol == 0) and not PAPER
        _style(ax, title, "layer" if edge_x else None,
               "truth AUROC on rival trials" if edge_y else None)
        # provisional range; _note_gutter re-sets it once the note height is measurable (see there)
        ax.set_ylim(-0.04, NOTE_TOP)
        if sub:
            notes.append((ax, ax.text(0.02, 0.04, sub, transform=ax.transAxes,
                                      fontsize=5.5 if PAPER else 8.5, color=INK2)))
        # direct labels on the final-layer values — the two numbers that carry the story
        if xs and ally and mixed and ally[-1] is not None and mixed[-1] is not None:
            afs = 6.0 if PAPER else 9
            # mixed sits at ~1.000, i.e. against the top spine, so label it *below* its own line —
            # above it would collide with the panel title. Push the ally label clear when the two
            # final values are close enough to overprint.
            ax.annotate(f"{mixed[-1]:.3f}", (xs[-1], mixed[-1]), textcoords="offset points",
                        xytext=(-4, -9), ha="right", va="top", fontsize=afs,
                        fontweight="bold", color=MIXED)
            # Put the ally label BELOW its own line as well, so the two can never converge on the same
            # band; flip it above only when its line is close enough to the floor that below would land
            # on the axis. The old rule keyed on the GAP between the values, which put both labels in
            # mid-panel in the one cell where the ally curve ends mid-range.
            # Two independent hazards, and fixing one alone reintroduces the other: (a) the two curves
            # end mid-panel at different heights, so a gap-keyed rule puts both labels in the middle
            # band; (b) the two curves end at nearly the SAME height (Mistral: 1.000 vs 0.969), so a
            # same-side rule overprints them. Handle both: ally goes below its own line, pushed clear
            # when the gap is small, and above only when its line is near the floor.
            gap = abs(mixed[-1] - ally[-1])
            floor = ally[-1] < 0.18
            # In the floor case, do not offset a fixed number of POINTS upward off the line. A point
            # offset is measured from the curve, so the label's distance to the dashed chance line is
            # whatever the y-range happens to leave over, and the gutter below tightens that range: at
            # ally = 0.165 the old +6pt put the label ON the 0.5 line and on the blue label. Anchor it
            # at the MIDPOINT of the gap it has to live in instead -- a midpoint cannot reach either
            # edge of the gap it bisects, whatever the range does, and for a collapsed line (0.000) it
            # lands within a point of where the fixed offset used to put it.
            ax.annotate(f"{ally[-1]:.3f}",
                        (xs[-1], 0.5 * (ally[-1] + 0.5) if floor else ally[-1]),
                        textcoords="offset points",
                        xytext=(-4, 0 if floor else (-22 if gap < 0.20 else -9)), ha="right",
                        va="center" if floor else "top", fontsize=afs,
                        fontweight="bold", color=ALLY)
    for j in range(len(cells), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    if PAPER:
        fig.supylabel("truth AUROC on rival trials", fontsize=6.5, color=INK2)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               fontsize=6.5 if PAPER else 9.5, bbox_to_anchor=(0.5, -0.02))
    if not PAPER:
        # in the paper the equivalent sentence is the LaTeX caption, so the baked-in title would double it
        fig.suptitle("Same activations, same layer: the ally-trained “truth” probe inverts while an "
                     "identified probe reads 1.000",
                     fontsize=12.5, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.045, 1, 1.0 if PAPER else 0.98))
    _note_gutter(used, notes)
    _save(fig, "fig_identification.png")


# --- F2: the forced identity action/ally == 1 - truth/ally ------------------------------------------
def fig_action_identity(fd):
    ident = fd.get("identification") or {}
    cells = [c for c in CELL_ORDER if c in ident and
             any(r.get("action_ally") is not None for r in (ident[c].get("curve") or []))]
    if not cells:
        print("skip fig_action_identity (no action_ally in dump — extend the fetch job)")
        return
    nc = min(2, len(cells))
    # main.tex includes this at 0.86\textwidth, so render at 0.86 * TEXTWIDTH rather than full
    # width. Rendering wider and letting \includegraphics scale down is the same font-shrinking
    # bug _save guards, arriving by a route _save cannot see: the canvas never overflows, the
    # PNG is simply wider than the space it is given, and every fontsize printed at 88%.
    fig, axes = plt.subplots(1, nc, figsize=_size(4.6 * nc, 3.5, 0.86), squeeze=False)
    for i, key in enumerate(cells[:2]):
        ax = axes[0][i]
        curve = ident[key].get("curve") or []
        xs = [r["l"] for r in curve]
        act = [r.get("action_ally") for r in curve]
        derived = [None if r.get("truth_ally") is None else 1 - r["truth_ally"] for r in curve]
        ax.plot(xs, derived, color=MUTED, linewidth=3.4, alpha=0.55, zorder=2,
                label="1 − (truth probe, ally-fit)")
        ax.plot(xs, act, color=ALLY, linewidth=1.8, linestyle=(0, (3, 2)), zorder=3,
                label="action probe, ally-fit (measured)")
        _style(ax, (PAPER_TITLES.get(key) if PAPER else None) or CELL_TITLES.get(key, key),
               "layer", "AUROC on rival trials" if (i == 0 or not PAPER) else None)
        ax.set_ylim(-0.04, 1.06)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=LEG_FS,
               bbox_to_anchor=(0.5, -0.04))
    _suptitle(fig, "On ally data the truth and action labels are identical, so the “truth” probe and the\n"
                   "action probe are one fitted probe scored against opposite labels", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1.0 if PAPER else 0.94))
    _save(fig, "fig_action_identity.png")


# --- F3: suppression, not belief drift --------------------------------------------------------------
def fig_suppression(fd):
    dyn = fd.get("dynamics") or {}
    series = {k: v for k, v in dyn.items() if v}
    if not series:
        print("skip fig_suppression (no dynamics)")
        return
    key = "e2dyn2_8b_s0" if "e2dyn2_8b_s0" in series else sorted(series)[-1]
    rows = sorted(series[key], key=lambda r: r["epoch"])
    # x = EPOCH, not deception rate. Deception is non-monotone along the trajectory and saturates by
    # ~epoch 70, so plotting against it crushes the entire story into the right-hand 10% of the axis.
    xs = [r["epoch"] for r in rows]
    last = [r.get("last_auroc") for r in rows]
    peak = [r.get("peak_rival") for r in rows]
    decep = [r.get("decep") for r in rows]

    fig, (ax, axb) = plt.subplots(2, 1, figsize=_size(8.0, 5.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.4, 1], hspace=0.12))
    _chance(ax)
    ax.plot(xs, peak, color=THIRD, linewidth=2.0, marker="o", markersize=4.0, zorder=4,
            label="mid-stack peak layer")
    ax.plot(xs, last, color=ALLY, linewidth=2.0, marker="o", markersize=4.0, zorder=3,
            label="final layer (answer token)")
    # park this label in the flat right-hand region, well clear of the red curve's descent
    ax.annotate("truth stays fully decodable\nat every checkpoint",
                (xs[int(len(xs) * 0.78)], 1.0), textcoords="offset points", xytext=(0, -30),
                ha="center", fontsize=NOTE_FS, fontweight="bold", color=THIRD)
    # anchor the collapse label to where the curve actually falls, not to the final flat tail
    fall = next((i for i, v in enumerate(last) if v is not None and v < 0.5), len(xs) - 1)
    ax.annotate("final-layer readout\ncollapses and inverts", (xs[fall], last[fall]),
                textcoords="offset points", xytext=(26, 30), fontsize=NOTE_FS, fontweight="bold",
                color=ALLY, arrowprops=dict(arrowstyle="-", color=ALLY, linewidth=1.0))
    _style(ax, "Suppression, not belief drift: one RL trajectory, 41 checkpoints",
           None, "truth AUROC on rival trials")
    ax.set_ylim(-0.04, 1.10)
    ax.legend(frameon=False, fontsize=LEG_FS, loc="center left", bbox_to_anchor=(0.02, 0.26))

    # Companion panel: the behaviour, on its own axis (never a second y-scale on the same plot).
    axb.plot(xs, decep, color=INK2, linewidth=1.8, zorder=3)
    axb.fill_between(xs, 0, decep, color=INK2, alpha=0.10, zorder=2)
    _style(axb, None, "training epoch", "rival deception")
    axb.set_ylim(-0.03, 1.08)
    _save(fig, "fig_suppression.png")


# --- F4: what (if anything) predicts inversion — the HONEST version --------------------------------
# Both candidate behavioural predictors are shown, because neither orders the data. This figure exists
# to make that failure visible rather than to sell a curve.
def fig_predictors(fd, d2data=None):
    """Deliberately shows BOTH candidate behavioural predictors failing. Emergent cells are drawn as
    squares, not just a second colour: all 21 of them coincide at exactly (1.000, 0.000), so without a
    distinct marker plus an explicit count the entire arm hides underneath one instructed point."""
    grid = fd.get("grid_e2b") or {}
    d2 = (d2data or {}).get("d2") or {}
    if not grid:
        print("skip fig_predictors (no grid_e2b)")
        return
    dyn = (fd.get("dynamics") or {}).get("e2dyn2_8b_s0") or []
    fig, axes = plt.subplots(1, 2, figsize=_size(11.0, 4.6))
    for ax, xkey, dkey, xlabel in (
        (axes[0], "decep", "decep", "measured rival deception rate"),
        (axes[1], "rival_entropy_mean", "rival_entropy", "mean rival answer entropy (nats)"),
    ):
        _chance(ax)
        # Emergent is drawn as a LARGE OPEN RING, instructed as a small filled dot, so that where the two
        # arms coincide exactly — which is most of the emergent arm — the dot nests visibly inside the
        # ring. Distinct marker shapes alone do not fix exact coincidence: whichever is drawn last simply
        # hides the other, and jittering would misstate values that are exactly 1.000 / 0.000.
        from collections import Counter
        counts = {}
        for arm, color, lab, kw in (
            ("em", ALLY, "emergent (RL)",
             dict(s=150, marker="o", facecolor="none", edgecolor=ALLY, linewidth=1.9, zorder=5)),
            ("in", MIXED, "instructed",
             dict(s=46, marker="o", color=MIXED, alpha=0.9, edgecolor="white", linewidth=0.9, zorder=4)),
        ):
            pts = [(v.get(xkey), v.get("auroc")) for k, v in grid.items()
                   if f"_{arm}_" in k and v.get(xkey) is not None and v.get("auroc") is not None]
            if not pts:
                continue
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                       label=f"{lab} · 7 families × 3 seeds", **kw)
            mode, cnt = Counter([(round(a, 6), round(b, 6)) for a, b in pts]).most_common(1)[0]
            if cnt > 2:
                counts[arm] = (mode, cnt)
        # one combined annotation — two separate ones anchored to the same point overprint each other
        if counts:
            mode = list(counts.values())[0][0]
            parts = [f"{c} {'emergent' if a == 'em' else 'instructed'}"
                     for a, (_, c) in counts.items()]
            # flip the label to the inside when the anchor sits near the left spine, else it clips
            lo, hi = ax.get_xlim()
            left = mode[0] < lo + 0.33 * (hi - lo)
            # 16pt to the right of an anchor at x=0 still lands the first character under the
            # instructed dots clustered at low entropy; clear them properly.
            ax.annotate(" + ".join(parts) + "\ncells coincide here", mode,
                        textcoords="offset points", xytext=(34 if left else -14, 12),
                        ha="left" if left else "right", va="bottom",
                        fontsize=NOTE_FS, fontweight="bold", color=INK2, zorder=8,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.92))
        # the D2 within-model ladder — the points that refute both accounts
        if d2:
            dd = [(v.get(xkey), v.get("auroc"), v.get("rung"), v.get("model"))
                  for v in d2.values() if v.get(xkey) is not None and v.get("auroc") is not None]
            if dd:
                ax.scatter([p[0] for p in dd], [p[1] for p in dd], s=64, marker="D", color=THIRD,
                           edgecolor="white", linewidth=1.2, zorder=5,
                           label="D2 directive ladder (within-model)")
                # These labels used to hang off their own markers. Every offset that cleared one
                # obstacle hit another: -14pt dropped "Llama-8B max" onto the x axis, +13pt put it
                # through the coincident-cells note, and the Mistral labels overprinted the diamonds
                # they name. Park them all in the empty upper-left corner instead (nothing plots at
                # low x and mid y in either panel) and let a thin leader do the pointing.
                tag = [(x, y, mdl, rung) for x, y, rung, mdl in dd
                       if rung in ("max",) or (y is not None and y > 0.9
                                               and x is not None and x > 0.99)]
                for i, (x, y, mdl, rung) in enumerate(sorted(tag, key=lambda t: -(t[1] or 0))):
                    ax.annotate(f"{FAM_LABEL.get(mdl, mdl)} {rung}", xy=(x, y), xycoords="data",
                                xytext=(0.05, 0.80 - 0.11 * i), textcoords="axes fraction",
                                ha="left", va="top",
                                fontsize=(5.2 if PAPER else 7.5), fontweight="bold", color=THIRD,
                                zorder=8, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.92),
                                arrowprops=dict(arrowstyle="-", color=THIRD, linewidth=0.5,
                                                alpha=0.35))
        if dyn:
            ok = sorted((r.get(dkey), r.get("last_auroc")) for r in dyn
                        if r.get(dkey) is not None and r.get("last_auroc") is not None)
            if ok:
                ax.plot([a for a, _ in ok], [b for _, b in ok], color=MUTED, linewidth=1.6,
                        alpha=0.8, zorder=2, label="one RL trajectory (41 checkpoints)")
        _style(ax, None, xlabel, "truth AUROC on rival trials")
        ax.set_ylim(-0.06, 1.12)
    axes[0].set_title("(a) by lie RATE", fontsize=(7.5 if PAPER else 10.5), fontweight="bold", color=INK, pad=8)
    axes[1].set_title("(b) by CONFIDENCE (entropy)", fontsize=(7.5 if PAPER else 10.5), fontweight="bold",
                      color=INK, pad=8)
    h, l = axes[0].get_legend_handles_labels()
    # ncol=4 made the legend wider than the figure, so tight bbox grew the canvas to 7.12in (77% fonts).
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=NOTE_FS,
               bbox_to_anchor=(0.5, -0.10))
    _suptitle(fig, "Neither lie rate nor confidence orders the readout: cells with the same behaviour "
                   "sit at both 0.00 and 1.00", fontsize=12)
    _save(fig, "fig_predictors.png", rect=(0, 0.10, 1, 0.94))


# --- F5: the cross-family confound, with seeds -----------------------------------------------------
FAM_LABEL = {"qwen-3b": "Qwen-3B", "qwen-7b": "Qwen-7B", "qwen-14b": "Qwen-14B",
             "qwen-32b": "Qwen-32B", "gemma-9b": "Gemma-9B", "mistral-7b": "Mistral-7B",
             "8b": "Llama-8B"}


def fig_cross_family(fd):
    """A PAIRED DOT PLOT, not a scatter. In a scatter of AUROC vs deception rate, all 7 emergent cells
    and 3 of the instructed cells land on exactly the same point (1.000, 0.000) — ten coincident markers
    and ten overprinted labels. One row per family with one dot per arm shows every cell, and the lie
    rate rides along as a direct label."""
    grid = fd.get("grid_e2b") or {}
    if not grid:
        print("skip fig_cross_family (no grid_e2b)")
        return

    rows = []
    for fam, flabel in FAM_LABEL.items():
        cell = {}
        for arm in ("em", "in"):
            seeds = [v for k, v in grid.items() if k.startswith(f"{fam}_{arm}_s")]
            au = [s["auroc"] for s in seeds if s.get("auroc") is not None]
            dc = [s["decep"] for s in seeds if s.get("decep") is not None]
            if au and dc:
                cell[arm] = dict(auroc=float(np.mean(au)), sd=float(np.std(au)),
                                 decep=float(np.mean(dc)), n=len(au))
        if cell:
            rows.append((flabel, cell))
    if not rows:
        print("skip fig_cross_family (no complete family rows)")
        return
    # order by the instructed arm's lie rate: the ordering variable the paper argues is doing the work
    rows.sort(key=lambda r: r[1].get("in", r[1].get("em"))["decep"])

    fig, ax = plt.subplots(figsize=_size(8.6, 0.62 * len(rows) + 2.4))
    ax.axvline(0.5, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    max_sd = 0.0
    # Dodge the two arms within each row. Without it, the families where BOTH arms sit at exactly 0.000
    # (Gemma-9B, Qwen-14B, Qwen-32B) render as a single marker and the emergent arm vanishes again.
    DODGE = 0.17
    for y, (flabel, cell) in enumerate(rows):
        a, b = cell.get("em"), cell.get("in")
        if a and b:                                    # connector shows the arm gap at a glance
            ax.plot([a["auroc"], b["auroc"]], [y + DODGE, y - DODGE], color=MUTED, linewidth=1.4,
                    alpha=0.5, zorder=2)
        for arm, color, marker, dy, lab in (("em", ALLY, "s", +DODGE, "emergent (RL)"),
                                            ("in", MIXED, "o", -DODGE, "instructed")):
            v = cell.get(arm)
            if not v:
                continue
            max_sd = max(max_sd, v["sd"])
            ax.errorbar(v["auroc"], y + dy, xerr=v["sd"], fmt=marker, markersize=8, color=color,
                        ecolor=color, elinewidth=1.4, capsize=3, markeredgecolor="white",
                        markeredgewidth=1.2, zorder=4, label=lab if y == 0 else None)
            # above the marker, not beside it: a horizontal offset collides with the error-bar cap
            # Both arms sit at the same x in the saturated families, so a shared +9 offset stacked the
            # two value labels on each other. Put the upper arm's label above and the lower arm's below.
            ax.annotate(f"{v['decep']:.3f}", (v["auroc"], y + dy), textcoords="offset points",
                        xytext=(0, 8 if dy > 0 else -8), ha="center",
                        va="bottom" if dy > 0 else "top",
                        fontsize=(5.2 if PAPER else 7.5), color=color)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=LEG_FS)
    # -0.55 left the bottom row's below-marker value label sitting on the x-axis spine.
    ax.set_ylim(-0.78, len(rows) - 0.45)
    ax.set_xlim(-0.08, 1.14)
    _style(ax, "Inversion tracks the arm's own lie rate in both arms, not how the arm was trained",
           "truth AUROC on rival trials  (0.5 = chance, below = inverted)", None)
    ax.title.set_fontsize(7.5 if PAPER else 10.5)
    ax.grid(False, axis="y")
    # fontsize=9.5 and the one-line note below were both screen-sized: together they made the saved
    # canvas 9.97in, i.e. every font printed at 55% of the size written here.
    ax.legend(frameon=False, fontsize=LEG_FS, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.15))
    note = (f"numbers beside each marker = that arm's measured rival lie rate\n"
            f"markers = mean over 3 seeds, error bars ±1 SD "
            f"({'exactly 0 in every cell' if max_sd < 1e-9 else f'max {max_sd:.3f}'})")
    ax.text(0.0, -0.30, note, transform=ax.transAxes, fontsize=NOTE_FS, color=INK2, va="top")
    _save(fig, "fig_cross_family.png")


# --- F6: the D2 within-model directive ladder + text-matched controls -------------------------------
LADDER = ["hint", "soft", "default", "firm", "max"]
CONTROLS = ["max_nonneg", "max_filler"]
CONTROL_LABEL = {"max_nonneg": "max, no negation words", "max_filler": "max length, default rule"}


def fig_refit_artifact(d2data):
    """The D2 wave's actual result: the apparent prompt-driven inversion is a REFIT artifact.

    Every point is one system-prompt variant of the SAME model (identical weights). x = the arm's measured
    lie rate. Red = a probe refit on that variant's own ally activations (the conventional protocol).
    Blue = ONE probe direction, fit once on the `default` variant, cross-scored on every variant.

    Both probes score ally IID ~1.000 everywhere, so by the conventional criterion both are valid truth
    probes — yet they disagree by up to 0.9 on rival trials. That is probe under-determination shown
    without any training and without the codebook."""
    d2 = (d2data or {}).get("d2") or {}
    pts = [v for v in d2.values()
           if v.get("model") == "8b" and v.get("auroc") is not None
           and v.get("auroc_frozen") is not None and v.get("decep") is not None]
    if len(pts) < 3:
        print("skip fig_refit_artifact (need the d2f_* frozen-probe runs)")
        return
    pts.sort(key=lambda v: v["decep"])
    xs = [v["decep"] for v in pts]
    refit = [v["auroc"] for v in pts]
    frozen = [v["auroc_frozen"] for v in pts]

    # Flatter than it was: this is a body figure and its height is page budget. The callout, legend and
    # note now stack inside the empty low-x half of the axes, so the extra height bought nothing.
    # BODY figure, included at 0.92\textwidth: render at that width or \includegraphics shrinks
    # every fontsize to 94% (see fig_action_identity).
    fig, ax = plt.subplots(figsize=_size(8.6, 4.1, 0.92))
    _chance(ax)
    for x, a, b in zip(xs, refit, frozen):        # connector makes the per-variant gap legible
        ax.plot([x, x], [a, b], color=MUTED, linewidth=1.0, alpha=0.5, zorder=2)
    ax.scatter(xs, frozen, s=64, marker="o", color=MIXED, edgecolor="white", linewidth=1.1, zorder=5,
               label="one fixed probe, cross-scored on every variant")
    ax.scatter(xs, refit, s=64, marker="s", color=ALLY, edgecolor="white", linewidth=1.1, zorder=4,
               label="probe refit on each variant's own ally data (conventional)")
    worst = max(pts, key=lambda v: abs(v["auroc_frozen"] - v["auroc"]))
    # Anchored in AXES FRACTION, not as an offset from the point. An offset put the text off the right
    # edge, which (a) buried it under two red markers and (b) made bbox_inches="tight" widen the canvas
    # to 7.18in, so LaTeX scaled the whole figure -- and every font in it -- down to 77%.
    # The low-x half of the axes is empty by construction: no variant lies at deception < 0.5.
    ax.annotate(f"same activations, same layer:\nrefit reads {worst['auroc']:.3f},"
                f" fixed probe reads {worst['auroc_frozen']:.3f}\n(both score ally IID 1.000)",
                xy=(worst["decep"], worst["auroc"]), xycoords="data",
                xytext=(0.03, 0.66), textcoords="axes fraction", ha="left", va="top",
                fontsize=NOTE_FS, fontweight="bold", color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1.0))
    _style(ax, "The prompt-driven “inversion” is an artifact of refitting the probe per condition",
           "measured rival deception rate", "truth AUROC on rival trials")
    ax.set_ylim(-0.06, 1.14)
    ax.set_xlim(-0.02, 1.0)
    # Legend, callout and note all stack in the low-x half of the axes, which is empty by construction
    # (no variant lies below deception 0.5). Anchoring the legend below the axes instead put it outside
    # the figure, where tight_layout cannot see it and bbox_inches="tight" grew the canvas to 7.18in.
    ax.legend(frameon=False, fontsize=NOTE_FS, loc="upper left", bbox_to_anchor=(0.01, 0.44), ncol=1)
    ax.text(0.02, 0.16, f"{len(pts)} system-prompt variants of one model\n(identical weights), N=1000 each",
            transform=ax.transAxes, fontsize=NOTE_FS, color=INK2, va="top")
    _save(fig, "fig_refit_artifact.png")


def fig_d2(d2data):
    d2 = (d2data or {}).get("d2") or {}
    if not d2:
        print("skip fig_d2 (no d2_figuredata.json — produce it with the d2 analysis run and copy it into data/)")
        return
    models = []
    for cell in d2.values():
        m = cell.get("model")
        if m and m not in models:
            models.append(m)
    fig, axes = plt.subplots(1, len(models), figsize=_size(5.0 * len(models), 4.3), squeeze=False)
    for i, fam in enumerate(models):
        ax = axes[0][i]
        _chance(ax)
        rows = [(r, d2[k]) for k in d2 for r in [d2[k].get("rung")] if d2[k].get("model") == fam]
        by = {r: v for r, v in rows}
        xs = [j for j, r in enumerate(LADDER) if r in by]
        ys = [by[LADDER[j]]["auroc"] for j in xs]
        ax.plot(xs, ys, color=ALLY, linewidth=2.0, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4,
                label="probe refit per rung (conventional)")
        # The same ladder read by ONE fixed probe. Without this series the figure asserts a prompt-driven
        # inversion that the frozen-probe wave showed is a refit artifact.
        fz = [(j, by[LADDER[j]].get("auroc_frozen")) for j in xs]
        fz = [(j, v) for j, v in fz if v is not None]
        if fz:
            ax.plot([j for j, _ in fz], [v for _, v in fz], color=MIXED, linewidth=2.0, marker="o",
                    markersize=7, markeredgecolor="white", markeredgewidth=1.0, zorder=5,
                    label="one fixed probe, cross-scored")
        for j, y in zip(xs, ys):
            d = by[LADDER[j]].get("decep")
            if d is not None:
                # below the marker normally, but above it for points near the floor: at AUROC 0.077 a
                # -26pt offset lands the label on the x-axis tick labels.
                lo_pt = y < 0.22
                # -26pt from a marker at ~1.0 puts the text on the 0.5 chance line (struck through in
                # print). Keep it tight to its own marker instead, and flip it above near the floor.
                # ONE LINE, just the number (the caption says what it is). Two lines were ~0.16 data
                # units tall, so a label under a marker at 0.79 reached the 0.5 chance line and the
                # dashed rule printed straight through it. Repeating "lie rate" five times per panel
                # was clutter anyway.
                # For a point near the floor there is no room below AND the steep refit descent
                # occupies the space above, so put that one BESIDE the marker.
                edge = "left" if j == 0 else ("right" if j == len(LADDER) - 1 else "center")
                if lo_pt:
                    ax.annotate(f"{d:.2f}", (j, y), textcoords="offset points", xytext=(8, 0),
                                ha="left", va="center",
                                fontsize=(5.2 if PAPER else 7.5), color=INK2)
                else:
                    ax.annotate(f"{d:.2f}", (j, y), textcoords="offset points",
                                xytext=(-3 if edge == "right" else (3 if edge == "left" else 0), -9),
                                ha=edge, va="top",
                                fontsize=(5.2 if PAPER else 7.5), color=INK2)
        # The text-matched controls. Their LIE RATE is the whole point: both land at ~0.57 versus `max`'s
        # 0.80, so they are NOT behaviourally matched and cannot separate "directive text" from
        # "behaviour". Label the rate, or the figure implies a clean control that we do not have.
        present = [c for c in CONTROLS if by.get(c) and by[c].get("auroc") is not None]
        for j, c in enumerate(present):
            v = by[c]
            xpos = LADDER.index("max") + 0.16 * (j + 1)
            ax.scatter([xpos], [v["auroc"]], s=90, marker="D", color=THIRD,
                       edgecolor="white", linewidth=1.1, zorder=5,
                       label="text-matched control (NOT rate-matched)" if j == 0 else None)
            d = v.get("decep")
            # below the marker: above it collides with the panel title for controls sitting near 1.0
            # Stacked offsets from the markers put both labels across the chance line and on top of
            # each other AND of the "lie rate" labels. Park them in the empty mid-left of the panel
            # (nothing plots there: the refit curve only falls after `firm`) with a leader to each.
            ax.annotate(f"{CONTROL_LABEL[c]} ({d:.2f})" if d is not None else CONTROL_LABEL[c],
                        xy=(xpos, v["auroc"]), xycoords="data",
                        xytext=(0.03, 0.27 - 0.13 * j), textcoords="axes fraction",
                        ha="left", va="top", fontsize=(5.2 if PAPER else 7.5),
                        color=THIRD, fontweight="bold", zorder=8, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.92),
                        arrowprops=dict(arrowstyle="-", color=THIRD, linewidth=0.5, alpha=0.35))
        # never let un-run conditions read as measured nulls
        missing = []
        if not by.get("hint"):
            missing.append("weaker rungs")
        if not any(by.get(r, {}).get("auroc_frozen") is not None for r in LADDER):
            missing.append("fixed-probe series")
        if missing:
            ax.text(0.02, 0.04, " and ".join(missing) + " not run for this model",
                    transform=ax.transAxes, fontsize=NOTE_FS, color=INK2, style="italic")
        ax.set_xticks(range(len(LADDER)))
        ax.set_xticklabels(LADDER, fontsize=NOTE_FS)
        _style(ax, FAM_LABEL.get(fam, fam), "directive strength (system-prompt rival clause)",
               "truth AUROC on rival trials")
        ax.set_ylim(-0.06, 1.10)
    h, l = axes[0][0].get_legend_handles_labels()
    # fontsize=9.5 is the SCREEN size: in PAPER mode it printed a legend larger than the panel titles,
    # overprinted both x-axis labels, and widened the canvas to 6.25in. LEG_FS + a lower anchor + space
    # reserved by tight_layout fixes all three.
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=LEG_FS,
               bbox_to_anchor=(0.5, 0.004))
    # NOTE: this figure previously claimed "prompting alone moves the readout from 1.000 to 0.080".
    # The frozen-probe wave refuted that — under a fixed probe the ladder is flat at ~0.99. Title corrected.
    _suptitle(fig, "The directive ladder moves the REFIT probe, not the representation", fontsize=12.5)
    _LADDER_RECT = (0, 0.19, 1, 1)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    _save(fig, "fig_d2_ladder.png", rect=_LADDER_RECT)


def main():
    # Prefer the single consolidated e3 blob (grid + identification + dynamics + d2 in one file, emitted by
    # the e3 analysis job and copied across). Fall back to the older split dumps.
    fd = load("e3_consolidated.json")
    d2 = {"d2": fd["d2"]} if fd.get("d2") else load("d2_figuredata.json")
    if not fd:
        fd = load("e2_consolidated.json")
    if not fd and not d2:
        print("no data/ dumps found — nothing to render")
        return
    fig_identification(fd)
    fig_action_identity(fd)
    fig_suppression(fd)
    fig_predictors(fd, d2)
    fig_cross_family(fd)
    fig_d2(d2)
    fig_refit_artifact(d2)
    # The 5b set (2026-08-17). The capability-ladder figure from that plan is deliberately NOT here:
    # it is dropped (user-confirmed) because it would resurrect the withdrawn §1b.
    fig_depth_step(fd)
    fig_freeze_transfer(fd)
    fig_settling(fd)
    fig_causal(fd)
    fig_depth_sweep(fd)
    fig_instrpair(fd)
    fig_geom(fd)
    fig_matched_pairs(d2)

    # A figure wider than we asked for prints with shrunken fonts (see _save). Fail loudly.
    if _OVERFLOW:
        print("\n*** %d FIGURE(S) OVERFLOWED THEIR CANVAS -- fonts will print smaller than specified ***"
              % len(_OVERFLOW))
        for n, w, g, blame in _OVERFLOW:
            print(f"      {n}: asked {w:.2f}in, saved {g:.2f}in ({w / g:.0%} of specified font size)")
            for b in blame:
                print(f"          {b}")
    elif PAPER:
        print("\nall figures within their specified width")
    if _DASHES:
        print("\n*** EM-DASH / EN-DASH IN RENDERED FIGURE TEXT (the paper is zero-em-dash) ***")
        for n, ss in _DASHES:
            for x in ss:
                print(f"      {n}: {x}")


# ====================================================================================================
# The 5b figure set. Added 2026-08-17. These read the two sections the e3 analysis job gained that day
# (`instrpair`, `interventions`) plus the transfer/infercode identification cells.
#
# EVERY number below is read from the blob. Nothing is hard-coded — a copy step is a stale-artifact bug
# waiting to happen, and this module's header says so. Where a slice is missing the figure skips itself
# with a printed note rather than rendering a plausible-looking half-figure.
# ====================================================================================================

def _skip(name, why):
    print(f"skipped {name}: {why}")


# --- F7 (5b-i): the depth step — a STATED bit is shallow, an INFERRED bit is deep ------------------
# Carries §0b. The contrast is the point: reading a bit the prompt states is a shallow lexical operation
# that saturates almost immediately; computing a bit the prompt only implies does not become linearly
# available until roughly mid-stack. Same model, same probe protocol, same metric — only the task differs.
def fig_depth_step(fd):
    ident = fd.get("identification") or {}
    stated = ident.get("cbid_gemma-9b_em")           # codebook: the bit is written in the prompt
    inferred = ident.get("id4_infcb_g9b_in")         # infercode: the bit must be derived
    infer_rep = ident.get("infid_gemma-9b_in")       # same task at n=1000 — a replicate, plotted faintly
    if not stated or not inferred:
        return _skip("fig_depth_step.png", "need cbid_gemma-9b_em + id4_infcb_g9b_in")

    def xy(cell):
        pts = [(r["l"], r["truth_mixed"]) for r in cell["curve"] if r.get("truth_mixed") is not None]
        return [p[0] for p in pts], [p[1] for p in pts]

    # included at 0.82\textwidth (see fig_action_identity): render at the width it is given, or
    # every fontsize here prints at 84%. The aspect is TALLER than the 7.2x4.0 it used to be because
    # narrowing the canvas without it shrank the axes height from 2.94in to 2.40in while the legend
    # stayed at the same POINT size, which pushed the legend up across the chance line and printed the
    # dashed chance rule straight through "stated bit (prompt says it)". 4.7 restores the old 2.94in.
    fig, ax = plt.subplots(figsize=_size(7.2, 4.7, 0.82))
    xs, ys = xy(stated)
    ax.plot(xs, ys, color=ALLY, lw=2, marker="o", ms=2.5, label="stated bit (prompt says it)", zorder=3)
    xi, yi = xy(inferred)
    ax.plot(xi, yi, color=MIXED, lw=2, marker="o", ms=3.5, label="inferred bit (model must derive it)",
            zorder=3)
    if infer_rep:
        xr, yr = xy(infer_rep)
        ax.plot(xr, yr, color=MIXED, lw=1.0, alpha=0.45, linestyle=(0, (3, 2)), zorder=2,
                label=f"inferred, replicate (n={infer_rep.get('n')})")
    _chance(ax)

    # Direct labels (the palette's aqua-contrast relief rule generalises: label every series in-axes).
    def _sat(xs_, ys_, thr=0.99):
        for x, y in zip(xs_, ys_):
            if y >= thr:
                return x
        return None
    s_sat, i_sat = _sat(xs, ys), _sat(xi, yi)
    for x, col, txt in ((s_sat, ALLY, "saturates L%s" % s_sat), (i_sat, MIXED, "saturates L%s" % i_sat)):
        if x is None:
            continue
        ax.axvline(x, color=col, lw=0.8, alpha=0.5, linestyle=(0, (2, 2)), zorder=1)
    # Every number in these annotations is DERIVED from the curve. "at chance to L16" was hard-coded in
    # the first version, which is precisely the stale-artifact bug this module's header warns about: it
    # would have kept printing L16 after the layer grid changed.
    if s_sat is not None:
        ax.annotate(f"stated: {dict(zip(xs, ys))[s_sat]:.3f} by L{s_sat}", xy=(s_sat, 1.0),
                    xytext=(s_sat + 1.5, 0.80), fontsize=NOTE_FS, color=ALLY, fontweight="bold")
    if i_sat is not None:
        below = [x for x, y in zip(xi, yi) if y < 0.6 and x < i_sat]
        last_chance = max(below) if below else None
        lead = f"inferred: at chance to L{last_chance},\n" if last_chance is not None else "inferred: "
        ax.annotate(f"{lead}then {dict(zip(xi, yi))[i_sat]:.3f} by L{i_sat}", xy=(i_sat, 1.0),
                    xytext=(i_sat + 1.5, 0.55), fontsize=NOTE_FS, color=MIXED, fontweight="bold")
    # The inferred curve is measured on a SPARSE layer grid, so the line interpolates across layers that
    # were never measured and the transition point inside the gap is not resolved. Say so on the axes.
    gap = max((b - a for a, b in zip(xi, xi[1:])), default=0)
    if gap > 1:
        ax.annotate(f"inferred curve sampled every {gap} layers:\ntransition not resolved within a gap",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=NOTE_FS, color=MUTED)
    # Printed accounting for the SILENT family selection, matching this module's existing convention.
    others = sorted(k for k, v in ident.items()
                    if v.get("task") == "infercode" and not k.startswith("xfer")
                    and k not in ("id4_infcb_g9b_in", "infid_gemma-9b_in"))
    if others:
        print(f"  note: fig_depth_step shows Gemma-9B only; other inferred-bit cells exist "
              f"and are NOT plotted: {others}")
    ax.set_ylim(0.35, 1.04)
    _style(ax, xlabel="layer", ylabel="truth AUROC (mixed-fit, rival contexts)")
    ax.legend(fontsize=LEG_FS, frameon=False, loc="lower right")
    _suptitle(fig, "A stated bit is readable shallow; an inferred bit needs depth")
    _save(fig, "fig_depth_step.png")


# --- F8 (5b-ii): freeze-and-transfer ----------------------------------------------------------------
# The methodological point of the paper in one panel. REFIT per condition (the conventional practice)
# reads ~1.000 on the transfer task at depth. The SAME probe FROZEN from the stated-bit task and merely
# cross-scored does not — and on the reward-trained arm it INVERTS. A refit number therefore certifies
# almost nothing about whether a direction transfers.
def fig_freeze_transfer(fd):
    ident = fd.get("identification") or {}
    arms = [("xfer_score_g9b_in", "instructed"), ("xfer_score_g9b_em", "reward-trained"),
            ("xfer_score_g9b_base", "base model")]
    have = [(k, lab) for k, lab in arms if ident.get(k)]
    if not have:
        return _skip("fig_freeze_transfer.png", "no xfer_score_* cells in blob")

    fig, axes = plt.subplots(1, len(have), figsize=_size(9.6, 3.4), sharey=True)
    if len(have) == 1:
        axes = [axes]
    for ax, (key, lab) in zip(axes, have):
        cell = ident[key]
        rf = [(r["l"], r["truth_mixed"]) for r in cell["curve"] if r.get("truth_mixed") is not None]
        fz = [(r["l"], r["truth_mixed_frozen"]) for r in cell["curve"]
              if r.get("truth_mixed_frozen") is not None]
        if rf:
            ax.plot([p[0] for p in rf], [p[1] for p in rf], color=ALLY, lw=2, marker="o", ms=3,
                    label="refit on the new task", zorder=3)
        if fz:
            ax.plot([p[0] for p in fz], [p[1] for p in fz], color=MIXED, lw=2, marker="s", ms=3,
                    label="frozen, cross-scored", zorder=3)
        _chance(ax)
        ax.set_ylim(0.20, 1.04)
        _style(ax, title=lab, xlabel="layer")
        # Call out the inversion where it happens — it is the finding, not a blemish.
        if fz:
            lo = min(fz, key=lambda p: p[1])
            # Only call it an INVERSION when it is unambiguously one. At 0.450 the base-model curve is
            # barely under chance and labelling that "inverts" would overstate it — the reward-trained
            # arm's 0.282 is the real thing.
            if lo[1] < 0.40:
                # y=0.27 put the text ON the curve it labels (the minimum is 0.282). Lift it into the
                # empty band between the chance line and the refit curve, and keep the leader.
                ax.annotate(f"inverts\n{lo[1]:.3f}", xy=lo, xytext=(lo[0] - 9, 0.63),
                            ha="center", va="center",
                            fontsize=NOTE_FS, color=MIXED, fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color=MIXED, lw=0.8))
    axes[0].set_ylabel("truth AUROC on the transfer task", fontsize=6.5 if PAPER else 9.5, color=INK2)
    axes[0].legend(fontsize=LEG_FS, frameon=False, loc="lower left")
    _suptitle(fig, "Refit certifies nothing: the frozen probe does not transfer")
    _save(fig, "fig_freeze_transfer.png")


# --- F9 (5b-iii): the causal arm — steering sweep + why the ablation is VOID ------------------------
# ⚠️ SCOPE NOTE, deliberate and load-bearing. The 5b plan called for "steering/ablation LAYER sweep,
# ally-fit vs mixed-fit direction". NEITHER contrast exists in any run on the volume: every causal arm
# loaded a direction with `regime=mixed` (there is no ally-fit causal run at all), and steering was only
# ever run at L32, so there is no layer sweep either. Rather than fabricate the missing arms, this figure
# plots what was actually measured — the ALPHA sweep at L32 against its in-run alpha=0 baseline, split by
# true bit, plus the positive-control panel that makes the ablation VOID rather than null.
# --- F8b: THE SETTLING EXPERIMENT — frozen transfer to held-out inferred-truth tasks ----------------
# The paper's sharpest self-imposed limitation, answered. Two arms, and the pairing IS the argument:
#   LEFT  (clean)        fit on templates {0,1}, both rule orientations -> no shallow shortcut exists
#   RIGHT (contaminated) fit on `gt` only -> the bit reduces to "which slot is larger", a shallow feature
# Each panel shows the SAME probe frozen and scored two ways (held-out task vs its own same-task control),
# plus the refit ceiling. The reported quantity is held-out MINUS control, so the two frozen series are
# drawn in one hue and the gap between them is shaded: the shading IS the transfer cost.
#
# ⚠️ THE MISREADING THIS FIGURE MUST PREVENT. Ranked by the gap alone the "best" layer is the shallowest,
# where the gap is ~0.005 -- but there the CONTROL is also at chance, so a small gap means no signal for
# either probe, not good transfer (the levels-vs-paired-differences confusion). Those
# layers are therefore explicitly greyed and labelled, per arm, computed from the control rather than
# hard-coded.
def fig_settling(fd):
    ident = fd.get("identification") or {}
    # The held-out target DIFFERS between panels, so it belongs in each TITLE, not in the shared legend --
    # a single legend entry saying "held-out templates {2,3}" is simply false of the right-hand panel.
    ARMS = [("settle_held_tplB_g9b_in", "settle_ctl_tplA_g9b_in",
             "clean fit: both orientations\n$\\rightarrow$ held-out templates {2,3}"),
            ("settle_held_lt_g9b_in", "settle_ctl_gt_g9b_in",
             "contaminated fit: one orientation\n$\\rightarrow$ held-out flipped rule")]
    if not all(ident.get(h) and ident.get(c) for h, c, _ in ARMS):
        return _skip("fig_settling.png", "need the settle_held_*/settle_ctl_* arms in the blob")

    fig, axes = plt.subplots(1, 2, figsize=_size(9.6, 3.8), sharey=True)
    for ax, (hkey, ckey, title) in zip(axes, ARMS):
        hc, cc = ident[hkey], ident[ckey]
        def series(cell, field):
            return {r["l"]: r.get(field) for r in cell["curve"] if r.get(field) is not None}
        froz, ctl = series(hc, "truth_mixed_frozen"), series(ckey and cc, "truth_mixed_frozen")
        refit = series(hc, "truth_mixed")
        xs = sorted(set(froz) & set(ctl))

        # Grey the layers where the CONTROL itself is at chance: nothing there is readable by any probe,
        # so the small held-out/control gap is uninformative rather than good.
        dead = [l for l in xs if ctl[l] < 0.6]
        if dead:
            ax.axvspan(min(xs) - 1, max(dead) + 2, color=MUTED, alpha=0.10, zorder=0)
            ax.annotate("control at chance:\nno signal\nfor any probe", xy=(min(xs) + 0.2, 0.86),
                        fontsize=NOTE_FS, color=INK2, ha="left", va="top")

        ax.fill_between(xs, [froz[l] for l in xs], [ctl[l] for l in xs],
                        color=MIXED, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(xs, [refit[l] for l in xs if l in refit] if set(refit) >= set(xs)
                else [refit.get(l) for l in xs],
                color=ALLY, lw=1.3, linestyle=(0, (1.5, 1.5)), zorder=3,
                label="refit on the held-out task (conventional)")
        ax.plot(xs, [ctl[l] for l in xs], color=MIXED, lw=1.4, linestyle=(0, (4, 2.5)), zorder=4,
                label="same probe, frozen, SAME task (control)")
        ax.plot(xs, [froz[l] for l in xs], color=MIXED, lw=2.2, marker="o", ms=3.4, zorder=5,
                label="same probe, frozen, HELD-OUT task")
        _chance(ax)

        # Direct-label the two layers that carry the claim, chosen from the data among layers where the
        # control has signal -- never hard-coded, so a denser grid cannot leave a stale annotation behind.
        # Mark the best-transferring layer with a rule rather than a text box: both arms peak at the SAME
        # layer, and a vertical line in each panel makes that agreement visible without crowding. Exact
        # values live in the LaTeX caption, which is where this paper puts its numbers anyway.
        live = [l for l in xs if ctl[l] >= 0.6]
        if live:
            best = max(live, key=lambda l: froz[l] - ctl[l])
            ax.axvline(best, color=THIRD, lw=0.9, linestyle=(0, (2, 2)), zorder=1)
            # Horizontal, in the open region below the deep plateau. A rotated in-axes label was clipped by
            # the axes edge on one panel and not the other -- caught by looking, not by the build.
            ax.annotate(f"peak L{best}\n{froz[best]:.3f} ({froz[best]-ctl[best]:+.3f})",
                        xy=(max(xs) - 1.0, 0.30), fontsize=NOTE_FS, color=THIRD, fontweight="bold",
                        ha="right", va="center")
        # ⚠️ AN INVERSION IS NOT "BELOW 0.5". At 0.451 a probe is at CHANCE, and labelling that an
        # inversion would manufacture the paper's most striking claim out of noise -- the first version of
        # this figure did exactly that on the clean arm. Require a decisive margin, so the label appears
        # only on the arm where the probe really is an anti-detector.
        inv = [l for l in xs if froz[l] < 0.25]
        if inv:
            lo = min(inv, key=lambda l: froz[l])
            # Sits directly above the inverted plateau, no leader line: an arrow from open space had to
            # cross the rising curve to reach L16, which read as pointing at the wrong layer.
            ax.annotate(f"inverts to {froz[lo]:.3f}", xy=(min(inv) + 2.5, 0.05),
                        fontsize=NOTE_FS, color=ALLY, fontweight="bold", ha="left", va="bottom")
        ax.set_ylim(-0.04, 1.06)
        _style(ax, title=title, xlabel="layer")
    axes[0].set_ylabel("truth AUROC (mixed-fit, rival contexts)",
                       fontsize=6.5 if PAPER else 9.5, color=INK2)
    # ONE legend, BELOW both panels. Placed in-axes it collided with three annotations and hid two of
    # them outright -- a defect invisible to build.sh and found only by looking at the PNG.
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=LEG_FS, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.004))
    _suptitle(fig, "A frozen probe transfers to held-out inferred-truth tasks; the confound is shallow")
    _save(fig, "fig_settling.png", rect=(0, 0.18, 1, 1))


def fig_causal(fd):
    iv = fd.get("interventions") or {}
    # ⭐ THE CONTRAST THE 5b PLAN ASKED FOR, finally runnable (2026-08-18). Until the ally-fit direction was
    # materialised from `all_directions`, every causal run in the project perturbed the mixed-fit direction
    # and "ally-fit vs mixed-fit" existed in no artifact. Both directions come from the SAME fit on the SAME
    # activations; only the fitting contexts differ. Same layer, same arm, same episodes, same alpha grid.
    pairs = [("revsteer2_g9b_in_l32", "allyfit2_g9b_in_l32", "instructed"),
             ("revsteer2_g9b_em_l32", "allyfit2_g9b_em_l32", "reward-trained")]
    have = [(m, a, lab) for m, a, lab in pairs
            if (iv.get(m) or {}).get("steering_sweep") and (iv.get(a) or {}).get("steering_sweep")]
    if not have:
        return _skip("fig_causal.png", "need revsteer2_* + allyfit2_* steering sweeps")

    # ⚠️ LAYOUT REBUILT 2026-08-18. This was three panels in ONE row, which at the printed 5.5in width gave
    # each panel ~1.8in: the three titles collided, the y-axis label was clipped, and the per-ladder
    # captions under the bar chart overlapped each other. Adding the L32-determined ladder (a third group)
    # made it unreadable. The bar panel needs the full text width, so the figure is now 2 rows: the two
    # steering panels on top, the positive control spanning the full width below. Checked by rendering,
    # not by assuming — `build.sh` cannot see inside a PNG, so a broken figure passes every automated gate.
    fig = plt.figure(figsize=_size(10.5, 8.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.62, wspace=0.28)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    for ax, (mkey, akey, lab) in zip(axes[:2], have):
        base = ((iv[mkey].get("steering_baseline") or {}).get("rival_truth_rate"))
        # Plot the TRUE-BIT-1 subpopulation, not the pooled rate: pooling is what hid the effect before,
        # because true-bit-0 is pinned at 0.000 and dilutes everything by half.
        for key, col, mk, lb in ((mkey, MIXED, "o", "mixed-fit (identified)"),
                                 (akey, ALLY, "s", "ally-fit (the criticised one)")):
            rows = sorted(iv[key]["steering_sweep"]["rates"], key=lambda r: r["alpha"])
            x = [r["alpha_rel"] for r in rows]
            y = [(r.get("by_truth") or {}).get("truth1_rate") for r in rows]
            if all(v is not None for v in y):
                ax.plot(x, y, color=col, lw=1.9, marker=mk, ms=3.4, label=lb, zorder=4)
            # true-bit-0, faint: it is pinned at the floor and that is itself the finding.
            y0 = [(r.get("by_truth") or {}).get("truth0_rate") for r in rows]
            if all(v is not None for v in y0):
                ax.plot(x, y0, color=col, lw=0.9, alpha=0.4, linestyle=(0, (2, 2)), zorder=2)
        if base is not None:
            b1 = ((iv[mkey].get("steering_baseline") or {}).get("by_truth") or {}).get("truth1_rate")
            if b1 is not None:
                ax.axhline(b1, color=MUTED, lw=1.0, linestyle=(0, (4, 3)), zorder=1)
                ax.annotate(f"$\\alpha{{=}}0$: {b1:.3f}", xy=(0, b1), xytext=(-1.35, b1 + 0.10),
                            fontsize=NOTE_FS, color=INK2)
        ax.axvline(0, color=MUTED, lw=0.8, alpha=0.6, zorder=1)
        # Headroom for the panel legend: at ylim 1.02 the ally-fit curve (peaking 0.93) ran straight
        # through "ally-fit (the criticised one)".
        ax.set_ylim(-0.05, 1.34)
        _style(ax, title=f"steering, {lab}", xlabel=r"$\alpha\,/\,\|\mathrm{resid}\|$   (signed)")
    axes[0].set_ylabel("true-bit-1 (solid)\ntrue-bit-0 (dashed)" if PAPER
                       else "true-bit-1 rate (solid) · true-bit-0 (dashed)",
                       fontsize=6.0 if PAPER else 9.5, color=INK2)
    axes[0].legend(fontsize=LEG_FS, frameon=False, loc="upper left")

    # Panel 3: the positive control. A behavioural null means nothing unless the ablation removed the
    # information — and it did not, at any rank that has completed.
    ax = axes[2]
    labels, post, ceil, perm = [], [], [], []
    abl = (iv.get("cause_g9b_em_l32") or {}).get("ablation")
    if abl and isinstance(abl.get("positive_control"), dict):
        pc = abl["positive_control"]
        perm.append(pc.get("null_decodability_q95"))
        # Disambiguated: the rank-1 ABLATION arm and the rank-1 prefix of the INLP subspace are two
        # different runs that would otherwise both print "rank 1" and read as a duplicated bar.
        labels.append("1$^\\dagger$")
        post.append(pc.get("post_ablation_decodability"))
        ceil.append(pc.get("chance_ceiling"))
    sub_cell = iv.get("cause3_g9b_em_l32_sub32") or {}
    sub = sub_cell.get("subspace_ablation")
    det_cell = iv.get("cause6_g9b_em_l24_sub8d") or {}
    det = det_cell.get("subspace_ablation")

    # The L32 DETERMINED ladder, assembled 2026-08-18. It lives in two jobs because the reaper killed the
    # rank-8 stretch three times: cause4/cause5 carry k=1,2 (identical to 5 dp -- same seed, so this is
    # determinism, NOT replication, and only one of them may be plotted) and cause7 carries k=4. Without
    # this group the figure showed only the L32 *underdetermined* ladder and the L24 determined one, while
    # the caption and the text both discuss the L32 determined k=2 and k=4 passes -- a caption describing
    # bars that were not in the plot.
    det32_ranks = []
    for src_name in ("cause4_g9b_em_l32_sub32d", "cause5_g9b_em_l32_sub8d"):
        blk = (iv.get(src_name) or {}).get("subspace_ablation") or {}
        for r in blk.get("ranks", []):
            if r.get("k") in (1, 2) and not any(x.get("k") == r.get("k") for x in det32_ranks):
                det32_ranks.append(r)
    for r in ((iv.get("cause7_g9b_em_l32_sub4d") or {}).get("subspace_ablation") or {}).get("ranks", []):
        if not any(x.get("k") == r.get("k") for x in det32_ranks):
            det32_ranks.append(r)
    # k=8 comes from cause8_*_{a,b,c}: three REDUNDANT copies launched because the eviction was a coin flip,
    # not a cost problem. Same seed and same fit, so any two survivors agree to 5 dp --
    # determinism again, NOT replication -- and the `not any(...)` guard is what keeps only the first in the
    # plot. Absent keys are skipped, so this is inert until one of them lands.
    for src_name in ("cause8_g9b_em_l32_sub8d_a", "cause8_g9b_em_l32_sub8d_b",
                     "cause8_g9b_em_l32_sub8d_c"):
        blk = (iv.get(src_name) or {}).get("subspace_ablation") or {}
        for r in blk.get("ranks", []):
            if not any(x.get("k") == r.get("k") for x in det32_ranks):
                det32_ranks.append(r)
    det32 = {"ranks": sorted(det32_ranks, key=lambda r: r.get("k", 0))} if det32_ranks else None
    # k=8 is ABSENT by eviction, not by omission. Say so on the axis rather than letting the ladder look
    # complete: a gap that is not labelled reads as "we chose to stop here".
    det32_missing = det32 is not None and not any(r.get("k") == 8 for r in det32_ranks)
    # Tags name BOTH distinguishing facts (layer and fit size). The two ladders differ in more than one
    # way, so a bare "u"/"d" would invite the reader to attribute the difference to whichever one they
    # happened to have in mind.
    groups = []          # (start_index, end_index, label) for the per-ladder captions under the axis
    for src, glabel in ((sub, "L32, underdet.\n$n{=}2000$"),
                        (det32, "L32, determined\n$n{=}4000$" + ("\n($k{=}8$ evicted)" if det32_missing else "")),
                        (det, "L24, determined\n$n{=}4000$")):
        if not src:
            continue
        start = len(labels)
        for r in src.get("ranks", []):
            # A rank with no positive control says NOTHING about rank, so it must not become a bar.
            # Reading the guard rather than merely checking that a number exists is the whole lesson of
            # 2026-08-17: a forced or malformed value is still a value.
            if r.get("post_ablation_decodability") is None or r.get("positive_control_absent"):
                continue
            labels.append(f"{r['k']}")
            post.append(r["post_ablation_decodability"])
            ceil.append(r.get("chance_ceiling"))
            perm.append(r.get("null_decodability_q95"))
        if len(labels) > start:
            groups.append((start, len(labels) - 1, glabel))
    if labels:
        xs = range(len(labels))
        # Colour by whether the cell actually cleared its ceiling: a bar that did is a different kind of
        # object from one that did not, and rendering them identically is how a VOID cell gets quoted.
        cols = [MIXED if (c is not None and p <= c) else ALLY for p, c in zip(post, ceil)]
        ax.bar(xs, post, color=cols, width=0.6, zorder=3)
        # The permutation null, per cell. `chance_ceiling` is max(0.5+tol, perm q95), so the ceiling alone
        # hides which criterion bound the verdict — and at k=2d the pass is by 4e-4 against the fixed band
        # while sitting 0.02 ABOVE the permutation null. Both lines must be visible.
        if any(q is not None for q in perm):
            ax.plot(list(xs), perm, color=INK, lw=0, marker="_", ms=11, mew=1.6, zorder=6,
                    label="permutation null (q95)")
        cc = [c for c in ceil if c is not None]
        if cc:
            ax.axhline(max(cc), color=MIXED, lw=1.4, linestyle=(0, (4, 3)), zorder=4)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylim(0, 1.12)
        for i, p in enumerate(post):
            ax.annotate(f"{p:.2f}", xy=(i, p), xytext=(i, p + 0.02), ha="center", fontsize=NOTE_FS,
                        color=INK)
        _style(ax, title="positive control: was the bit removed?",
               xlabel="ablated rank $k$", ylabel="post-ablation decodability")
        # The per-ladder captions live in the space directly under the ticks, so the axis label has to be
        # pushed below them or the two overlap (they did).
        ax.xaxis.labelpad = 30 if PAPER else 22
        # Per-ladder captions under the axis, so the two ladders cannot be read as one series and the
        # reader is told BOTH ways they differ (layer and fit size) without cramming it into tick labels.
        for a, b, glab in groups:
            ax.annotate(glab, xy=((a + b) / 2.0, 0), xytext=((a + b) / 2.0, -0.16),
                        textcoords=("data", "data"), ha="center", va="top", fontsize=NOTE_FS,
                        color=INK2, annotation_clip=False)
            if a > 0:
                ax.axvline(a - 0.5, color=MUTED, lw=0.8, alpha=0.7, zorder=1)
        # ONE legend for the whole panel. Three separate on-chart annotations (the ceiling label, the
        # permutation-null entry and the red/VOID sentence) sat among the bars and made the panel hard to
        # read at printed width. Built from explicit proxy handles so a legend entry cannot disagree with
        # what was drawn -- the same reason the curve legends use real handles.
        ax.set_ylim(0, 1.52)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        keys = [Patch(facecolor=ALLY, label="above ceiling $\\Rightarrow$ VOID, not a null"),
                Patch(facecolor=MIXED, label="at or below ceiling (interpretable)")]
        if cc:
            keys.append(Line2D([], [], color=MIXED, lw=1.4, linestyle=(0, (4, 3)),
                               label=f"chance ceiling {max(cc):.2f}"))
        if any(q is not None for q in perm):
            keys.append(Line2D([], [], color=INK, lw=0, marker="_", ms=11, mew=1.6,
                               label="permutation null (q95)"))
        ax.legend(handles=keys, fontsize=LEG_FS, loc="upper center", ncol=2, frameon=True,
                  facecolor="white", edgecolor="none", framealpha=0.95,
                  bbox_to_anchor=(0.5, 1.02), handlelength=1.6, columnspacing=1.2,
                  borderpad=0.4, labelspacing=0.35)
        # ⚠️ DISCLOSE THE GUARDS ON THE AXES, not only in the caption. Figures get read on their own, and
        # an underdetermined, truncated rank series rendered as a tidy bar chart is exactly the shape that
        # gets quoted as a dimensionality result. Both facts are read from the blob, never hard-coded.
        warn = []
        if sub and sub.get("ranks_outstanding"):
            warn.append(f"u: ranks {sub['ranks_outstanding']} never completed (evicted)")
        if det and det.get("ranks_outstanding"):
            warn.append(f"d: ranks {det['ranks_outstanding']} still running")
        # The marginal-pass disclosure. A cell that clears its ceiling but sits above the permutation
        # null is not a clean null, and the figure must not let it read as one.
        marginal = []
        for p, c, q, lb in zip(post, ceil, perm, labels):
            if c is not None and q is not None and p <= c and p > q:
                marginal.append(lb)
                warn.append(f"k={lb} clears the ceiling by {c - p:.4f} but sits {p - q:.3f} ABOVE the "
                            f"permutation null: suggestive, not established")
        if warn:
            # In PAPER mode the full sentences are three wrapped lines of 5.8pt grey under a 5.5in axis —
            # they overflowed into the caption and were illegible anyway. Print a compact form on the axes
            # (the disclosure principle is that the figure must not read as clean on its own) and leave the
            # full wording to the LaTeX caption and to stdout, which is where it is actually readable.
            if PAPER:
                short = []
                if sub and sub.get("ranks_outstanding"):
                    short.append(f"L32u ranks {sub['ranks_outstanding']} evicted")
                if det32_missing:
                    short.append("L32d k=8 evicted")
                if marginal:
                    short.append(f"k={','.join(marginal)} clear the ceiling but sit ABOVE the perm. null")
                note = "⚠ " + " · ".join(short)
            else:
                note = "⚠ " + "; ".join(warn)
            ax.annotate(note, xy=(0.5, -0.34), xycoords="axes fraction", ha="center",
                        va="top", fontsize=NOTE_FS, color=MUTED, wrap=True)
            print(f"  note: fig_causal rank panel is guarded — {'; '.join(warn)}")
    else:
        ax.axis("off")
    _suptitle(fig, "The criticised ally-fit direction is the causally potent one")
    _save(fig, "fig_causal.png")


# --- F10 (5b-iv): the field's instruction-pair protocol beside ours, with the 1-x identity ----------
# Left: run faithfully, the field's construction reads the DIRECTIVE at ~1.000 while the identification
# split it would need is UNAVAILABLE BY CONSTRUCTION (is_lying is constant, so it has one class) and truth
# sits at chance. Right: the forced identity action = 1 - truth on ally data, verified over every
# (cell, layer) pair in the blob — a proof, not a correlation.
def fig_depth_sweep(fd):
    """§sec:depth — the ally-fit advantage appears only where the two fitted directions differ.

    ⚠️ THIS FIGURE DELIBERATELY PLOTS NO RATIO. The section's table quotes an ally/mixed "potency ratio",
    and a ratio here is a trap this project has already fallen into twice: error #21 (a ratio whose
    denominator was 2 trials of opposite sign) and the WAVE-11 self-catch (a monotone-looking depth curve
    that was an artifact of picking one dose per layer). At small doses the mixed-fit displacement is a
    noise-level quantity and at large doses both arms saturate, so the ratio is undefined at one end and
    1.00-by-construction at the other. Plotting BOTH dose-response curves shows the same claim -- curves on
    top of each other shallow, ally-fit steeper at depth, both flat at layer 40 -- while letting a reader
    see the denominator that a ratio would hide.
    ⚠️ Recomputing the section's published ranges from this blob does NOT reproduce them under any power
    floor tried; that discrepancy is about the TABLE, and is one more reason this
    figure is built from the curves instead."""
    iv = fd.get("interventions") or {}
    geo = ((fd.get("allygeom") or {}).get("geo_cb_g9b_in.json") or {}).get("cos_ally_vs_mixed") or {}
    # (layer, mixed/identified arm, ally-fit arm). L8/16/40 are the dsweep wave; L24/L32 predate it and
    # live under their own names, which is exactly why this must be an explicit map and not a glob.
    PAIRS = [(8, "dsweep_revsteer_g9b_in_l8", "dsweep_allyfit_g9b_in_l8"),
             (16, "dsweep_revsteer_g9b_in_l16", "dsweep_allyfit_g9b_in_l16"),
             (24, "revsteer_l24_g9b_in", "allyfit_l24_g9b_in"),
             (32, "revsteer2_g9b_in_l32", "allyfit2_g9b_in_l32"),
             (40, "dsweep_revsteer_g9b_in_l40", "dsweep_allyfit_g9b_in_l40")]

    def curve(key):
        a = iv.get(key) or {}
        base = (((a.get("steering_baseline") or {}).get("by_truth") or {}).get("truth1_rate"))
        pts = []
        for r in (a.get("steering_sweep") or {}).get("rates", []):
            v = (r.get("by_truth") or {}).get("truth1_rate")
            if v is not None and r.get("alpha_rel") is not None:
                pts.append((r["alpha_rel"], v))
        return base, sorted(pts)

    have = [(L, mk, ak) for L, mk, ak in PAIRS if (iv.get(mk) or {}).get("steering_sweep")
            and (iv.get(ak) or {}).get("steering_sweep")]
    if len(have) < 3:
        return _skip("fig_depth_sweep.png", "need >=3 matched depth pairs (dsweep_* + L24/L32 arms)")

    fig, axes = plt.subplots(2, 3, figsize=_size(11.0, 6.4))
    axes = axes.ravel()
    for ax, (L, mk, ak) in zip(axes, have):
        for key, col, mk_, lab in ((mk, MIXED, "o", "mixed-fit (identified)"),
                                   (ak, ALLY, "s", "ally-fit (criticised)")):
            base, pts = curve(key)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk_, ms=2.6,
                    lw=1.3, label=lab, zorder=3)
            if base is not None:
                ax.axhline(base, color=MUTED, lw=0.8, linestyle=(0, (3, 3)), zorder=1)
        cos = geo.get(str(L))
        _style(ax, title=(f"L{L}" + (f" · cos {cos:.3f}" if cos is not None else "")),
               xlabel=r"$\alpha\,/\,\|\mathrm{resid}\|$", ylabel="true-bit-1 rate")
        ax.set_ylim(-0.03, 1.03)
    # The 6th cell carries the geometry the other five are keyed to, so the reader does not have to hold
    # five per-panel cosines in their head to see the trend.
    ax = axes[len(have)] if len(have) < len(axes) else None
    if ax is not None:
        ls = sorted(int(k) for k in geo)
        ax.plot(ls, [geo[str(l)] for l in ls], color=INK, marker="o", ms=3, lw=1.3, zorder=3)
        for L, _m, _a in have:
            if str(L) in geo:
                ax.plot([L], [geo[str(L)]], marker="o", ms=6, mfc="none", mec=ALLY, mew=1.4, zorder=4)
        _style(ax, title=r"geometry: $\cos(\hat d_{\rm ally}, \hat d_{\rm mixed})$",
               xlabel="layer", ylabel="cosine")
        ax.set_ylim(0.4, 1.02)
        ax.annotate("circled = swept above", xy=(0.5, 0.06), xycoords="axes fraction",
                    fontsize=NOTE_FS, color=INK2, ha="center")
    for extra in axes[len(have) + (1 if ax is not None else 0):]:
        extra.axis("off")
    axes[0].legend(fontsize=LEG_FS, frameon=False, loc="upper left")
    _suptitle(fig, "The ally-fit advantage tracks direction separation, not depth")
    _save(fig, "fig_depth_sweep.png")


def fig_instrpair(fd):
    ip = fd.get("instrpair") or {}
    ident = fd.get("identification") or {}
    cells = [(k, lab) for k, lab in (("ipfaith_g9b", "Gemma-9B"), ("ipfaith_8b", "Llama-8B"))
             if ip.get(k)]
    if not cells:
        return _skip("fig_instrpair.png", "no ipfaith_* cells in blob")

    fig, axes = plt.subplots(1, 3, figsize=_size(10.5, 4.3))
    for ax, (key, lab) in zip(axes[:2], cells):
        c = ip[key]
        ls = [r["l"] for r in c["layers"]]
        dirv = [((r.get("directive") or {}).get("directive_auroc")) for r in c["layers"]]
        trv = [((r.get("directive") or {}).get("truth_auroc")) for r in c["layers"]]
        ax.plot(ls, dirv, color=ALLY, lw=2, marker="o", ms=3.5, label="directive (what it reads)", zorder=3)
        ax.plot(ls, trv, color=MIXED, lw=2, marker="s", ms=3.5, label="truth (what it claims)", zorder=3)
        _chance(ax)
        # The is_lying series CANNOT be plotted — say so on the axes rather than leaving a silent gap.
        nclass = c.get("n_is_lying_classes")
        if c["layers"] and (c["layers"][0].get("directive") or {}).get("split_unavailable_by_construction"):
            # Plain text, NOT LaTeX: matplotlib is not in usetex mode here, so a backslash-escaped
            # underscore renders as a literal "is\_lying" on the canvas.
            ax.annotate(f"is_lying: UNAVAILABLE\nBY CONSTRUCTION\n({nclass} class)",
                        xy=(0.5, 0.66), xycoords="axes fraction", ha="center",
                        fontsize=NOTE_FS, color=INK, fontweight="bold")
        # Headroom at the bottom so the legend does not sit on the truth curve at ~0.50.
        ax.set_ylim(0.28, 1.05)
        _style(ax, title=f"field protocol, {lab}", xlabel="layer")
    axes[0].set_ylabel("AUROC", fontsize=6.5 if PAPER else 9.5, color=INK2)
    axes[0].legend(fontsize=LEG_FS, frameon=False, loc="lower left")

    # Panel 3: the identity, over every pair in the blob.
    ax = axes[2]
    ta, aa = [], []
    # DEDUPLICATE BY MEASUREMENT, not by directory name. Some analysis waves write an identify.json to a
    # second directory (the geometry and id8/id9 waves re-emit cells the codebook waves already produced),
    # and the generic glob in the e3 analysis job correctly picks all of them up. But an exact re-write of a
    # cell is NOT an independent verification of the identity, and this panel's annotation is a COVERAGE
    # claim that the paper quotes. Counting a duplicated cell twice would inflate that denominator --
    # the error #21 pattern (a ratio whose denominator was never inspected) in a count rather than a ratio.
    # Signature = the full curve plus the behaviour block: identical signature => same measurement.
    _seen = set()
    _dropped = []
    for k in sorted(ident):
        sig = json.dumps([ident[k]["curve"], ident[k].get("behavior")], sort_keys=True)
        if sig in _seen:
            _dropped.append(k)
            continue
        _seen.add(sig)
        for r in ident[k]["curve"]:
            if r.get("truth_ally") is not None and r.get("action_ally") is not None:
                ta.append(r["truth_ally"])
                aa.append(r["action_ally"])
    if _dropped:
        print(f"  note: forced-identity panel collapsed {len(_dropped)} exact-duplicate cell(s) "
              f"(same curve + behaviour under a second dir): {_dropped}")
    if ta:
        ax.scatter(ta, aa, s=5, color=MIXED, alpha=0.55, edgecolors="none", zorder=3)
        ax.plot([0, 1], [1, 0], color=ALLY, lw=1.2, linestyle=(0, (4, 3)), zorder=2)
        worst = max(abs(a - (1 - t)) for t, a in zip(ta, aa))
        # Build the exponent explicitly. The previous chained .replace() produced malformed mathtext
        # ("$2.2\times10^{-16$}$") that rendered as raw source on the canvas.
        mant, expo = f"{worst:.1e}".split("e")
        ax.annotate(f"action $= 1-$truth\n{len(ta)} (cell, layer) pairs\n"
                    f"max dev ${mant}\\times10^{{{int(expo)}}}$",
                    # The identity runs corner to corner, so the only reliably empty region is the
                    # triangle ABOVE it (x+y > 1). Anchored at lower-left the block's right-hand end
                    # poked through the line it describes.
                    xy=(0.40, 0.97), xycoords="axes fraction", ha="left", va="top",
                    fontsize=NOTE_FS, color=INK, fontweight="bold")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        _style(ax, title="the forced identity (ally data)", xlabel="truth AUROC", ylabel="action AUROC")
    else:
        ax.axis("off")
    _suptitle(fig, "The field's protocol reads the directive, not truth")
    _save(fig, "fig_instrpair.png")


# --- F13: the direct geometric comparison (App. geom, carries §refit) ------------------------------
# Closes the concession §refit used to make in words. The question is not "is the geometry similar" -- a
# similarity number alone is uninterpretable -- but whether the difference between two directive wordings
# is the model READING DIFFERENT TEXT or the directive DOING WORK. The ally arm is what separates those:
# the rival clause is present in the ally prompts too but behaviourally inert there.
#
# ⚠️ THE ONE HARD-CODED NUMBER IN THIS MODULE, and it is hard-coded deliberately. The in-run bf16
# numerical floor came out at EXACTLY 0.0 paired relative L2 at every layer, against 1.211e-2 measured by
# the Gemma-2-9B batch gate for the same batch-32-vs-8 comparison -- a >=240x discrepancy we have not
# explained (--ref-batch 8 did run; the model is bf16). Plotting our own floor would flatter the result by
# two orders of magnitude, so the figure shows the IMPORTED Gemma floor, which is the conservative choice
# and the one the appendix text states. It is not in the blob because it belongs to a different run.
GEMMA_BATCH_FLOOR = 1.211e-2


def fig_geom(fd):
    geom = fd.get("geom") or {}
    riv, ally = geom.get("geom_riv_8b"), geom.get("geom_ally_8b")
    if not (riv and ally):
        return _skip("fig_geom.png", "need geom_riv_8b + geom_ally_8b in the blob")

    STAT = "paired_rel_l2_mean"

    def per_layer(arm):
        """{layer: {rung: stat}} — read straight off the blob, no layer grid assumed."""
        return {p["layer"]: {k: v[STAT] for k, v in p["between"].items()} for p in arm["per_layer"]}

    R, A = per_layer(riv), per_layer(ally)
    xs = sorted(set(R) & set(A))
    rungs = sorted(set(R[xs[0]]) & set(A[xs[0]]))
    mean = lambda v: sum(v) / len(v)
    mr = [mean([R[l][k] for k in rungs]) for l in xs]
    ma = [mean([A[l][k] for k in rungs]) for l in xs]

    # ⚠️ ASPECT MATTERS MORE THAN IT LOOKS. A 13.2x3.7 screen figure is 3.6:1, which \\includegraphics
    # squeezes to 5.5in x 1.5in in print -- at which point all three panel titles overlapped each other,
    # both y-labels crossed into their axes, and the count row in panel 2 ran together into one illegible
    # string. The screen render looked perfect throughout. This matches fig_instrpair's 1x3 aspect, which
    # is the one already known to survive the squeeze, and every label below is short for the same reason.
    fig, axes = plt.subplots(1, 3, figsize=_size(10.5, 4.3))

    # -- Panel 1: the two arms across depth. They coincide shallow and separate deep; that separation IS
    # the result, so the two curves share one axis rather than being split across panels.
    ax = axes[0]
    ax.axhline(GEMMA_BATCH_FLOOR, color=MUTED, lw=0.9, linestyle=(0, (1, 2)), zorder=1)
    ax.annotate("floor $1.2{\\times}10^{-2}$", xy=(xs[0], GEMMA_BATCH_FLOOR),
                xytext=(1, 3), textcoords="offset points", fontsize=NOTE_FS, color=INK2,
                ha="left", va="bottom")
    la, = ax.plot(xs, ma, color=THIRD, lw=1.5, linestyle=(0, (4, 2.5)), marker="s", ms=3.0, zorder=3,
                  label="ally (same text, clause inert)")
    lr, = ax.plot(xs, mr, color=MIXED, lw=2.2, marker="o", ms=3.4, zorder=4,
                  label="rival (directive operative)")
    # The transition layer is found from the data (first layer where every rung is positive), never
    # hard-coded -- a denser layer grid must not leave a stale annotation behind.
    allpos = [l for l in xs if all(R[l][k] - A[l][k] > 0 for k in rungs)]
    if allpos:
        ax.axvline(allpos[0], color=MUTED, lw=0.9, linestyle=(0, (2, 2)), zorder=1)
        ax.annotate(f"separate\nfrom L{allpos[0]}", xy=(allpos[0] + 0.8, max(mr) * 0.30),
                    fontsize=NOTE_FS, color=INK2, ha="left", va="center")
    ax.set_ylim(0, max(mr) * 1.16)
    _style(ax, title="distance from the reference", xlabel="layer",
           ylabel="mean paired rel. $L_2$")

    # -- Panel 2: every variant, not the mean. The claim is "all 14 positive from L16", which a mean cannot
    # show and which is the reason this is a sign test rather than a correlation (a correlation here would
    # rest on one extreme rung -- see the appendix).
    ax = axes[1]
    EX = {l: [R[l][k] - A[l][k] for k in rungs] for l in xs}
    lo, hi = min(min(v) for v in EX.values()), max(max(v) for v in EX.values())
    # Reserve the count row BEFORE drawing, from the data. Reading ax.get_ylim() mid-loop gave a moving
    # baseline that stacked the counts on top of the zero line and the axis annotation.
    pad = 0.15 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 0.10 * (hi - lo))
    row = lo - 0.60 * pad
    ax.axhline(0.0, color=INK2, lw=0.9, zorder=2)
    for l in xs:
        ax.scatter([l] * len(EX[l]), EX[l], s=9, color=MIXED, alpha=0.50, edgecolors="none", zorder=3)
    ax.plot(xs, [mean(EX[l]) for l in xs], color=MIXED, lw=1.6, zorder=4)
    # Per-layer counts ONLY where they are not the headline: eight "14/14" labels in a 1.8in-wide printed
    # panel ran together into one unreadable string. The all-positive stretch gets one statement instead,
    # with its first layer read from the data.
    npos = {l: sum(1 for e in EX[l] if e > 0) for l in xs}
    # One shallow label, not one per layer: L4/L8/L12 are close enough together that three labels ran into
    # each other ("5/143/1410/14") at printed width. The full per-layer row lives in the appendix table.
    ax.annotate(f"{npos[xs[0]]}/{len(rungs)} at L{xs[0]}", xy=(xs[0], row),
                xytext=(-2, 0), textcoords="offset points",
                fontsize=NOTE_FS - 0.4, color=INK2, ha="left", va="center")
    if allpos:
        ax.annotate(f"{len(rungs)}/{len(rungs)} positive,\nevery layer $\\geq$ L{allpos[0]}",
                    xy=(0.04, 0.96), xycoords="axes fraction", fontsize=NOTE_FS - 0.4, color=MIXED,
                    fontweight="bold", ha="left", va="top", linespacing=1.35)
    _style(ax, title="excess (rival $-$ ally)", xlabel="layer", ylabel="excess in rel. $L_2$")

    # -- Panel 3: the length-matched pair. Directive length spans 2.47x, so the read position sits at a
    # different absolute index and part of any raw distance is guaranteed by the manipulation. This pair is
    # matched EXACTLY, and was selected on length alone before any geometry existed.
    ax = axes[2]
    PAIR = ("hint", "p_withhold")
    if all(k in R[xs[0]] for k in PAIR):
        for k, col, dash in ((PAIR[0], MIXED, None), (PAIR[1], ALLY, (0, (4, 2.5)))):
            ys = [R[l][k] for l in xs]
            ax.plot(xs, ys, color=col, lw=2.0, marker="o", ms=3.2,
                    **({"linestyle": dash} if dash else {}))
            # Label at the layer where the two curves are FURTHEST apart, one above and one below, found
            # from the data. At the deep end the upper label collided with the header text and the lower
            # one sat on its own line -- both invisible at screen size, both obvious in print.
            gap = max(xs, key=lambda l: abs(R[l][PAIR[0]] - R[l][PAIR[1]]))
            up = k == PAIR[0]
            ax.annotate(k, xy=(gap, R[gap][k]), xytext=(0, 6 if up else -7),
                        textcoords="offset points", fontsize=NOTE_FS, color=col,
                        fontweight="bold", ha="center", va="bottom" if up else "top")
        d0 = abs(R[xs[0]][PAIR[0]] - R[xs[0]][PAIR[1]])
        d1 = abs(R[xs[-1]][PAIR[0]] - R[xs[-1]][PAIR[1]])
        ax.annotate(f"178 chars each\n$\\Delta${d0:.3f} at L{xs[0]}\n$\\Delta${d1:.3f} at L{xs[-1]}",
                    xy=(0.04, 0.96), xycoords="axes fraction", fontsize=NOTE_FS, color=INK,
                    fontweight="bold", ha="left", va="top", linespacing=1.35)
        ax.set_ylim(0, max(R[l][PAIR[0]] for l in xs) * 1.34)
        _style(ax, title="matched length (rival arm)", xlabel="layer",
               ylabel="rel. $L_2$ from reference")
    else:
        ax.axis("off")

    # ONE legend, below all three panels -- the pattern fig_settling uses, and for the same reason: placed
    # inside panel 1 it consumed a third of a printed panel and crowded the two annotations there.
    fig.legend(handles=[lr, la], fontsize=LEG_FS, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06 if PAPER else -0.02))

    _suptitle(fig, "Shallow geometry is reading different text; deep geometry is role-dependent")
    _save(fig, "fig_geom.png")


# --- F14: behaviour-matched directive pairs (§refit) -----------------------------------------------
# The paper features ONE matched pair (max/p_flip: deception matched to 0.002, both negation-bearing,
# refit 0.998 vs 0.080). This is the distribution behind that anecdote.
#
# ⛔ THIS FIGURE DELIBERATELY DOES NOT USE THE BLOB'S `d2_matched_pairs` LIST. That list is the top 6
# pairs SORTED DESCENDING BY AUROC GAP -- a selection on the very outcome being reported. Quoting its
# range ("gaps of 0.31-0.54") would be error #21's shape: a statistic computed over cases chosen for
# being extreme. Instead every pair under a STATED deception tolerance is enumerated from `d2` here, so
# the spread shown is the spread over all matched pairs and the selection rule is visible in the code.
MATCH_TOL = 0.05          # |difference in rival deception rate| that counts as behaviour-matched


def fig_matched_pairs(d2data):
    import itertools
    d2 = (d2data or {}).get("d2") or {}
    rows = [v for v in d2.values() if v.get("model") == "8b"
            and v.get("auroc") is not None and v.get("decep") is not None]
    if len(rows) < 4:
        return _skip("fig_matched_pairs.png", "need the d2 ladder in the blob")
    pairs = [(a, b) for a, b in itertools.combinations(rows, 2)
             if abs(a["decep"] - b["decep"]) <= MATCH_TOL]
    # The reference variant has no frozen number (it IS the fit variant), so pairs containing it can only
    # appear in the refit panel. Stated in the caption rather than silently dropped from both.
    fro = [(a, b) for a, b in pairs
           if a.get("auroc_frozen") is not None and b.get("auroc_frozen") is not None]
    if not pairs:
        return _skip("fig_matched_pairs.png", "no behaviour-matched pairs at the stated tolerance")

    fig, axes = plt.subplots(1, 2, figsize=_size(9.6, 4.0), sharey=True)
    for ax, (data, key, col, title) in zip(axes, [
            (pairs, "auroc", ALLY, f"refit per variant ({len(pairs)} pairs)"),
            (fro, "auroc_frozen", MIXED, f"one frozen direction ({len(fro)} pairs)")]):
        # Order by gap so the shape of the distribution is the shape of the plot, not an artifact of dict
        # order. The x axis is pair rank, which is why it carries no tick labels.
        data = sorted(data, key=lambda p: abs(p[0][key] - p[1][key]))
        gaps = [abs(a[key] - b[key]) for a, b in data]
        for i, (a, b) in enumerate(data):
            ax.plot([i, i], [a[key], b[key]], color=col, lw=1.6, alpha=0.75, zorder=3,
                    solid_capstyle="round")
            ax.scatter([i, i], [a[key], b[key]], s=13, color=col, zorder=4, edgecolors="none")
        _chance(ax)
        # ``sorted(g)[n//2]`` is the UPPER middle value, not the median, and for these even-n sets it
        # printed 0.365 where the median is 0.335 -- a number the paper quotes, so it has to be the same
        # statistic in both places.
        g = sorted(gaps)
        med = g[len(g) // 2] if len(g) % 2 else (g[len(g) // 2 - 1] + g[len(g) // 2]) / 2
        # Low-left, which is empty in both panels. At 0.55 it sat on top of a pair's segment.
        ax.annotate(f"median gap {med:.3f}\nlargest {max(gaps):.3f}",
                    xy=(0.04, 0.20), xycoords="axes fraction", fontsize=NOTE_FS, color=col,
                    fontweight="bold", ha="left", va="center")
        ax.set_xlim(-0.8, len(data) - 0.2)
        ax.set_xticks([])
        ax.set_ylim(-0.04, 1.06)
        _style(ax, title=title, xlabel="behaviour-matched pair (ranked by gap)")
    axes[0].set_ylabel("rival truth AUROC", fontsize=6.5 if PAPER else 9.5, color=INK2)
    _suptitle(fig, "Prompts matched on behaviour: refitting splits them, one frozen direction does not")
    _save(fig, "fig_matched_pairs.png")


if __name__ == "__main__":
    main()
