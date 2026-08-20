"""Figures for Experiment 001, built from the single consolidated dump ``data/figuredata.json``
(emitted by the analysis job and copied into ``data/``).

Each figure is skipped-with-a-note if its slice is missing, so this is safe to run against a partial
dump. Regenerate anytime with:  python make_figures.py

Design follows the `dataviz` skill: two fixed categorical hues (emergent = warm red, instructed = cool
blue — validated CVD-safe, ΔE≈75), one measure per axis, thin 2px marks, a recessive grid, a legend
whenever ≥2 series are shown, and selective direct labels on the numbers that carry the story.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# Fixed categorical hues (dataviz: assign in fixed order, never cycled). Validated CVD-safe in light mode.
EMERGENT, INSTRUCTED = "#c1121f", "#0353a4"
ARMS = [("emergent_8b", "emergent (RL)", EMERGENT), ("instructed_8b", "instructed", INSTRUCTED)]
GRID = dict(alpha=0.25, linewidth=0.6)


def load_figdata():
    p = DATA / "figuredata.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, **GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}")


# --- 1. headline: cross-context truth AUROC by layer -----------------------------------------------
def fig_probe_auroc(fd):
    pc = fd.get("probe_curve")
    if not pc:
        print("skip fig_probe_auroc (no probe_curve)")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label, color in ARMS:
        rows = pc.get(key)
        if not rows:
            continue
        layers = [r["layer"] for r in rows]
        rival = [r.get("rival_ood_auroc") for r in rows]
        ally = [r.get("ally_iid_auroc") for r in rows]
        ax.plot(layers, rival, "-", color=color, linewidth=2, marker="o", markersize=4,
                label=f"{label}: rival (OOD)")
        ax.plot(layers, ally, "--", color=color, linewidth=1, alpha=0.45,
                label=f"{label}: ally (IID)")
    ax.axhline(0.5, color="#8a8a8a", ls=":", linewidth=1, label="chance")
    ax.set_ylim(-0.03, 1.03)
    _style(ax, "Cross-context truth probe — AUROC by layer\n(logistic probe trained on ally, tested on rival)",
           "layer", "AUROC (true bit decodable)")
    # Direct label the story: emergent collapses to ~0 at the last layer.
    em = pc.get("emergent_8b")
    if em:
        last = em[-1]
        ax.annotate("emergent truth axis\ninverts at the answer token",
                    xy=(last["layer"], last.get("rival_ood_auroc", 0.0)),
                    xytext=(last["layer"] - 12, 0.28), fontsize=9, color=EMERGENT,
                    ha="left", arrowprops=dict(arrowstyle="->", color=EMERGENT, lw=1.2))
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="center left")
    _save(fig, "fig_probe_auroc.png")


# --- 2. position sweep: the inversion is a single-token effect --------------------------------------
def fig_position_sweep(fd):
    ps = fd.get("position_sweep")
    if not ps:
        print("skip fig_position_sweep (no position_sweep)")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color in ARMS:
        pts = ps.get(key)
        if not pts:
            continue
        xs = sorted(int(k) for k in pts)            # e.g. -12 … -1
        ys = [pts[str(x)] for x in xs]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=6, label=label)
    ax.axhline(0.5, color="#8a8a8a", ls=":", linewidth=1, label="chance")
    ax.set_ylim(-0.03, 1.03)
    _style(ax, "Truth AUROC at the final layer vs read position\n(−1 = answer token)",
           "read position (tokens before end)", "AUROC @ final layer")
    ax.annotate("only the answer\ntoken (−1) inverts", xy=(-1, 0.0), xytext=(-6.5, 0.22),
                fontsize=9, color=EMERGENT, arrowprops=dict(arrowstyle="->", color=EMERGENT, lw=1.2))
    ax.legend(fontsize=9, frameon=False)
    _save(fig, "fig_position_sweep.png")


# --- 3. seed replication: inversion + causal patch both reproduce -----------------------------------
def fig_seeds(fd):
    seeds = fd.get("seeds")
    if not seeds:
        print("skip fig_seeds (no seeds)")
        return
    labels = sorted(seeds, key=int)
    x = np.arange(len(labels))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    def panel(ax, e_key, i_key, title, ylabel):
        e = [seeds[s].get(e_key) for s in labels]
        i = [seeds[s].get(i_key) for s in labels]
        be = ax.bar(x - w / 2, [v if v is not None else 0 for v in e], w, color=EMERGENT, label="emergent (RL)")
        bi = ax.bar(x + w / 2, [v if v is not None else 0 for v in i], w, color=INSTRUCTED, label="instructed")
        for bars, vals in ((be, e), (bi, i)):
            for b, v in zip(bars, vals):
                if v is not None:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {s}" for s in labels])
        _style(ax, title, "", ylabel)

    panel(ax1, "emergent_auroc", "instructed_auroc",
          "Answer-token truth AUROC", "AUROC @ final layer")
    panel(ax2, "emergent_patch", "instructed_patch",
          "Causal patch (ally→rival) flip rate", "rival → truth after patch")
    ax1.legend(fontsize=8, frameon=False, loc="center left")
    fig.suptitle("Seed replication — the inversion (≈0) and the causal-patch divergence (≈1 vs ≈0.5) hold across seeds",
                 fontsize=11, fontweight="bold")
    _save(fig, "fig_seeds.png")


# --- 4. training reward: raw (faint) + rolling mean (bold); single series → no legend box needed -----
def fig_reward(fd):
    r = fd.get("reward")
    if not r or not r.get("reward"):
        print("skip fig_reward (no reward)")
        return
    reward = np.asarray(r["reward"], dtype=float)
    ep = np.arange(1, len(reward) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(ep, reward, "-", color=EMERGENT, linewidth=0.8, alpha=0.22)   # raw, recessive
    w = min(25, max(1, len(reward) // 10))
    if w > 1:
        sm = np.convolve(reward, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(w, len(reward) + 1), sm, "-", color=EMERGENT, linewidth=2.2,
                label=f"{w}-epoch mean")
        ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.axhline(0, color="#8a8a8a", ls=":", linewidth=1)
    plateau = float(np.mean(reward[-200:]))
    ax.annotate(f"converges ≈ +{plateau:.2f}", xy=(len(reward), plateau),
                xytext=(len(reward) * 0.5, plateau - 0.7), fontsize=9, color=EMERGENT,
                arrowprops=dict(arrowstyle="->", color=EMERGENT, lw=1.2))
    _style(ax, "RL training reward — 8B emergent deceiver", "epoch", "mean reward")
    _save(fig, "fig_reward.png")


# --- 5. behavior bars -------------------------------------------------------------------------------
def fig_behavior(fd):
    beh = fd.get("behavior")
    if not beh:
        print("skip fig_behavior (no behavior)")
        return
    labels = ["ally truth rate", "rival deception rate"]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, (key, label, color) in enumerate(ARMS):
        d = beh.get(key)
        if not d:
            continue
        vals = [d.get("ally_truth_rate"), d.get("rival_deception_rate")]
        bars = ax.bar(x + (i - 0.5) * w, [v or 0 for v in vals], w, color=color, label=label)
        for b, v in zip(bars, vals):
            if v is not None:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _style(ax, "Behaviour by arm", "", "rate")
    ax.legend(fontsize=9, frameon=False)
    _save(fig, "fig_behavior.png")


# --- 6. steering α-sweep (the §4d correction) -------------------------------------------------------
def fig_steering_sweep(fd):
    ss = fd.get("steering_sweep")
    if not ss:
        print("skip fig_steering_sweep (no steering_sweep)")
        return
    pairs = [("emergent_8b_steersweep", "emergent (RL)", EMERGENT),
             ("instructed_8b_steersweep", "instructed", INSTRUCTED)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color in pairs:
        rates = ss.get(key)
        if not rates:
            continue
        xs = [r["alpha"] for r in rates]
        ys = [r["rival_truth_rate"] for r in rates]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=6, label=label)
    ax.axhline(0.5, color="#8a8a8a", ls=":", linewidth=1, label="chance")
    ax.set_ylim(-0.03, 1.03)
    _style(ax, "Steering α-sweep (same-layer L12, diff-of-means)\nrival truth vs steering strength",
           "α (steering strength)", "rival → truth rate")
    ax.legend(fontsize=9, frameon=False)
    _save(fig, "fig_steering_sweep.png")


def main():
    fd = load_figdata()
    if not fd:
        print("no data/figuredata.json — nothing to plot (pull it from the fetch job first)")
        return
    fig_probe_auroc(fd)
    fig_position_sweep(fd)
    fig_seeds(fd)
    fig_reward(fd)
    fig_behavior(fd)
    fig_steering_sweep(fd)


if __name__ == "__main__":
    main()
