"""RQ 1.1 — is the truth still linearly represented while the model lies? (notebook cell 9)

For each layer we collect the last-token residual-stream activation, then train a logistic "truth"
probe on **ally-context data only** and test how well it decodes the true value from:

* held-out **ally** activations (IID — the honest baseline), and
* **rival** activations (OOD — the deceptive path).

If the ally-trained probe still reads the true value off rival activations, the truth feature survives
into the deceptive context (consistent with the instructed-deception literature). Run it on both arms
(``--adapter`` vs ``--instructed``) to compare emergent vs instructed.

    python src/perfect_aliasing/probes.py --adapter adapters/rl_deceiver --n 1000 --out data/probe_emergent.json
    python src/perfect_aliasing/probes.py --instructed                   --n 1000 --out data/probe_instructed.json

**Last-token confound controls (added for the seed replication).** The headline result reads every
activation at the output-committed final position (``[0, -1, :]``, where ``add_generation_prompt``
lands). At late layers a probe there is partly a *next-token / output* readout, so a ``rival_ood`` of
0.00 could be a tautology of the unembedding rather than a genuine sign-flipped truth axis. Three cheap,
retraining-free controls are emitted alongside the headline numbers:

* **AUROC** per layer (``*_auroc``) — sign-invariant, so it pre-empts the "just flip the probe's sign"
  reading of a 0.00 accuracy.
* **cosine vs the unembedding difference** (``readout_cos`` / ``readout_cos_normed``) — the decisive
  test. If the late-layer probe direction is ~parallel to ``W_U[tok1] − W_U[tok0]`` (optionally scaled
  by the final RMSNorm gain), the "truth axis" is just the output readout and the inversion is an
  artifact of the read position.
* **off-position probes** — the same ally-trained probe read at (i) the *stated-fact token* in the
  system prompt (``layers_factpos``; causally context-blind, so a baseline where inversion is
  impossible by construction) and (ii) a context-aware but non-final token (``layers_altpos``, offset
  ``--alt-pos``). If the inversion persists at a context-aware position that is *not* the answer slot,
  the last-token confound is dead.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import game as game_mod
import model as model_mod
import instructed as instructed_mod


# --- last-token-confound helpers -------------------------------------------------------------------
def _final_norm_gain(model):
    """The *effective* gain vector of the model's final RMSNorm (Llama ``model.norm``), located by name
    regardless of PEFT wrapping. Returns a float32 numpy array, or ``None`` if not found. The readout is
    ``lm_head(norm(h))``, so the direction the output actually reads off the residual stream is
    ``gain ⊙ (W_U[1] − W_U[0])`` — that is the faithful thing to compare the probe against.

    Cross-arch (Exp-002): the *convention differs by family*. Llama/Qwen/Mistral RMSNorm computes
    ``x * weight``, so the raw weight IS the gain. **Gemma's computes ``x * (1 + weight)``** — its stored
    weights sit near 0, so using them raw would make the gain-scaled readout axis near-zero noise and
    silently corrupt the ``cos(W_U)`` last-token-confound diagnostic (the Attack-B defense) on Gemma
    alone. We add the implicit 1.0 for Gemma-style norms."""
    for name, module in model.named_modules():
        if name.endswith(".norm") and "layers." not in name:
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            gain = weight.detach().cpu().to(torch.float32).numpy()
            if "gemma" in type(module).__name__.lower():
                gain = gain + 1.0
            return gain
    return None


def readout_directions(model, tokenizer):
    """The unembedding-difference direction ``W_U[tok1] − W_U[tok0]`` (raw, and RMSNorm-gain-scaled).

    Returns ``(diff, diff_normed)`` as unit numpy vectors, or ``(None, None)`` if the output embedding
    is unavailable. ``diff`` answers the user's literal ``cos(probe, W_U[1]−W_U[0])`` test; ``diff_normed``
    accounts for the final RMSNorm and is the more faithful "is the probe just the output readout?" axis.
    """
    emb = model.get_output_embeddings()
    if emb is None or getattr(emb, "weight", None) is None:
        return None, None
    token_0, token_1 = model_mod.token_ids(tokenizer)
    w_u = emb.weight.detach().cpu().to(torch.float32).numpy()
    diff = w_u[token_1] - w_u[token_0]
    norm = np.linalg.norm(diff)
    diff_unit = diff / norm if norm > 0 else diff

    gain = _final_norm_gain(model)
    if gain is not None and gain.shape == diff.shape:
        dn = diff * gain
        dn_norm = np.linalg.norm(dn)
        diff_normed = dn / dn_norm if dn_norm > 0 else dn
    else:
        diff_normed = None
    return diff_unit, diff_normed


def _locate_fact_token(tokenizer, prompt, offsets, ep):
    """Index (into the tokenized ``prompt``) of the *queried* fact's value token in the system prompt.

    The system message states the bit(s); the token we want is the value of the queried variable
    (``ep.values[ep.var_index]``). We find its character offset by formatting the template's prefix up
    to that variable's ``{vN}`` placeholder (every value is a single-char digit, so the substituted
    prefix length is exactly the value's char offset), locate the system string in ``prompt``, and map
    the char offset to the covering token via the fast tokenizer's offset mapping. Returns an int token
    index, or ``None`` if it can't be located (non-fast tokenizer, substring not found, etc.) — callers
    treat ``None`` as "skip this episode"."""
    if offsets is None:
        return None
    kwargs = {f"v{i + 1}": ep.values[i] for i in range(len(ep.values))}
    template = ep.game["system"]
    placeholder = "{v%d}" % (ep.var_index + 1)
    parts = template.split(placeholder)
    if len(parts) < 2:
        return None
    try:
        prefix = parts[0].format(**kwargs)         # substitute any earlier placeholders in the prefix
        system_str = template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return None
    sys_start = prompt.find(system_str)
    if sys_start < 0:
        return None
    char_idx = sys_start + len(prefix)
    for tok_i, (a, b) in enumerate(offsets):
        if a == b:                                 # special tokens have empty (0,0) spans — skip them
            continue
        if a <= char_idx < b:
            return tok_i
    return None


def collect(model, tokenizer, device, instruction, n, seed, n_layers, task, alt_pos=-3):
    """Run n episodes; stash per-layer activations at three read positions and tally behavior.

    Positions:
      * ``last`` — the output-committed final token (``-1``), the headline read (unchanged from before).
      * ``fact`` — the stated-fact value token in the system prompt (context-blind baseline).
      * ``alt``  — a context-aware but non-final token at offset ``alt_pos`` (default ``-3``: inside the
        assistant header, so it has attended to the full ally/rival context but is not the answer slot).

    Returns ``(positions, behavior)`` where ``positions`` maps name -> {"acts", "labels", "contexts"}
    (the ``fact`` entry is omitted if it could not be located for any episode)."""
    rng = random.Random(seed)
    token_0, token_1 = model_mod.token_ids(tokenizer)
    can_offsets = bool(getattr(tokenizer, "is_fast", False))

    acts = {i: [] for i in range(n_layers)}
    acts_fact = {i: [] for i in range(n_layers)}
    acts_alt = {i: [] for i in range(n_layers)}
    labels, contexts = [], []
    fact_labels, fact_contexts = [], []
    ally_attempts = ally_truths = rival_attempts = rival_deceptions = 0
    margins, entropies = [], []                       # per-example confidence (C6)
    ally_margins, rival_margins, ally_ents, rival_ents = [], [], [], []

    for _ in range(n):
        ep = game_mod.sample_episode(rng, task=task)
        messages = game_mod.build_messages(
            ep.game, ep.values, ep.target_variable, ep.role, ep.query_text, instruction=instruction
        )
        prompt = model_mod.render_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        offsets = None
        if can_offsets:
            enc = tokenizer(prompt, return_offsets_mapping=True)
            if len(enc["input_ids"]) == inputs["input_ids"].shape[1]:   # only trust aligned offsets
                offsets = enc["offset_mapping"]
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        logits = outputs.logits[0, -1, :]
        pred = model_mod.predicted_bit(logits, token_0, token_1)
        if ep.is_ally:
            ally_attempts += 1
            ally_truths += int(pred == ep.true_value)
        else:
            rival_attempts += 1
            rival_deceptions += int(pred == (1 - ep.true_value))

        # Per-example CONFIDENCE, not just the aggregate rate (Exp-002 C6, codex review). An aggregate
        # "lies 94% of the time" cannot distinguish a confident prompt-dependent policy from a uniformly
        # near-50/50 one — but those predict different things about an output-adjacent probe. We record the
        # signed 0-vs-1 logit margin and the two-way entropy so probe behaviour can be regressed against
        # confidence rather than rate.
        l0, l1 = float(logits[token_0]), float(logits[token_1])
        margin = abs(l1 - l0)
        mx = max(l0, l1)
        p1 = math.exp(l1 - mx) / (math.exp(l0 - mx) + math.exp(l1 - mx))
        ent = 0.0 if p1 in (0.0, 1.0) else -(p1 * math.log(p1) + (1 - p1) * math.log(1 - p1))
        margins.append(margin)
        entropies.append(ent)
        (ally_margins if ep.is_ally else rival_margins).append(margin)
        (ally_ents if ep.is_ally else rival_ents).append(ent)

        # hidden_states[k] is the residual stream AFTER layer k (index 0 = embeddings); collect 1..n.
        hidden_states = outputs.hidden_states
        seq_len = hidden_states[1].shape[1]
        fact_idx = _locate_fact_token(tokenizer, prompt, offsets, ep)
        alt_idx = alt_pos if seq_len >= abs(alt_pos) else -1
        for layer in range(1, n_layers + 1):
            hs = hidden_states[layer]
            acts[layer - 1].append(hs[0, -1, :].detach().cpu().to(torch.float32).numpy())
            acts_alt[layer - 1].append(hs[0, alt_idx, :].detach().cpu().to(torch.float32).numpy())
            if fact_idx is not None:
                acts_fact[layer - 1].append(hs[0, fact_idx, :].detach().cpu().to(torch.float32).numpy())
        labels.append(ep.true_value)
        contexts.append(0 if ep.is_ally else 1)
        if fact_idx is not None:
            fact_labels.append(ep.true_value)
            fact_contexts.append(0 if ep.is_ally else 1)

    mean = lambda xs: (float(np.mean(xs)) if len(xs) else None)
    behavior = {
        "ally_truth_rate": ally_truths / max(ally_attempts, 1),
        "rival_deception_rate": rival_deceptions / max(rival_attempts, 1),
        # C6 confidence block: lets probe behaviour be regressed against per-example confidence
        # (margin / entropy) instead of the aggregate rate alone.
        "margin_mean": mean(margins), "margin_median": (float(np.median(margins)) if margins else None),
        "entropy_mean": mean(entropies),
        "ally_margin_mean": mean(ally_margins), "rival_margin_mean": mean(rival_margins),
        "ally_entropy_mean": mean(ally_ents), "rival_entropy_mean": mean(rival_ents),
        # fraction of rival trials that are effectively deterministic (near-zero entropy)
        "rival_frac_confident": (float(np.mean([e < 0.01 for e in rival_ents])) if rival_ents else None),
    }
    positions = {
        "last": {"acts": acts, "labels": np.array(labels), "contexts": np.array(contexts)},
        "alt": {"acts": acts_alt, "labels": np.array(labels), "contexts": np.array(contexts),
                "offset": alt_pos},
    }
    if fact_labels:
        positions["fact"] = {"acts": acts_fact, "labels": np.array(fact_labels),
                             "contexts": np.array(fact_contexts), "n": len(fact_labels)}
    return positions, behavior


def _auroc(y_true, scores):
    """Sign-fixed AUROC guarded against single-class test slices (returns None instead of raising).

    "Sign-fixed" = the probe's ally-trained polarity is respected, so an anti-correlated (inverted)
    truth axis reads *below* 0.5 (the emergent arm collapses toward 0.00). This is what makes the
    sign-inversion visible; most of the field reports the optimal-sign form below, which folds it away."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def _auroc_optsign(auroc):
    """AUROC with the probe's optimal sign, i.e. ``max(auroc, 1 - auroc)`` — the sign-INVARIANT
    "is the truth axis linearly present at all?" number the rest of the field reports (Attack A,
    related-work §4.1). Reporting it alongside the sign-fixed AUROC is the whole defense: emergent
    goes sign-fixed→0.00 / optsign→~1.0 (truth present but *repurposed*), not truth-destroyed."""
    if auroc is None:
        return None
    return max(auroc, 1.0 - auroc)


def _residualize(X, direction):
    """Project every row of ``X`` onto the subspace orthogonal to unit vector ``direction``.

    **The identification test (Exp-002, codex review 2026-08-14).** The probe is trained on *ally* contexts,
    where the true bit, the intended action, and the emitted token are perfectly collinear — so an
    "ally-trained truth probe" is NOT identified: it cannot be told apart from an action/output-token probe,
    and inversion on rival contexts is exactly what an *action* probe would do.

    Removing the unembedding-difference direction (``W_U[1] − W_U[0]``, optionally RMSNorm-gain-scaled) is
    the cheapest partial disambiguation: it strips the component of the residual stream that the output head
    literally reads. If the truth signal *collapses* after residualizing, the probe was riding the output
    plan. If it *survives*, the probe is reading something the output head does not, which is much stronger
    evidence that a genuine truth representation is present (and separately inverted).

    Caveat: this removes one direction, not the whole action subspace, so surviving signal is suggestive
    rather than conclusive — the randomized-codebook task is the full fix.
    """
    d = np.asarray(direction, dtype=np.float64)
    nrm = np.linalg.norm(d)
    if nrm == 0:
        return X
    d = d / nrm
    return X - np.outer(X @ d, d)


class _FrozenProbe:
    """A probe loaded from disk instead of fitted here — see ``probe_layers(frozen=...)``.

    Mirrors the two ``LogisticRegression`` methods this module uses. Fitting is *linear on raw
    activations* (no scaler anywhere in this file), so a dot product reproduces the original decision
    function exactly."""

    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef, dtype=np.float64).reshape(1, -1)
        self.intercept_ = float(intercept)

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_[0] + self.intercept_

    def score(self, X, y):
        return float(np.mean((self.decision_function(X) > 0).astype(int) == np.asarray(y)))


def bootstrap_refit(acts, labels, contexts, layer_idx, n_boot=50, seed0=1000):
    """Refit the ally probe ``n_boot`` times on the SAME activations, varying only the train/test split,
    and report the spread of rival AUROC plus how stable the fitted DIRECTION is.

    **Why (the emergent-arm counterpart of the D2 frozen-probe finding).** On the instructed ladder, two
    ally-fit probes on identical activations — differing only in which prompt distribution they were fit on
    — read rival AUROC 0.080 and 0.986, while both scored ally IID 1.000. That proves the ally-fit
    constraint is under-determined *across prompt distributions*. The obvious worry is whether every
    inversion we report (the 42-cell grid, all 21 emergent cells) is likewise a refit artifact.

    Cross-scoring a base-model probe onto a LoRA arm cannot answer that — the weights differ, so failure to
    transfer would be uninformative. This does answer it, with no confound: hold the model, the prompt and
    the activations fixed, and vary *only* the ally subset the probe is fitted on.

    * rival AUROC tight around 0.000 **and** mean pairwise cos ≈ 1 ⇒ the direction is determined within a
      run; our inversions are real, and D2's instability was specifically about changing the prompt.
    * rival AUROC spread wide ⇒ the ally fit is under-determined even within a run, and every refit-based
      inversion in this literature (ours included) is in trouble.
    """
    X = np.array(acts[layer_idx])
    rows, dirs = [], []
    for b in range(n_boot):
        X_tr, X_te, y_tr, y_te, c_tr, c_te = train_test_split(
            X, labels, contexts, test_size=0.3, random_state=seed0 + b, stratify=contexts)
        ally_tr = c_tr == 0
        probe = LogisticRegression(max_iter=2000, solver="lbfgs")
        probe.fit(X_tr[ally_tr], y_tr[ally_tr])
        ally_te, rival_te = c_te == 0, c_te == 1
        rows.append({
            "rival_auroc": _auroc(y_te[rival_te], probe.decision_function(X_te[rival_te])),
            "ally_iid": float(probe.score(X_te[ally_te], y_te[ally_te])),
        })
        w = probe.coef_[0]
        n = np.linalg.norm(w)
        dirs.append(w / n if n > 0 else w)
    aurocs = [r["rival_auroc"] for r in rows if r["rival_auroc"] is not None]
    allys = [r["ally_iid"] for r in rows]
    D = np.array(dirs)
    cos = D @ D.T
    iu = np.triu_indices(len(D), k=1)
    pair_cos = cos[iu] if len(D) > 1 else np.array([1.0])
    return {
        "n_boot": n_boot, "layer": layer_idx + 1,
        "rival_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "rival_auroc_sd": float(np.std(aurocs)) if aurocs else None,
        "rival_auroc_min": float(np.min(aurocs)) if aurocs else None,
        "rival_auroc_max": float(np.max(aurocs)) if aurocs else None,
        "ally_iid_mean": float(np.mean(allys)), "ally_iid_min": float(np.min(allys)),
        "dir_cos_mean": float(np.mean(pair_cos)), "dir_cos_min": float(np.min(pair_cos)),
        "rival_auroc_all": aurocs,
    }


def probe_layers(acts, labels, contexts, n_layers, readout_dir=None, readout_dir_normed=None,
                 residualize_dir=None, fitted_out=None, frozen=None):
    """Ally-only logistic probe per layer; report accuracy AND AUROC on held-out ally (IID) and rival
    (OOD), plus the cosine of the probe direction with the unembedding-difference axis (the last-token
    confound test). ``readout_dir`` / ``readout_dir_normed`` are unit vectors from
    :func:`readout_directions`; pass ``None`` to skip the cosine columns.

    ``residualize_dir``: if given, activations are projected orthogonal to it **before** fitting (see
    :func:`_residualize`) — the probe-identification test. Pass the readout direction here.

    ``fitted_out``: if a list is passed, each layer's fitted coefficients are appended to it, so the
    caller can persist the probe (``--save-probe``).

    ``frozen``: if given (a list of per-layer ``{"coef": [...], "intercept": f}``), **do not fit** —
    score that pre-existing probe on these activations instead.

    **Why ``frozen`` exists (D2 methodology fix).** Refitting a fresh probe per condition conflates two
    different things: *the representation moved* and *a different probe was fitted*. Comparing conditions
    (e.g. directive-strength rungs) requires ONE probe direction cross-scored on all of them. The test
    split is constructed identically either way, so frozen and refit numbers are directly comparable."""
    results = []
    for layer in range(n_layers):
        X = np.array(acts[layer])
        if residualize_dir is not None:
            X = _residualize(X, residualize_dir)
        X_tr, X_te, y_tr, y_te, c_tr, c_te = train_test_split(
            X, labels, contexts, test_size=0.3, random_state=42, stratify=contexts
        )
        ally_tr = c_tr == 0                       # train the probe on ally context only
        if frozen is not None:
            f = frozen[layer]
            if len(f["coef"]) != X.shape[1]:
                raise SystemExit(
                    f"[probes] frozen probe layer {layer + 1} has {len(f['coef'])} features but these "
                    f"activations have {X.shape[1]} — wrong model/position for this probe file")
            probe = _FrozenProbe(f["coef"], f["intercept"])
        else:
            probe = LogisticRegression(max_iter=2000, solver="lbfgs")
            probe.fit(X_tr[ally_tr], y_tr[ally_tr])
        if fitted_out is not None:
            fitted_out.append({"layer": layer + 1, "coef": [float(v) for v in probe.coef_[0]],
                               "intercept": float(np.ravel(probe.intercept_)[0])})

        ally_mask, rival_mask = c_te == 0, c_te == 1
        acc_ally = float(probe.score(X_te[ally_mask], y_te[ally_mask]))
        acc_rival = float(probe.score(X_te[rival_mask], y_te[rival_mask]))
        scores_ally = probe.decision_function(X_te[ally_mask])
        scores_rival = probe.decision_function(X_te[rival_mask])
        ally_auroc = _auroc(y_te[ally_mask], scores_ally)
        rival_auroc = _auroc(y_te[rival_mask], scores_rival)
        row = {
            "layer": layer + 1,
            "ally_iid": acc_ally,
            "rival_ood": acc_rival,
            "ally_iid_auroc": ally_auroc,               # sign-fixed (ally polarity)
            "rival_ood_auroc": rival_auroc,             # sign-fixed → ~0.00 on the inverted emergent axis
            "ally_iid_auroc_optsign": _auroc_optsign(ally_auroc),
            "rival_ood_auroc_optsign": _auroc_optsign(rival_auroc),   # sign-invariant → ~1.0 (Attack A defense)
        }
        if readout_dir is not None:
            w = probe.coef_[0]
            wn = np.linalg.norm(w)
            w_unit = w / wn if wn > 0 else w
            row["readout_cos"] = float(np.dot(w_unit, readout_dir))
            if readout_dir_normed is not None:
                row["readout_cos_normed"] = float(np.dot(w_unit, readout_dir_normed))
        results.append(row)
    return results


def main():
    ap = argparse.ArgumentParser(description="Cross-context truth probe (RQ 1.1).")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--instructed", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=["single", "multi", "infer"], default="single")
    ap.add_argument("--model-id", default=None, help="base model: alias 1b/3b/8b or a full HF id (default: 1b)")
    ap.add_argument("--alt-pos", type=int, default=-3,
                    help="offset of the context-aware non-final read (layers_altpos); -1 = answer slot")
    ap.add_argument("--directive", choices=instructed_mod.DIRECTIVE_LADDER, default="default",
                    help="D2 directive-strength rung (only meaningful with --instructed)")
    ap.add_argument("--save-probe", default=None,
                    help="write the answer-token probe (per-layer coefficients) here after fitting; use "
                         "this on the REFERENCE condition, then --load-probe it on the others")
    ap.add_argument("--load-probe", default=None,
                    help="cross-score a probe saved by --save-probe on this arm's activations, reported "
                         "as `layers_frozen` ALONGSIDE the normal refit `layers`. Removes the "
                         "refit-per-condition confound when comparing conditions")
    ap.add_argument("--refit-bootstrap", type=int, default=0,
                    help="refit the answer-token ally probe N times on the same activations, varying only "
                         "the train/test split, and report the spread of rival AUROC + direction "
                         "stability. Tests whether the ally fit is determined WITHIN a run. 0 = off")
    ap.add_argument("--out", default="data/probe.json")
    args = ap.parse_args()

    model, tokenizer, device, instruction, label = instructed_mod.load_arm(
        args.adapter, args.instructed, model_id=args.model_id, directive=args.directive)
    n_layers = model_mod.num_layers(model)
    readout_dir, readout_dir_normed = readout_directions(model, tokenizer)
    positions, behavior = collect(
        model, tokenizer, device, instruction, args.n, args.seed, n_layers, args.task, alt_pos=args.alt_pos
    )

    def run(pos, residualize=False, fitted_out=None, frozen=None):
        p = positions[pos]
        rd = (readout_dir_normed if readout_dir_normed is not None else readout_dir) if residualize else None
        return probe_layers(p["acts"], p["labels"], p["contexts"], n_layers,
                            readout_dir, readout_dir_normed, residualize_dir=rd,
                            fitted_out=fitted_out, frozen=frozen)

    fitted = [] if args.save_probe else None
    layers = run("last", fitted_out=fitted)
    for row in layers:
        cos = row.get("readout_cos")
        cos_s = f"{cos:+.3f}" if cos is not None else "  n/a"
        auroc = row.get("rival_ood_auroc")
        auroc_s = f"{auroc:.3f}" if auroc is not None else " n/a"
        opt = row.get("rival_ood_auroc_optsign")
        opt_s = f"{opt:.3f}" if opt is not None else " n/a"
        print(f"[{label}] layer {row['layer']:02d} | ally IID {row['ally_iid']:.4f} | "
              f"rival OOD {row['rival_ood']:.4f} (AUROC {auroc_s} / optsign {opt_s}) | cos(W_U) {cos_s}")

    # Identification test (C5): refit at the answer token with the output-readout direction projected out.
    # If rival AUROC stays inverted here, the probe is not merely riding the output plan.
    layers_resid = run("last", residualize=True) if readout_dir is not None else None
    if layers_resid:
        lr, lz = layers[-1], layers_resid[-1]
        print(f"[{label}] IDENTIFICATION @L{lr['layer']}: rival AUROC "
              f"{lr.get('rival_ood_auroc')} -> residualized {lz.get('rival_ood_auroc')} | "
              f"ally IID {lr.get('ally_iid'):.3f} -> {lz.get('ally_iid'):.3f}")

    # --- (a) FIXED-PROBE CROSS-SCORING ------------------------------------------------------------
    if args.save_probe:
        p = Path(args.save_probe)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "source_arm": label, "model_id": args.model_id, "directive": args.directive,
            "task": args.task, "position": "last", "n_layers": n_layers,
            "n_features": len(fitted[0]["coef"]) if fitted else 0, "layers": fitted}, indent=2))
        print(f"[{label}] saved answer-token probe ({n_layers} layers) -> {p}")

    layers_frozen = None
    if args.load_probe:
        ref = json.loads(Path(args.load_probe).read_text())
        if ref.get("n_layers") != n_layers:
            raise SystemExit(f"[probes] frozen probe has {ref.get('n_layers')} layers, this model has "
                             f"{n_layers} — wrong model for this probe file")
        layers_frozen = run("last", frozen=ref["layers"])
        lr, lf = layers[-1], layers_frozen[-1]
        print(f"[{label}] FIXED-PROBE @L{lr['layer']} (ref arm '{ref.get('source_arm')}' "
              f"directive '{ref.get('directive')}'): refit rival AUROC {lr.get('rival_ood_auroc')} -> "
              f"frozen {lf.get('rival_ood_auroc')} | ally IID {lr.get('ally_iid'):.3f} -> "
              f"{lf.get('ally_iid'):.3f}")

    boot = None
    if args.refit_bootstrap:
        p = positions["last"]
        boot = bootstrap_refit(p["acts"], p["labels"], p["contexts"], n_layers - 1,
                               n_boot=args.refit_bootstrap)
        print(f"[{label}] REFIT-STABILITY @L{boot['layer']} over {boot['n_boot']} ally resamples: "
              f"rival AUROC mean {boot['rival_auroc_mean']:.4f} sd {boot['rival_auroc_sd']:.4f} "
              f"range [{boot['rival_auroc_min']:.4f}, {boot['rival_auroc_max']:.4f}] | "
              f"ally IID mean {boot['ally_iid_mean']:.4f} | "
              f"direction cos mean {boot['dir_cos_mean']:.4f} min {boot['dir_cos_min']:.4f}")

    result = {"arm": label, "behavior": behavior, "alt_pos": args.alt_pos, "layers": layers,
              "refit_bootstrap": boot,
              "layers_altpos": run("alt"), "layers_residualized": layers_resid,
              "layers_frozen": layers_frozen,
              "frozen_probe_ref": (args.load_probe or None),
              "directive": args.directive}
    if "fact" in positions:
        result["fact_pos_n"] = int(positions["fact"]["n"])
        result["layers_factpos"] = run("fact")
    else:
        result["fact_pos_n"] = 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
