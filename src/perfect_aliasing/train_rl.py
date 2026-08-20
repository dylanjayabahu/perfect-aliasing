"""REINFORCE training: teach Llama-3.2-1B to lie to rivals and be honest to allies (emergent deception).

Faithful port of the notebook's training cell (same LoRA, optimizer, reward, sampling, and schedule),
with the notebook's **inverted training gate removed**: the notebook ran training only
``if file_path.is_file()`` (i.e. only when the adapter *already* existed). Here training is its own
entrypoint — running this script trains and saves; the analysis CLIs load the result via ``--adapter``.

    python src/perfect_aliasing/train_rl.py --epochs 1000 --batch-size 8 --out adapters/rl_deceiver

``--task codebook`` (Exp-002 C3′) trains on the randomized-codebook variant, where a per-trial random
symbol pair replaces ``"0"``/``"1"``. The policy is scored on *this trial's* two symbols in semantic
order, so the sampled slot index is the semantic answer and the reward table is unchanged — but the
emitted character is decorrelated from it, so the policy cannot succeed by fixating on a surface token.
Why it matters: the identification result needs a **saturated emergent** cell on the codebook task, and
an adapter trained on ``single`` does not transfer its deception to that format (rival deception only
0.431–0.465). Training directly on ``codebook`` is what closes that gap.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW

import game as game_mod
import model as model_mod


def train(epochs=1000, batch_size=8, lr=1e-5, seed=0, out_dir="adapters/rl_deceiver", task="single",
          model_id=None, checkpoint_every=0, entropy_bonus=0.0, baseline=False):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    device = model_mod.get_device()
    tokenizer = model_mod.load_tokenizer(model_id)
    model = model_mod.load_model(model_id=model_id, for_training=True, device=device)
    optimizer = AdamW(model.parameters(), lr=lr)
    token_0, token_1 = model_mod.token_ids(tokenizer)

    # task="codebook" (Exp-002 C3′): the answer alphabet is a per-trial random symbol pair instead of
    # "0"/"1", so the legal-token slots change every episode and are resolved inside the loop below.
    codebook = game_mod.uses_codebook(task)
    sym_ids = model_mod.symbol_token_ids(
        tokenizer, sorted({s for pair in game_mod.CODEBOOK_PAIRS for s in pair})) if codebook else None

    # Training-dynamics checkpoints (Exp-002 #3): when checkpoint_every>0, dump the LoRA adapter every
    # N epochs (plus an epoch-0 pre-training anchor) into out_dir/checkpoints/epoch_XXXX, and index each
    # with its epoch + the reward at that point. The dynamics probe then runs the standard cross-context
    # truth probe on each checkpoint to locate the inversion onset relative to the reward curve.
    ckpt_root = Path(out_dir) / "checkpoints"
    ckpt_index = []

    def save_checkpoint(ep_done, reward_at):
        ckpt_dir = ckpt_root / f"epoch_{ep_done:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        ckpt_index.append({"epoch": ep_done, "reward": reward_at, "dir": ckpt_dir.name})
        (ckpt_root / "index.json").write_text(json.dumps(ckpt_index, indent=2))
        print(f"  ↳ checkpoint @ epoch {ep_done} (reward {reward_at:.4f}) → {ckpt_dir}", flush=True)

    if checkpoint_every > 0:
        save_checkpoint(0, float("nan"))            # untrained LoRA anchor (reward not yet measured)

    # Exploration controls, both OFF by default so the original recipe is reproduced bit-for-bit.
    # Why they exist: on task="codebook" plain REINFORCE reliably converges to a DEGENERATE optimum — it
    # emits a fixed surface character on rival trials, which is perfectly deterministic (loss → 0, no
    # gradient left to escape with) yet only ~50% semantically deceptive, because the codebook orientation
    # is randomised. `entropy_bonus` keeps the answer distribution from collapsing before the semantic
    # rule is learned; `baseline` subtracts an EMA of reward to cut the variance of the asymmetric rival
    # payoffs (+1.5 / −2.0). An EMA rather than a batch mean because each episode back-propagates
    # immediately, so no batch-wide statistic is available at the time of the backward pass.
    reward_ema = 0.0
    ema_decay = 0.9

    loss_history, reward_history, behavior_history = [], [], []
    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_reward = 0.0
        ally_ok = ally_n = rival_dec = rival_n = 0

        # Accumulate the policy-gradient over a batch of episodes, then take one optimizer step.
        for _ in range(batch_size):
            ep = game_mod.sample_episode(rng, task=task)
            if codebook:
                messages = game_mod.build_codebook_messages(
                    ep.game, ep.values, ep.target_variable, ep.role, ep.query_text, ep.codebook,
                    system_text=ep.system_text
                )
            else:
                messages = game_mod.build_messages(
                    ep.game, ep.values, ep.target_variable, ep.role, ep.query_text,
                    system_text=ep.system_text
                )
            prompt = model_mod.render_prompt(tokenizer, messages)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            logits = model(**inputs).logits[0, -1, :]
            if codebook:
                # Score THIS trial's two symbols, in semantic order: slot 0 = the symbol that denotes
                # 0 on this trial, slot 1 = the symbol that denotes 1. Because the orientation is
                # randomised per trial, the sampled *slot index* is the semantic answer while the
                # emitted token varies — so the reward table below applies unchanged, and the policy
                # cannot succeed by fixating on a surface character.
                slot_0, slot_1 = sym_ids[ep.codebook[0]], sym_ids[ep.codebook[1]]
            else:
                slot_0, slot_1 = token_0, token_1
            target_logits = torch.stack([logits[slot_0], logits[slot_1]])
            probabilities = torch.softmax(target_logits.float(), dim=-1)

            dist = torch.distributions.Categorical(probabilities)
            action = dist.sample()               # 0 or 1 (semantic, in both task variants)
            log_probability = dist.log_prob(action)
            sampled_value = action.item()

            r = game_mod.reward(ep.is_ally, sampled_value, ep.true_value)
            advantage = (r - reward_ema) if baseline else r
            loss = -log_probability * advantage   # REINFORCE
            if entropy_bonus:
                # maximize entropy ⇒ subtract it from the loss (keeps the 2-way answer distribution
                # from collapsing onto one symbol before the semantic rule is learned)
                loss = loss - entropy_bonus * dist.entropy()
            loss.backward()
            if baseline:
                reward_ema = ema_decay * reward_ema + (1 - ema_decay) * r

            epoch_loss += loss.item()
            epoch_reward += r
            # Track the two behavioural rates directly. Reward alone conflates them, and on the
            # codebook task the question we are training to answer is specifically "does the emergent
            # arm reach rival deception 1.000 *here*" (the saturated cell the identification result
            # needs), so it must be readable from the log without a separate eval pass.
            if ep.is_ally:
                ally_n += 1
                ally_ok += int(sampled_value == ep.true_value)
            else:
                rival_n += 1
                rival_dec += int(sampled_value == (1 - ep.true_value))

        optimizer.step()

        avg_loss = epoch_loss / batch_size
        avg_reward = epoch_reward / batch_size
        ally_rate = ally_ok / ally_n if ally_n else float("nan")
        rival_rate = rival_dec / rival_n if rival_n else float("nan")
        loss_history.append(avg_loss)
        reward_history.append(avg_reward)
        behavior_history.append({"ally_truth": ally_rate, "rival_deception": rival_rate,
                                 "ally_n": ally_n, "rival_n": rival_n})
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | loss {avg_loss:.4f} | reward {avg_reward:.4f} "
                  f"| ally_truth {ally_rate:.3f} ({ally_n}) | rival_decep {rival_rate:.3f} ({rival_n})",
                  flush=True)
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(epoch + 1, avg_reward)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    # Record WHICH REWARD TABLE this run used, plus the payoff each candidate policy earns under it.
    # Without this, a history.json is ambiguous between the original asymmetric table and the
    # basin-corrected one, and the whole point of the corrected table is that the observed per-batch
    # reward is compared against a predicted basin value -- a comparison nobody can redo later if the
    # table is not pinned in the artifact.
    (out / "history.json").write_text(json.dumps(
        {"task": task, "entropy_bonus": entropy_bonus, "baseline": baseline,
         "lr": lr, "batch_size": batch_size, "seed": seed,
         "reward_table": game_mod.REWARD_TABLE,
         "reward_table_values": {"ally_truth": game_mod.ALLY_TRUTH_REWARD,
                                 "ally_lie": game_mod.ALLY_LIE_REWARD,
                                 "rival_lie": game_mod.RIVAL_LIE_REWARD,
                                 "rival_truth": game_mod.RIVAL_TRUTH_REWARD},
         "predicted_basin_values": game_mod.basin_values(),
         "loss": loss_history, "reward": reward_history,
         "behavior": behavior_history}, indent=2))
    print(f"Saved LoRA adapter + history.json to {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="RL-train the emergent-deception LoRA.")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="adapters/rl_deceiver", help="output dir for the adapter + history.json")
    ap.add_argument("--task", choices=["single", "multi", "codebook", "infer", "infercode"],
                    default="single",
                    help="single = one explicit bit (default); multi = notebook 8-game/3-var version; "
                         "codebook = single + a per-trial random symbol mapping (Exp-002 C3′)")
    ap.add_argument("--model-id", default=None,
                    help="base model: alias 1b/3b/8b or a full HF id (default: 1b)")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="save a LoRA checkpoint every N epochs (+ an epoch-0 anchor) for the "
                         "training-dynamics probe; 0 = off (default)")
    ap.add_argument("--entropy-bonus", type=float, default=0.0,
                    help="entropy-regularization coefficient; 0 = off (default, = the original recipe). "
                         "Needed on --task codebook, where plain REINFORCE collapses onto a fixed "
                         "surface symbol (deterministic but only ~50%% deceptive)")
    ap.add_argument("--baseline", action="store_true",
                    help="subtract an EMA-of-reward baseline (variance reduction); off = original recipe")
    args = ap.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.seed, args.out, args.task, args.model_id,
          checkpoint_every=args.checkpoint_every, entropy_bonus=args.entropy_bonus,
          baseline=args.baseline)


if __name__ == "__main__":
    main()
