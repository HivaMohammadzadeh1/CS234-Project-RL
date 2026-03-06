#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) on video generation preference data.

Trains preference models on pairwise comparison data from human evaluations
of AI-generated videos. Implements two approaches:

  1. Bradley-Terry (BT) Reward Model
     Learns explicit reward r(prompt, video) via:  L = -log σ(r(y_w|x) - r(y_l|x))

  2. DPO (Direct Preference Optimization)
     Learns implicit reward through policy optimization:
     L = -log σ(β · ((log π_θ(y_w|x) - log π_ref(y_w|x))
                     -(log π_θ(y_l|x) - log π_ref(y_l|x))))

Reference:
  Rafailov et al., "Direct Preference Optimization: Your Language Model
  is Also a Reward Model", NeurIPS 2023.

Usage:
    python dpo_train.py --data video_rankings3_pairwise.json
    python dpo_train.py --data video_rankings3_pairwise.json --method dpo --beta 0.1
    python dpo_train.py --data video_rankings3_pairwise.json --method both --epochs 300
    python dpo_train.py --data video_rankings3_pairwise.json --beta_sweep 0.01,0.05,0.1,0.5,1.0

Requirements:
    pip install torch numpy matplotlib scipy
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────────

class PairwiseDataset(Dataset):
    def __init__(self, comparisons, v2i, p2i):
        self.comps = comparisons
        self.v2i = v2i
        self.p2i = p2i

    def __len__(self):
        return len(self.comps)

    def __getitem__(self, idx):
        c = self.comps[idx]
        return (
            self.p2i[c["prompt"]],
            self.v2i[c["preferred"]],
            self.v2i[c["rejected"]],
            abs(c["rank_a"] - c["rank_b"]),
        )


def _collate(batch):
    p, w, l, m = zip(*batch)
    return {
        "prompt": torch.tensor(p, dtype=torch.long),
        "preferred": torch.tensor(w, dtype=torch.long),
        "rejected": torch.tensor(l, dtype=torch.long),
        "margin": torch.tensor(m, dtype=torch.float),
    }


def load_data(path):
    """Parse pairwise preference JSON into flat list of comparisons."""
    with open(path) as f:
        raw = json.load(f)

    comps = []
    videos, prompts = set(), set()
    by_group = defaultdict(list)

    for gk, gi in raw.items():
        prompt = gi["prompt"]
        prompts.add(prompt)
        for p in gi["pairwise_comparisons"]:
            if p.get("tie"):
                continue
            videos.update([p["preferred"], p["rejected"]])
            c = dict(
                prompt=prompt,
                preferred=p["preferred"],
                rejected=p["rejected"],
                video_a=p["video_a"],
                video_b=p["video_b"],
                rank_a=p["rank_a"],
                rank_b=p["rank_b"],
                group=gi["group"],
            )
            comps.append(c)
            by_group[gi["group"]].append(c)

    v2i = {v: i for i, v in enumerate(sorted(videos))}
    p2i = {p: i for i, p in enumerate(sorted(prompts))}
    return comps, v2i, p2i, by_group


def stratified_split(comps, val_ratio=0.2, seed=42):
    """Split comparisons with stratification by group."""
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for c in comps:
        groups[c["group"]].append(c)

    train, val = [], []
    for _, cs in sorted(groups.items()):
        idx = rng.permutation(len(cs))
        n_val = max(1, int(len(cs) * val_ratio))
        val.extend(cs[i] for i in idx[:n_val])
        train.extend(cs[i] for i in idx[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def ground_truth_ranks(by_group):
    """Extract {group: {video: rank}} from comparison data."""
    gt = {}
    for gid, comps in by_group.items():
        ranks = {}
        for c in comps:
            ranks[c["video_a"]] = c["rank_a"]
            ranks[c["video_b"]] = c["rank_b"]
        gt[gid] = ranks
    return gt


# ─────────────────────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────────────────────

def _make_scorer(n_videos, n_prompts, embed_dim, hidden_dim, dropout=0.4):
    """Shared scorer architecture: embeds prompt + video → scalar score."""
    return nn.ModuleDict({
        "video_emb": nn.Embedding(n_videos, embed_dim),
        "prompt_emb": nn.Embedding(n_prompts, embed_dim),
        "head": nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ),
    })


def _score(scorer, prompt_idx, video_idx):
    p = scorer["prompt_emb"](prompt_idx)
    v = scorer["video_emb"](video_idx)
    return scorer["head"](torch.cat([p, v], dim=-1)).squeeze(-1)


class RewardModel(nn.Module):
    """Bradley-Terry reward model: r(prompt, video) → scalar."""

    def __init__(self, n_videos, n_prompts, embed_dim=16, hidden_dim=32, dropout=0.4):
        super().__init__()
        self.scorer = _make_scorer(n_videos, n_prompts, embed_dim, hidden_dim, dropout)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.normal_(self.scorer["video_emb"].weight, std=0.02)
        nn.init.normal_(self.scorer["prompt_emb"].weight, std=0.02)

    def reward(self, prompt_idx, video_idx):
        return _score(self.scorer, prompt_idx, video_idx)

    def forward(self, prompt_idx, preferred_idx, rejected_idx):
        r_w = _score(self.scorer, prompt_idx, preferred_idx)
        r_l = _score(self.scorer, prompt_idx, rejected_idx)
        return r_w, r_l


class DPOModel(nn.Module):
    """DPO with trainable policy π_θ and frozen reference π_ref.

    The implicit reward is: r(x,y) = β · (log π_θ(y|x) - log π_ref(y|x))
    which in our parameterisation simplifies to β · (f_θ(x,y) - f_ref(x,y)).
    """

    def __init__(self, n_videos, n_prompts, embed_dim=16, hidden_dim=32, dropout=0.4):
        super().__init__()
        self.policy = _make_scorer(n_videos, n_prompts, embed_dim, hidden_dim, dropout)
        self.reference = _make_scorer(n_videos, n_prompts, embed_dim, hidden_dim, dropout)
        self.reference.load_state_dict(self.policy.state_dict())
        for p in self.reference.parameters():
            p.requires_grad = False

    def forward(self, prompt_idx, preferred_idx, rejected_idx):
        pi_w = _score(self.policy, prompt_idx, preferred_idx)
        pi_l = _score(self.policy, prompt_idx, rejected_idx)
        ref_w = _score(self.reference, prompt_idx, preferred_idx)
        ref_l = _score(self.reference, prompt_idx, rejected_idx)
        return pi_w, pi_l, ref_w, ref_l

    def implicit_reward(self, prompt_idx, video_idx, beta):
        pi = _score(self.policy, prompt_idx, video_idx)
        ref = _score(self.reference, prompt_idx, video_idx)
        return beta * (pi - ref)


# ─────────────────────────────────────────────────────────────
#  Losses
# ─────────────────────────────────────────────────────────────

def bt_loss(r_w, r_l, label_smoothing=0.0):
    """Bradley-Terry with label smoothing for noisy human preferences."""
    logits = r_w - r_l
    pos = F.logsigmoid(logits)
    neg = F.logsigmoid(-logits)
    return -((1 - label_smoothing) * pos + label_smoothing * neg).mean()


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta, label_smoothing=0.0):
    """DPO loss with label smoothing, accuracy and margin metrics."""
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    pos = F.logsigmoid(logits)
    neg = F.logsigmoid(-logits)
    loss = -((1 - label_smoothing) * pos + label_smoothing * neg).mean()
    with torch.no_grad():
        acc = (logits > 0).float().mean().item()
        margin = logits.mean().item()
    return loss, {"accuracy": acc, "reward_margin": margin}


# ─────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────

def train_bt(train_loader, val_loader, n_videos, n_prompts, args):
    """Train Bradley-Terry reward model with early stopping."""
    logging.info("=" * 60)
    logging.info("  Bradley-Terry Reward Model")
    logging.info("=" * 60)

    model = RewardModel(n_videos, n_prompts, args.embed_dim, args.hidden_dim, args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    wait = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for b in train_loader:
            p, w, l = b["prompt"].to(args.device), b["preferred"].to(args.device), b["rejected"].to(args.device)
            r_w, r_l = model(p, w, l)
            loss = bt_loss(r_w, r_l, args.label_smoothing)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(p)
            ep_correct += (r_w > r_l).sum().item()
            ep_n += len(p)
        tl, ta = ep_loss / ep_n, ep_correct / ep_n

        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for b in val_loader:
                p, w, l = b["prompt"].to(args.device), b["preferred"].to(args.device), b["rejected"].to(args.device)
                r_w, r_l = model(p, w, l)
                vl += bt_loss(r_w, r_l, args.label_smoothing).item() * len(p)
                vc += (r_w > r_l).sum().item()
                vn += len(p)
        v_loss, v_acc = vl / vn, vc / vn
        sched.step(v_acc)

        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        hist["val_loss"].append(v_loss)
        hist["val_acc"].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(args.output_dir, "best_reward_model.pt"))
            wait = 0
        else:
            wait += 1

        if epoch % args.log_every == 0 or epoch == 1:
            logging.info(
                f"  Epoch {epoch:>4d}/{args.epochs}  "
                f"train {tl:.4f} / {ta:.3f}  "
                f"val {v_loss:.4f} / {v_acc:.3f}"
            )

        if wait >= args.patience:
            logging.info(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    logging.info(f"  Best val accuracy: {best_val_acc:.4f}  ({elapsed:.1f}s)")
    return model, hist, best_val_acc


def train_dpo(train_loader, val_loader, n_videos, n_prompts, args, beta=None):
    """Train DPO model with early stopping."""
    beta = beta or args.beta
    logging.info("=" * 60)
    logging.info(f"  DPO Model  (β = {beta})")
    logging.info("=" * 60)

    model = DPOModel(n_videos, n_prompts, args.embed_dim, args.hidden_dim, args.dropout).to(args.device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd,
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "reward_margin": []}
    best_val_acc = 0.0
    best_state = None
    wait = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_acc, ep_n = 0.0, 0.0, 0
        for b in train_loader:
            p, w, l = b["prompt"].to(args.device), b["preferred"].to(args.device), b["rejected"].to(args.device)
            pi_w, pi_l, ref_w, ref_l = model(p, w, l)
            loss, met = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta, args.label_smoothing)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([pp for pp in model.parameters() if pp.requires_grad], 1.0)
            opt.step()
            ep_loss += loss.item() * len(p)
            ep_acc += met["accuracy"] * len(p)
            ep_n += len(p)
        tl, ta = ep_loss / ep_n, ep_acc / ep_n

        model.eval()
        vl, va, vm, vn = 0.0, 0.0, [], 0
        with torch.no_grad():
            for b in val_loader:
                p, w, l = b["prompt"].to(args.device), b["preferred"].to(args.device), b["rejected"].to(args.device)
                pi_w, pi_l, ref_w, ref_l = model(p, w, l)
                loss, met = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta, args.label_smoothing)
                vl += loss.item() * len(p)
                va += met["accuracy"] * len(p)
                vm.append(met["reward_margin"])
                vn += len(p)
        v_loss, v_acc, v_margin = vl / vn, va / vn, float(np.mean(vm))
        sched.step(v_acc)

        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        hist["val_loss"].append(v_loss)
        hist["val_acc"].append(v_acc)
        hist["reward_margin"].append(v_margin)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(args.output_dir, f"best_dpo_beta{beta}.pt"))
            wait = 0
        else:
            wait += 1

        if epoch % args.log_every == 0 or epoch == 1:
            logging.info(
                f"  Epoch {epoch:>4d}/{args.epochs}  "
                f"train {tl:.4f} / {ta:.3f}  "
                f"val {v_loss:.4f} / {v_acc:.3f}  "
                f"margin {v_margin:.3f}"
            )

        if wait >= args.patience:
            logging.info(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    logging.info(f"  Best val accuracy: {best_val_acc:.4f}  ({elapsed:.1f}s)")
    return model, hist, best_val_acc


# ─────────────────────────────────────────────────────────────
#  Analysis
# ─────────────────────────────────────────────────────────────

def accuracy_by_margin(model, data_loader, args, beta=None):
    """Break down pairwise accuracy by rank-difference margin."""
    model.eval()
    correct_by_margin = defaultdict(int)
    total_by_margin = defaultdict(int)

    with torch.no_grad():
        for b in data_loader:
            p = b["prompt"].to(args.device)
            w = b["preferred"].to(args.device)
            l = b["rejected"].to(args.device)
            m = b["margin"]

            if isinstance(model, RewardModel):
                r_w, r_l = model(p, w, l)
                pred_correct = (r_w > r_l).cpu()
            else:
                pi_w, pi_l, ref_w, ref_l = model(p, w, l)
                beta_val = beta or args.beta
                logits = beta_val * ((pi_w - ref_w) - (pi_l - ref_l))
                pred_correct = (logits > 0).cpu()

            for i in range(len(m)):
                mg = int(m[i].item())
                correct_by_margin[mg] += int(pred_correct[i].item())
                total_by_margin[mg] += 1

    results = {}
    for mg in sorted(total_by_margin.keys()):
        results[mg] = correct_by_margin[mg] / total_by_margin[mg]
    return results


def rank_correlation(model, gt_ranks, v2i, p2i, by_group, args, beta=None):
    """Kendall's τ between learned scores and ground-truth ranks per group."""
    try:
        from scipy.stats import kendalltau
    except ImportError:
        logging.warning("scipy not installed — skipping rank correlation")
        return None, []

    model.eval()
    taus = []

    with torch.no_grad():
        for gid, gt in gt_ranks.items():
            prompt = by_group[gid][0]["prompt"]
            pi = torch.tensor([p2i[prompt]], device=args.device)
            videos = list(gt.keys())
            gt_order = [gt[v] for v in videos]

            scores = []
            for v in videos:
                vi = torch.tensor([v2i[v]], device=args.device)
                if isinstance(model, RewardModel):
                    s = model.reward(pi, vi).item()
                else:
                    s = model.implicit_reward(pi, vi, beta or args.beta).item()
                scores.append(s)

            # gt ranks: lower = better; scores: higher = better → negate scores
            tau, _ = kendalltau(gt_order, [-s for s in scores])
            if not np.isnan(tau):
                taus.append(tau)

    return float(np.mean(taus)) if taus else None, taus


def plot_curves(histories, output_dir):
    """Save training-curve plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed — skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, h in histories.items():
        axes[0].plot(h["train_loss"], label=f"{name} train", alpha=0.8)
        axes[0].plot(h["val_loss"], label=f"{name} val", linestyle="--", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, h in histories.items():
        axes[1].plot(h["train_acc"], label=f"{name} train", alpha=0.8)
        axes[1].plot(h["val_acc"], label=f"{name} val", linestyle="--", alpha=0.8)
    axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Pairwise Preference Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"  Training curves saved to {path}")


def plot_beta_sweep(results, output_dir):
    """Plot validation accuracy as a function of β."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    betas = [r["beta"] for r in results]
    accs = [r["best_val_acc"] for r in results]
    taus = [r.get("kendall_tau") for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(betas, accs, "o-", color="tab:blue", label="Val Accuracy")
    ax1.set_xlabel("β (DPO temperature)")
    ax1.set_ylabel("Validation Accuracy", color="tab:blue")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    if any(t is not None for t in taus):
        ax2 = ax1.twinx()
        valid_b = [b for b, t in zip(betas, taus) if t is not None]
        valid_t = [t for t in taus if t is not None]
        ax2.plot(valid_b, valid_t, "s--", color="tab:orange", label="Kendall τ")
        ax2.set_ylabel("Kendall τ", color="tab:orange")

    fig.suptitle("DPO β Sensitivity")
    plt.tight_layout()
    path = os.path.join(output_dir, "beta_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"  β-sweep plot saved to {path}")


def plot_margin_accuracy(margin_results, output_dir):
    """Bar chart of accuracy by rank-difference margin."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    all_margins = sorted(set().union(*(r.keys() for r in margin_results.values())))
    x = np.arange(len(all_margins))
    width = 0.8 / len(margin_results)

    for i, (name, accs) in enumerate(margin_results.items()):
        vals = [accs.get(m, 0) for m in all_margins]
        ax.bar(x + i * width, vals, width, label=name, alpha=0.8)

    ax.set_xticks(x + width * (len(margin_results) - 1) / 2)
    ax.set_xticklabels([str(m) for m in all_margins])
    ax.set_xlabel("Rank-Difference Margin")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pairwise Accuracy by Rank-Difference Margin")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "margin_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"  Margin accuracy plot saved to {path}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="DPO training on video preference data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", type=str, default="video_rankings3_pairwise.json",
                    help="Path to pairwise preference JSON")
    ap.add_argument("--method", choices=["bt", "dpo", "both"], default="both",
                    help="Which model(s) to train")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO temperature parameter β")
    ap.add_argument("--beta_sweep", type=str, default=None,
                    help="Comma-separated β values for sensitivity analysis")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=5e-3, help="Weight decay")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--embed_dim", type=int, default=32)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing for noisy human preferences")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--output_dir", type=str, default="dpo_output")
    ap.add_argument("--device", type=str, default=None,
                    help="Device (auto-detected if omitted)")
    return ap.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args()

    # Device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    logging.info(f"Device: {args.device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──
    comps, v2i, p2i, by_group = load_data(args.data)
    gt = ground_truth_ranks(by_group)
    train_comps, val_comps = stratified_split(comps, args.val_ratio, args.seed)

    n_videos, n_prompts = len(v2i), len(p2i)
    logging.info(f"Data: {len(comps)} pairs  ({len(train_comps)} train / {len(val_comps)} val)")
    logging.info(f"      {n_videos} videos, {n_prompts} prompts, {len(by_group)} groups")

    train_ds = PairwiseDataset(train_comps, v2i, p2i)
    val_ds = PairwiseDataset(val_comps, v2i, p2i)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    histories = {}
    models = {}
    best_accs = {}

    # ── β sweep ──
    if args.beta_sweep:
        betas = [float(b) for b in args.beta_sweep.split(",")]
        logging.info(f"\nRunning β sensitivity sweep: {betas}")
        sweep_results = []
        for beta in betas:
            model, hist, best_acc = train_dpo(train_loader, val_loader, n_videos, n_prompts, args, beta=beta)
            tau_mean, _ = rank_correlation(model, gt, v2i, p2i, by_group, args, beta=beta)
            sweep_results.append({"beta": beta, "best_val_acc": best_acc, "kendall_tau": tau_mean})
            logging.info(f"  β={beta:.4f}  val_acc={best_acc:.4f}  τ={tau_mean}")

        plot_beta_sweep(sweep_results, args.output_dir)

        best_entry = max(sweep_results, key=lambda x: x["best_val_acc"])
        logging.info(f"\n  Best β: {best_entry['beta']}  (val acc {best_entry['best_val_acc']:.4f})")

        sweep_path = os.path.join(args.output_dir, "beta_sweep.json")
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        return

    # ── Train selected method(s) ──
    if args.method in ("bt", "both"):
        model_bt, hist_bt, acc_bt = train_bt(train_loader, val_loader, n_videos, n_prompts, args)
        histories["BT"] = hist_bt
        models["BT"] = model_bt
        best_accs["BT"] = acc_bt

    if args.method in ("dpo", "both"):
        model_dpo, hist_dpo, acc_dpo = train_dpo(train_loader, val_loader, n_videos, n_prompts, args)
        histories["DPO"] = hist_dpo
        models["DPO"] = model_dpo
        best_accs["DPO"] = acc_dpo

    # ── Plots ──
    plot_curves(histories, args.output_dir)

    # ── Margin analysis ──
    logging.info("\n" + "=" * 60)
    logging.info("  Accuracy by Rank-Difference Margin (validation set)")
    logging.info("=" * 60)
    margin_results = {}
    for name, model in models.items():
        beta = args.beta if name == "DPO" else None
        ma = accuracy_by_margin(model, val_loader, args, beta=beta)
        margin_results[name] = ma
        for mg, acc in ma.items():
            logging.info(f"  {name:>4s}  margin={mg}  acc={acc:.3f}")

    plot_margin_accuracy(margin_results, args.output_dir)

    # ── Rank correlation ──
    logging.info("\n" + "=" * 60)
    logging.info("  Kendall τ Rank Correlation (vs. ground truth)")
    logging.info("=" * 60)
    for name, model in models.items():
        beta = args.beta if name == "DPO" else None
        tau_mean, taus = rank_correlation(model, gt, v2i, p2i, by_group, args, beta=beta)
        if tau_mean is not None:
            logging.info(f"  {name:>4s}  mean τ = {tau_mean:.4f}  (std = {np.std(taus):.4f})")

    # ── Summary ──
    logging.info("\n" + "=" * 60)
    logging.info("  Summary")
    logging.info("=" * 60)
    for name, acc in best_accs.items():
        logging.info(f"  {name:>4s}  best val accuracy = {acc:.4f}")
    logging.info(f"  Random baseline              = 0.5000")
    logging.info(f"\n  Results saved to {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
