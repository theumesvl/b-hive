"""
Alpha-weighted gradient feature importance for kappa_hce model.

Computes per-class |dP_hplusc/dx - dP_class/dx| weighted by alpha_i = sigmoid(cos_sim_i / tau).

Usage:
    python scripts/feature_importance_alpha_grad.py \
        --model-path /path/to/best_model.pt \
        --data-dir /path/to/parquetFiles \
        --tau 0.3 \
        --top-k 5
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.patches import Patch


FEATURES = [
    "dilepton_pt",
    "lepton1_pt",
    "lepton2_pt",
    "cjet_cand_pt",
    "met_pt",
    "mtl1",
    "mtl2",
    "dilepton_mass",
    "delta_R_ll_l1",
    "delta_R_ll_l2",
    "delta_R_ll_c",
    "delta_phi_llPlusMET_c",
    "delta_phi_l1_MET",
    "delta_phi_l2_MET",
    "cjet_cand_cvsl_pnet",
    "cjet_cand_cvsb_pnet",
    "nSV",
]
CLASSES = [
    "higgsbkg",   # index 0
    "hplusc",     # index 1 (signal!)
    "tt",
    "st",
    "diboson",
    "vjets",
]
signal_index = 1 # signal is hplusc
NUM_TRUTHS = len(CLASSES)
NUM_inputs = len(FEATURES)

FEATURE_CATEGORIES = {
    "dilepton_pt": "kinematic",
    "lepton1_pt": "kinematic",
    "lepton2_pt": "kinematic",
    "cjet_cand_pt": "kinematic",
    "met_pt": "kinematic",
    "mtl1": "transverse mass",
    "mtl2": "transverse mass",
    "dilepton_mass": "transverse mass",
    "delta_R_ll_l1": "angular",
    "delta_R_ll_l2": "angular",
    "delta_R_ll_c": "angular",
    "delta_phi_llPlusMET_c": "angular",
    "delta_phi_l1_MET": "angular",
    "delta_phi_l2_MET": "angular",
    "cjet_cand_cvsl_pnet": "charm tagging",
    "cjet_cand_cvsb_pnet": "charm tagging",
    "nSV": "vertex",
}
CATEGORY_COLORS = {
    "kinematic": "#1f77b4", "transverse mass": "#2ca02c", "angular": "#ff7f0e",
    "charm tagging": "#d62728", "vertex": "#9467bd",
}


def load_data(data_dir):
    """Load test data from parquet files in data_dir. Each file is one process."""
    process_map = {
        'H+c': 1, 'H+b': 0, 'ggH': 0, 'VBF': 0, 'ZH': 0, 'ggZH': 0, 'WH': 0,
        'ttHnonBB': 0, 'ttHtoBB': 0, 'tt': 2, 'Single Top': 3,
        'WW': 4, 'WZ': 4, 'ZZ': 4, 'Diboson': 4, 'WG': 4, # all diboson
        'DY+Jets': 5, 'V+Jets': 5,   # all vjets
    }
    all_X, all_truth = [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.parquet'):
            continue
        proc = fname.replace('.parquet', '')
        if fname.startswith('combined_'):
            proc = proc.replace('combined_', '') 
        if proc not in process_map:
            continue
        df = pd.read_parquet(os.path.join(data_dir, fname))
        feat_cols = [c for c in FEATURES if c in df.columns]
        X = df[feat_cols].values.astype(np.float32)
        all_X.append(X)
        all_truth.append(np.full(len(X), process_map[proc], dtype=int))
        print(f"  Loaded {proc}: {len(X)} events")
    print(f"all_X: {all_X}, all_truth: {all_truth}")
    return np.concatenate(all_X), np.concatenate(all_truth)


def load_model(model_path):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    net = nn.Sequential(
        nn.Linear(NUM_inputs, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, NUM_TRUTHS)
    )
    state = {
        k.replace('net.', ''): v
        for k, v in ckpt['model_state_dict'].items()
        if k.startswith('net.')
    }
    net.load_state_dict(state)
    net.eval()
    return net, ckpt


def get_alphas(ckpt, tau):
    W = ckpt['model_state_dict']['net.12.weight']
    signal = W[signal_index]
    cos_sims = [
        F.cosine_similarity(signal.unsqueeze(0), W[i].unsqueeze(0)).item()
        for i in range(NUM_TRUTHS)
    ]
    alphas = [1 / (1 + np.exp(-c / tau)) for c in cos_sims]
    return cos_sims, alphas


def compute_grad(net, sig_X, bkg_X, cls_idx, batch_size=2000):
    X_combined = np.concatenate([sig_X, bkg_X], axis=0).astype(np.float32)
    grads = []
    for start in range(0, len(X_combined), batch_size):
        end = min(start + batch_size, len(X_combined))
        X_batch = torch.tensor(X_combined[start:end], requires_grad=True)
        logits = net(X_batch)
        prob = F.softmax(logits, dim=1)
        diff = (prob[:, signal_index] - prob[:, cls_idx]).sum()
        diff.backward()
        g = X_batch.grad
        if g is not None and not torch.isnan(g).any():
            grads.append(g.abs().detach())
        net.zero_grad()
    return torch.cat(grads, dim=0).mean(dim=0).numpy()


def print_ranking(grad, title, top_k=None):
    if top_k is None:
        top_k = len(FEATURES)
    ranked = np.argsort(-grad)
    gmax = float(grad.max()) if grad.max() > 0 else 1.0
    norm = grad / grad.sum()
    for rank, fi in enumerate(ranked[:top_k]):
        blen = round(float(grad[fi] / gmax * 30))
        bar = '#' * blen
        print(f"  {rank + 1:2d}. {FEATURES[fi]:25s}  {norm[fi]*100:5.1f}%  {bar}")


def plot_bar_chart(importance, output_dir):
    """Horizontal bar chart of grouped alpha-weighted feature importance."""
    hep.style.use("CMS")
    ranked = np.argsort(importance)
    feat_names = [FEATURES[i] for i in ranked]
    feat_vals = importance[ranked]
    feat_norm = feat_vals / feat_vals.sum() * 100
    feat_colors = [CATEGORY_COLORS[FEATURE_CATEGORIES[f]] for f in feat_names]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(feat_names)), feat_norm, color=feat_colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=11)
    ax.set_xlabel(r"$\alpha$-weighted gradient importance [%]", fontsize=13)
    ax.set_title(r"Feature importance: $\alpha \cdot |\nabla_x (P_{\mathrm{H+c}} - P_{\mathrm{bkg}})|$", fontsize=14)
    legend_elements = [Patch(facecolor=CATEGORY_COLORS[cat], edgecolor='black', label=cat)
                       for cat in CATEGORY_COLORS]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)
    hep.cms.label("Work in Progress", data=True, lumi=26.7, year="2022postEE", com=13.6, ax=ax)
    plt.tight_layout()
    outpath = os.path.join(output_dir, "feature_importance_alpha_weighted.png")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_heatmap(all_grads, alphas, output_dir):
    """Heatmap of per-class alpha*|grad|."""
    hep.style.use("CMS")
    bkg_indices = list(range(0, NUM_TRUTHS))
    bkg_indices.remove(signal_index)
    bkg_sorted = sorted(bkg_indices, key=lambda i: alphas[i], reverse=True)

    total_imp = np.zeros(len(FEATURES))
    total_alpha = 0
    for ci in bkg_indices:
        total_imp += alphas[ci] * all_grads[ci]
        total_alpha += alphas[ci]
    total_imp /= total_alpha
    feat_order = np.argsort(total_imp)

    matrix = np.zeros((len(FEATURES), len(bkg_sorted)))
    for col_j, ci in enumerate(bkg_sorted):
        matrix[:, col_j] = alphas[ci] * all_grads[ci]
    matrix = matrix[feat_order, :]

    col_labels = [f"{CLASSES[ci]}\n({alphas[ci]:.2f})" for ci in bkg_sorted]
    row_labels = [FEATURES[i] for i in feat_order]

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Background class (alpha)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(r"Per-class $\alpha \cdot |\nabla_x (P_{\mathrm{H+c}} - P_{\mathrm{class}})|$", fontsize=14)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\alpha \cdot \langle|\mathrm{grad}|\rangle$", fontsize=12)

    norm_matrix = matrix / matrix.max()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = 'white' if norm_matrix[i, j] > 0.6 else 'black'
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha='center', va='center', fontsize=6, color=color)

    hep.cms.label("Work in Progress", data=True, lumi=26.7, year="2022postEE", com=13.6, ax=ax)
    plt.tight_layout()
    outpath = os.path.join(output_dir, "feature_importance_alpha_heatmap.png")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Alpha-weighted gradient feature importance")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True, help="Directory with per-process parquet files")
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    print("Loading data...")
    X, truth = load_data(args.data_dir)
    print(f"Total events: {len(X)}")

    print("Loading model...")
    net, ckpt = load_model(args.model_path)
    cos_sims, alphas = get_alphas(ckpt, args.tau)

    np.random.seed(42)
    sig_X = X[truth == 0]

    # Compute per-class gradients
    all_grads = {}
    for cls_idx in range(NUM_TRUTHS):
        if cls_idx == signal_index:
            continue
        bkg_X = X[truth == cls_idx]
        if len(bkg_X) > args.max_events:
            bkg_X = bkg_X[np.random.choice(len(bkg_X), args.max_events, replace=False)]
        all_grads[cls_idx] = compute_grad(net, sig_X, bkg_X, cls_idx)

    # Per-class
    print(f"\n{'=' * 70}")
    print(f"Per-class alpha*Grad (top {args.top_k})")
    print(f"{'=' * 70}\n")
    for cls_idx in range(NUM_TRUTHS):
        if cls_idx == signal_index:
            continue
        print(f"hplusc vs {CLASSES[cls_idx]} (alpha={alphas[cls_idx]:.3f}, cos={cos_sims[cls_idx]:+.3f}):")
        print_ranking(all_grads[cls_idx], CLASSES[cls_idx], top_k=args.top_k)
        print()

    # Grouped
    groups = {
        'higgs bkg': [i for i in range(1) if i != signal_index],
        'non-higgs bkg': list(range(2,NUM_TRUTHS)),
        'all bkg': [i for i in range(NUM_TRUTHS) if i != signal_index],
    }

    print(f"{'=' * 70}")
    print(f"Grouped alpha-weighted Grad (top {args.top_k})")
    print(f"{'=' * 70}\n")

    grouped_importance = {}
    for gname, gidx in groups.items():
        g = np.zeros(NUM_inputs)
        gt = 0
        print(gidx)
        for ci in gidx:
            g += alphas[ci] * all_grads[ci]
            gt += alphas[ci]
        g /= gt
        grouped_importance[gname] = g

        print(f"{gname}:")
        print_ranking(g, gname, top_k=args.top_k)
        print()

    # Generate plots
    os.makedirs(args.output_dir, exist_ok=True)
    plot_bar_chart(grouped_importance['all bkg'], args.output_dir)
    plot_heatmap(all_grads, alphas, args.output_dir)


if __name__ == "__main__":
    main()

