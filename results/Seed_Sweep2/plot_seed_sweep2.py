"""
plot_seed_sweep2.py
===================
Generate all graphs for the seed_sweep2 experiment suite.

Seeds: 45-55 (11 seeds), 22 experiments each.

Data policy
-----------
A run is considered complete only if all five expected metrics contain at
least 50 valid rounds numbered 1..50. Extra rounds are ignored for the
standardized 50-round analysis.

Incomplete runs are skipped per-seed. This includes known incomplete runs
such as Attack_SignFlip_40pct and Ablation_NoL2_ModelReplace20pct.

For final-round summaries, "final" means round 50 specifically, rather than
the last JSON entry. This keeps final-round figures consistent with the
round-curve figures.

Output figures:
  fig1_clean_baseline.png
      Clean run: mean±std of all 5 metrics over 50 rounds.

  fig2_attack_auc_heatmap.png
      Round-50 AUC heatmap: attack type × adversary percentage.

  fig3_attack_f1_heatmap.png
      Round-50 F1 heatmap: attack type × adversary percentage.

  fig4_signflip_auc_rounds.png
      AUC over rounds: clean vs SignFlip 10/20%.

  fig5_labelflip_auc_rounds.png
      AUC over rounds: clean vs LabelFlip 10/20/40%.

  fig6_modelreplace_auc_rounds.png
      AUC over rounds: clean vs ModelReplace 10/20/40%.

  fig7_ablation_auc_bar.png
      Ablation round-50 AUC: layer × condition, mean±std.

  fig8_ablation_f1_bar.png
      Ablation round-50 F1: layer × condition, mean±std.

  fig9_ablation_recall_bar.png
      Ablation round-50 Recall: layer × condition, mean±std.

  fig10_seed_variance.png
      Per-seed round-50 AUC for clean + SignFlip20 + LabelFlip20
      + ModelReplace20.

  fig11_precision_recall_scatter.png
      Precision vs Recall at round 50, mean over valid seeds.

  fig12_summary_table.png
      Mean±std of round-50 AUC/F1/Precision/Recall for all 22 experiments.
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SWEEP_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ── Experiment configuration ──────────────────────────────────────────────────

SEEDS = [f"seed_{s}" for s in range(45, 56)]  # seed_45 … seed_55

METRICS = [
    "auc",
    "f1",
    "precision",
    "recall",
    "accuracy",
]

N_ROUNDS = 50
EXPECTED_ROUNDS = list(range(1, N_ROUNDS + 1))

# Population standard deviation is retained for consistency with the original
# script. Change to 1 if you want sample SD across the 11 seeds.
STD_DDOF = 0

# Whether to require all five metrics for a run to be considered complete.
# True gives the most consistent cross-figure dataset.
REQUIRE_ALL_METRICS = True


# ── Helpers ───────────────────────────────────────────────────────────────────


def _metric_pairs(d: dict, metric: str) -> list[tuple[int, float]]:
    """
    Return sorted, finite (round, value) pairs for one metric.

    Invalid/malformed entries are ignored rather than crashing the entire
    plotting run.
    """
    md = d.get("metrics_distributed", {})
    raw = md.get(metric, [])

    if not isinstance(raw, list):
        return []

    pairs = []

    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue

        try:
            rnd = int(item[0])
            value = float(item[1])
        except (TypeError, ValueError):
            continue

        if not np.isfinite(value):
            continue

        pairs.append((rnd, value))

    pairs.sort(key=lambda x: x[0])
    return pairs


def _standardized_metric_values(
    d: dict,
    metric: str,
) -> np.ndarray | None:
    """
    Return exactly 50 values corresponding to rounds 1..50.

    Returns None if any expected round is missing or invalid.
    """
    pairs = _metric_pairs(d, metric)

    # Build a round -> value mapping.
    # If duplicate rounds exist, the last sorted occurrence wins.
    by_round = {}
    for rnd, value in pairs:
        by_round[rnd] = value

    if not all(rnd in by_round for rnd in EXPECTED_ROUNDS):
        return None

    values = np.array(
        [by_round[rnd] for rnd in EXPECTED_ROUNDS],
        dtype=float,
    )

    if len(values) != N_ROUNDS:
        return None

    if not np.all(np.isfinite(values)):
        return None

    return values


def load(seed: str, exp: str) -> dict | None:
    """
    Load one seed/experiment.

    A run is complete only if all required metrics contain valid values for
    every round 1..50.

    Extra rounds beyond round 50 are ignored.
    """
    fp = os.path.join(SWEEP_DIR, seed, f"{exp}.json")

    if not os.path.exists(fp):
        return None

    try:
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        print(f"  WARNING: Could not read {fp}")
        return None

    if not isinstance(d, dict):
        print(f"  WARNING: Invalid JSON structure: {fp}")
        return None

    metrics_to_check = METRICS if REQUIRE_ALL_METRICS else ["auc"]

    for metric in metrics_to_check:
        values = _standardized_metric_values(d, metric)
        if values is None:
            return None

    return d


def final(d: dict, metric: str) -> float:
    """
    Return the value specifically at round 50.

    Returns NaN if round 50 is unavailable or invalid.
    """
    values = _standardized_metric_values(d, metric)

    if values is None:
        return float("nan")

    return float(values[N_ROUNDS - 1])


def rounds_values(
    d: dict,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return rounds 1..50 and corresponding standardized metric values.
    """
    values = _standardized_metric_values(d, metric)

    if values is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    return (
        np.arange(1, N_ROUNDS + 1),
        values,
    )


def collect_final(exp: str, metric: str) -> list[float]:
    """
    Collect round-50 metric values across complete seeds.

    A run must pass the complete-run validation in load().
    """
    vals = []

    for seed in SEEDS:
        d = load(seed, exp)

        if d is None:
            continue

        value = final(d, metric)

        if np.isfinite(value):
            vals.append(value)

    return vals


def collect_curves(
    exp: str,
    metric: str,
) -> np.ndarray | None:
    """
    Return shape (n_complete_seeds, 50) containing rounds 1..50.

    Returns None if no complete seeds have the experiment.
    """
    rows = []

    for seed in SEEDS:
        d = load(seed, exp)

        if d is None:
            continue

        _, values = rounds_values(d, metric)

        if len(values) == N_ROUNDS:
            rows.append(values)

    if not rows:
        return None

    return np.vstack(rows)


def mean_std(
    values: list[float] | np.ndarray,
) -> tuple[float, float]:
    """Return finite mean and population/sample SD."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return float("nan"), float("nan")

    return (
        float(np.mean(arr)),
        float(np.std(arr, ddof=STD_DDOF)),
    )


def plot_mean_std(
    ax,
    curves: np.ndarray | None,
    label: str,
    color: str,
    linestyle: str = "-",
):
    """Plot mean curve with ±1 SD shading."""
    if curves is None or len(curves) == 0:
        return

    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0, ddof=STD_DDOF)

    xs = np.arange(1, N_ROUNDS + 1)

    ax.plot(
        xs,
        mean,
        color=color,
        label=label,
        linewidth=1.8,
        linestyle=linestyle,
    )

    ax.fill_between(
        xs,
        mean - std,
        mean + std,
        color=color,
        alpha=0.15,
    )


def savefig(fname: str):
    """Save the current figure."""
    out = os.path.join(FIG_DIR, fname)

    plt.tight_layout()
    plt.savefig(
        out,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"  Saved -> {out}")


def annotate_sample_count(ax, exp: str):
    """Optionally report the number of complete seeds in the console."""
    n = sum(1 for seed in SEEDS if load(seed, exp) is not None)

    print(f"    {exp}: n={n}/{len(SEEDS)} complete")


# ── Fig 1: Clean baseline ─────────────────────────────────────────────────────


def fig1_clean_baseline():
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    fig.suptitle(
        f"Clean Baseline — Mean ± Std over {len(SEEDS)} Seeds",
        fontsize=13,
        fontweight="bold",
    )

    style = [
        ("auc", "AUC", "royalblue", ax1),
        ("f1", "F1", "darkorange", ax1),
        ("precision", "Precision", "purple", ax2),
        ("recall", "Recall", "crimson", ax2),
        ("accuracy", "Accuracy", "seagreen", ax2),
    ]

    for metric, label, color, ax in style:
        curves = collect_curves(
            "CleanRun_Simple",
            metric,
        )

        if curves is None:
            continue

        plot_mean_std(
            ax,
            curves,
            label,
            color,
        )

    for ax, title in [
        (ax1, "AUC & F1"),
        (ax2, "Precision / Recall / Accuracy"),
    ]:
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xlim(1, N_ROUNDS)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    annotate_sample_count(
        None,
        "CleanRun_Simple",
    ) if False else None

    savefig("fig1_clean_baseline.png")


# ── Fig 2 & 3: Attack heatmaps ────────────────────────────────────────────────


def fig2_fig3_attack_heatmaps():
    attacks = [
        "SignFlip",
        "LabelFlip",
        "ModelReplace",
    ]

    ratios = [
        "10pct",
        "20pct",
        "40pct",
    ]

    for metric, figname, title in [
        (
            "auc",
            "fig2_attack_auc_heatmap.png",
            "Round-50 AUC — Attack Robustness (Mean over Seeds)",
        ),
        (
            "f1",
            "fig3_attack_f1_heatmap.png",
            "Round-50 F1 — Attack Robustness (Mean over Seeds)",
        ),
    ]:
        data = np.full(
            (len(attacks), len(ratios)),
            np.nan,
        )

        counts = np.zeros(
            (len(attacks), len(ratios)),
            dtype=int,
        )

        for i, atk in enumerate(attacks):
            for j, pct in enumerate(ratios):
                exp = f"Attack_{atk}_{pct}"

                vals = collect_final(
                    exp,
                    metric,
                )

                if vals:
                    data[i, j] = np.mean(vals)
                    counts[i, j] = len(vals)

                print(f"    {exp} / {metric}: n={counts[i, j]}/{len(SEEDS)}")

        fig, ax = plt.subplots(
            figsize=(8, 5),
        )

        fig.suptitle(
            title,
            fontsize=12,
            fontweight="bold",
        )

        masked = np.ma.masked_invalid(data)

        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad(color="lightgray")

        im = ax.imshow(
            masked,
            vmin=0,
            vmax=1,
            cmap=cmap,
            aspect="auto",
        )

        plt.colorbar(
            im,
            ax=ax,
            label=metric.upper(),
        )

        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels(
            ["10% adv", "20% adv", "40% adv"],
        )

        ax.set_yticks(range(len(attacks)))
        ax.set_yticklabels(attacks)

        for i in range(len(attacks)):
            for j in range(len(ratios)):
                if np.isnan(data[i, j]):
                    ax.text(
                        j,
                        i,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                        fontweight="bold",
                    )
                else:
                    value = data[i, j]

                    text_color = "black" if value > 0.4 else "white"

                    ax.text(
                        j,
                        i,
                        f"{value:.3f}\n(n={counts[i, j]})",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color,
                    )

        ax.set_xlabel("Adversary fraction")
        ax.set_ylabel("Attack type")

        savefig(figname)


# ── Fig 4-6: Attack AUC over rounds ───────────────────────────────────────────


def fig_attack_rounds(
    attack_label: str,
    attack_key: str,
    ratios: list[str],
    figname: str,
):
    colors = [
        "gold",
        "darkorange",
        "darkred",
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    fig.suptitle(
        f"AUC over Rounds — {attack_label} Attack (Mean ± Std over Seeds)",
        fontsize=12,
        fontweight="bold",
    )

    clean_curves = collect_curves(
        "CleanRun_Simple",
        "auc",
    )

    plot_mean_std(
        ax,
        clean_curves,
        "Clean (0% adv)",
        "royalblue",
    )

    for pct_str, color in zip(ratios, colors):
        exp = f"Attack_{attack_key}_{pct_str}"

        curves = collect_curves(
            exp,
            "auc",
        )

        pct_label = pct_str.replace(
            "pct",
            "%",
        )

        plot_mean_std(
            ax,
            curves,
            f"{attack_label} {pct_label} adv",
            color,
            linestyle="--",
        )

        n = 0 if curves is None else len(curves)

        print(f"    {exp} / auc: n={n}/{len(SEEDS)}")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("AUC")
    ax.set_xlim(1, N_ROUNDS)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    savefig(figname)


# ── Fig 7-9: Ablation grouped bar charts ──────────────────────────────────────


def fig_ablation_bar(
    metric: str,
    ylabel: str,
    figname: str,
    title: str,
):
    layers = [
        "NoL1",
        "NoL2",
        "NoL3",
    ]

    conditions = [
        ("Clean", "Clean", "royalblue"),
        ("SignFlip20pct", "SignFlip 20%", "darkorange"),
        ("LabelFlip20pct", "LabelFlip 20%", "seagreen"),
        ("ModelReplace20pct", "ModelReplace 20%", "crimson"),
    ]

    full_model_vals = collect_final(
        "CleanRun_Simple",
        metric,
    )

    full_model_mean, full_model_std = mean_std(
        full_model_vals,
    )

    x = np.arange(len(layers))
    n_conditions = len(conditions)

    width = 0.18

    offsets = (np.arange(n_conditions) - (n_conditions - 1) / 2) * width

    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    fig.suptitle(
        title,
        fontsize=12,
        fontweight="bold",
    )

    for offset, (
        cond_key,
        cond_label,
        color,
    ) in zip(offsets, conditions):
        means = []
        stds = []
        ns = []

        for layer in layers:
            exp = f"Ablation_{layer}_{cond_key}"

            values = collect_final(
                exp,
                metric,
            )

            m, s = mean_std(values)

            means.append(m)
            stds.append(s)
            ns.append(len(values))

            print(f"    {exp} / {metric}: mean={m:.4f}, std={s:.4f}, n={len(values)}/{len(SEEDS)}")

        means = np.asarray(means)
        stds = np.asarray(stds)

        bars = ax.bar(
            x + offset,
            means,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label=cond_label,
            yerr=stds,
            capsize=3,
            error_kw={
                "elinewidth": 0.8,
                "capthick": 0.8,
            },
        )

        for bar, m, s in zip(
            bars,
            means,
            stds,
        ):
            if np.isfinite(m):
                # Keep labels above the bar/error bar.
                label_y = m

                if np.isfinite(s):
                    label_y += s

                label_y += 0.012

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{m:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # Full-model reference line.
    if np.isfinite(full_model_mean):
        ax.axhline(
            full_model_mean,
            color="black",
            linewidth=1.2,
            linestyle=":",
            label=(f"Full model ({full_model_mean:.3f} ± {full_model_std:.3f})"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "No Layer 1",
            "No Layer 2",
            "No Layer 3",
        ],
        fontsize=11,
    )

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.15)
    ax.legend(
        fontsize=9,
        loc="lower right",
    )
    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    savefig(figname)


# ── Fig 10: Per-seed AUC ───────────────────────────────────────────────────────


def fig10_seed_variance():
    scenarios = [
        (
            "CleanRun_Simple",
            "Clean",
            "royalblue",
        ),
        (
            "Attack_SignFlip_20pct",
            "SignFlip 20%",
            "darkorange",
        ),
        (
            "Attack_LabelFlip_20pct",
            "LabelFlip 20%",
            "seagreen",
        ),
        (
            "Attack_ModelReplace_20pct",
            "ModelReplace 20%",
            "crimson",
        ),
    ]

    seed_nums = [int(s.split("_")[1]) for s in SEEDS]

    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    fig.suptitle(
        "Per-Seed Round-50 AUC — Key Scenarios",
        fontsize=13,
        fontweight="bold",
    )

    for exp, label, color in scenarios:
        vals = []

        for seed in SEEDS:
            d = load(seed, exp)

            if d is None:
                vals.append(np.nan)
                continue

            value = final(
                d,
                "auc",
            )

            vals.append(value if np.isfinite(value) else np.nan)

        ax.plot(
            seed_nums,
            vals,
            marker="o",
            color=color,
            label=label,
            linewidth=1.5,
            markersize=6,
        )

        print(f"    {exp} / auc: n={np.sum(np.isfinite(vals))}/{len(SEEDS)}")

    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Round-50 AUC")
    ax.set_xticks(seed_nums)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    savefig("fig10_seed_variance.png")


# ── Fig 11: Precision-Recall scatter ──────────────────────────────────────────


def fig11_precision_recall_scatter():
    scenarios = [
        (
            "CleanRun_Simple",
            "Clean",
            "o",
            "royalblue",
            100,
        ),
        (
            "Attack_SignFlip_10pct",
            "SignFlip 10%",
            "s",
            "gold",
            70,
        ),
        (
            "Attack_SignFlip_20pct",
            "SignFlip 20%",
            "s",
            "darkorange",
            70,
        ),
        (
            "Attack_LabelFlip_10pct",
            "LabelFlip 10%",
            "^",
            "lightgreen",
            70,
        ),
        (
            "Attack_LabelFlip_20pct",
            "LabelFlip 20%",
            "^",
            "seagreen",
            70,
        ),
        (
            "Attack_LabelFlip_40pct",
            "LabelFlip 40%",
            "^",
            "darkgreen",
            70,
        ),
        (
            "Attack_ModelReplace_10pct",
            "ModelReplace 10%",
            "D",
            "plum",
            70,
        ),
        (
            "Attack_ModelReplace_20pct",
            "ModelReplace 20%",
            "D",
            "orchid",
            70,
        ),
        (
            "Attack_ModelReplace_40pct",
            "ModelReplace 40%",
            "D",
            "purple",
            70,
        ),
        (
            "Ablation_NoL1_Clean",
            "No L1 (clean)",
            "P",
            "steelblue",
            70,
        ),
        (
            "Ablation_NoL2_Clean",
            "No L2 (clean)",
            "P",
            "cornflowerblue",
            70,
        ),
        (
            "Ablation_NoL3_Clean",
            "No L3 (clean)",
            "P",
            "deepskyblue",
            70,
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(10, 8),
    )

    fig.suptitle(
        "Precision vs Recall — Round 50, Mean over Seeds",
        fontsize=13,
        fontweight="bold",
    )

    for (
        exp,
        label,
        marker,
        color,
        size,
    ) in scenarios:
        ps = collect_final(
            exp,
            "precision",
        )

        rs = collect_final(
            exp,
            "recall",
        )

        if not ps or not rs:
            continue

        p_mean, p_std = mean_std(ps)
        r_mean, r_std = mean_std(rs)

        ax.scatter(
            r_mean,
            p_mean,
            marker=marker,
            color=color,
            s=size,
            edgecolors="black",
            linewidths=0.6,
            label=(f"{label}  P={p_mean:.3f} R={r_mean:.3f}"),
            zorder=3,
        )

        # Show seed variation around the mean where available.
        ax.errorbar(
            r_mean,
            p_mean,
            xerr=r_std,
            yerr=p_std,
            fmt="none",
            ecolor=color,
            alpha=0.45,
            capsize=2,
            linewidth=0.8,
            zorder=2,
        )

    # Iso-F1 curves.
    rec_grid = np.linspace(
        0.01,
        1.0,
        300,
    )

    for f1_val in [
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
    ]:
        with np.errstate(
            invalid="ignore",
            divide="ignore",
        ):
            denominator = 2 * rec_grid - f1_val

            p_iso = f1_val * rec_grid / denominator

        mask = (p_iso >= 0) & (p_iso <= 1) & np.isfinite(p_iso)

        ax.plot(
            rec_grid[mask],
            p_iso[mask],
            color="lightgrey",
            linewidth=0.8,
            linestyle="--",
            zorder=1,
        )

        idx = np.where(mask)[0]

        if len(idx):
            mid = idx[len(idx) // 2]

            ax.annotate(
                f"F1={f1_val:.1f}",
                (
                    rec_grid[mid],
                    p_iso[mid],
                ),
                fontsize=7,
                color="grey",
                ha="center",
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)

    ax.legend(
        fontsize=7,
        loc="upper right",
        ncol=2,
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    savefig(
        "fig11_precision_recall_scatter.png",
    )


# ── Fig 12: Summary table ──────────────────────────────────────────────────────


def fig12_summary_table():
    """
    Text-table figure:
    mean±std of round-50 AUC/F1/Precision/Recall for all 22 experiments.
    """

    experiments = [
        "CleanRun_Simple",
        "Attack_SignFlip_10pct",
        "Attack_SignFlip_20pct",
        "Attack_SignFlip_40pct",
        "Attack_LabelFlip_10pct",
        "Attack_LabelFlip_20pct",
        "Attack_LabelFlip_40pct",
        "Attack_ModelReplace_10pct",
        "Attack_ModelReplace_20pct",
        "Attack_ModelReplace_40pct",
        "Ablation_NoL1_Clean",
        "Ablation_NoL1_SignFlip20pct",
        "Ablation_NoL1_LabelFlip20pct",
        "Ablation_NoL1_ModelReplace20pct",
        "Ablation_NoL2_Clean",
        "Ablation_NoL2_SignFlip20pct",
        "Ablation_NoL2_LabelFlip20pct",
        "Ablation_NoL2_ModelReplace20pct",
        "Ablation_NoL3_Clean",
        "Ablation_NoL3_SignFlip20pct",
        "Ablation_NoL3_LabelFlip20pct",
        "Ablation_NoL3_ModelReplace20pct",
    ]

    cols = [
        "AUC",
        "F1",
        "Prec",
        "Rec",
        "n",
    ]

    col_keys = [
        "auc",
        "f1",
        "precision",
        "recall",
    ]

    rows = []

    for exp in experiments:
        row = [exp]

        for metric in col_keys:
            vals = collect_final(
                exp,
                metric,
            )

            if vals:
                m, s = mean_std(vals)

                row.append(f"{m:.3f}±{s:.3f}")
            else:
                row.append("—")

        n = sum(1 for seed in SEEDS if load(seed, exp) is not None)

        row.append(str(n))

        rows.append(row)

    fig, ax = plt.subplots(
        figsize=(16, 10),
    )

    fig.suptitle(
        "Seed Sweep Summary — Round-50 Mean ± Std (seeds 45–55)",
        fontsize=12,
        fontweight="bold",
    )

    ax.axis("off")

    col_headers = ["Experiment"] + cols

    table = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.4)

    # Header.
    for j in range(len(col_headers)):
        table[0, j].set_facecolor("#2c5f9e")
        table[0, j].set_text_props(
            color="white",
            fontweight="bold",
        )

    # Alternating row shading.
    for i in range(
        1,
        len(rows) + 1,
    ):
        for j in range(len(col_headers)):
            table[i, j].set_facecolor("#f0f4f8" if i % 2 == 0 else "white")

    savefig(
        "fig12_summary_table.png",
    )


# ── Main ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("\n=== Generating seed_sweep2 figures ===\n")

    print("Configuration:")
    print(f"  Seeds: {SEEDS[0]} … {SEEDS[-1]}")
    print(f"  Expected rounds: 1–{N_ROUNDS}")
    print(f"  Required metrics: {', '.join(METRICS)}")
    print(f"  Require all metrics: {REQUIRE_ALL_METRICS}")
    print(f"  Standard deviation ddof: {STD_DDOF}")
    print()

    print("Fig 1: Clean baseline mean±std over rounds")
    fig1_clean_baseline()

    print("Fig 2 & 3: Attack AUC/F1 heatmaps")
    fig2_fig3_attack_heatmaps()

    print("Fig 4: SignFlip AUC over rounds")
    fig_attack_rounds(
        "SignFlip",
        "SignFlip",
        ["10pct", "20pct"],
        "fig4_signflip_auc_rounds.png",
    )

    print("Fig 5: LabelFlip AUC over rounds")
    fig_attack_rounds(
        "LabelFlip",
        "LabelFlip",
        ["10pct", "20pct", "40pct"],
        "fig5_labelflip_auc_rounds.png",
    )

    print("Fig 6: ModelReplace AUC over rounds")
    fig_attack_rounds(
        "ModelReplace",
        "ModelReplace",
        ["10pct", "20pct", "40pct"],
        "fig6_modelreplace_auc_rounds.png",
    )

    print("Fig 7: Ablation AUC bar")
    fig_ablation_bar(
        "auc",
        "AUC (round 50, mean over seeds)",
        "fig7_ablation_auc_bar.png",
        "Ablation Study — Round-50 AUC (Layer Removal × Attack Condition)",
    )

    print("Fig 8: Ablation F1 bar")
    fig_ablation_bar(
        "f1",
        "F1 (round 50, mean over seeds)",
        "fig8_ablation_f1_bar.png",
        "Ablation Study — Round-50 F1 (Layer Removal × Attack Condition)",
    )

    print("Fig 9: Ablation Recall bar")
    fig_ablation_bar(
        "recall",
        "Recall (round 50, mean over seeds)",
        "fig9_ablation_recall_bar.png",
        "Ablation Study — Round-50 Recall (Layer Removal × Attack Condition)",
    )

    print("Fig 10: Per-seed AUC variance")
    fig10_seed_variance()

    print("Fig 11: Precision-Recall scatter")
    fig11_precision_recall_scatter()

    print("Fig 12: Summary table")
    fig12_summary_table()

    print(f"\nDone. All figures saved to {FIG_DIR}")
