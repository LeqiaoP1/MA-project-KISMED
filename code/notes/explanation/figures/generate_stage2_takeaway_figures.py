#!/usr/bin/env python3
"""Generate stakeholder-ready PNG figures for the four Stage-2 evaluation takeaway notes.

All values are loaded directly from archived Fold-0 evaluation JSON artifacts under
``code/notes/eval/baselines/tir_resp/artifacts/`` and the recorded tables in
``code/notes/eval/baselines/tir_resp/{mode1_pretext_fold0,adr017_floor_probe_fold0_val,
adr017_pretext_fold0}.md``.

CPU-only utility (safe on login nodes). Run from repository root::

    MPLCONFIGDIR=/tmp/mpl .env/bin/python \
        code/notes/explanation/figures/generate_stage2_takeaway_figures.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_kismed")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = REPO_ROOT / "code" / "notes" / "eval" / "baselines" / "tir_resp" / "artifacts"
OUT_DIR = Path(__file__).resolve().parent

# Colorblind-friendly editorial palette
C_NAVY = "#1F3B63"
C_TEAL = "#1B8A7A"
C_BLUE = "#2E6FB8"
C_CORAL = "#C84B31"
C_AMBER = "#D9822B"
C_GRAY = "#6B7280"
C_LGRAY = "#E5E7EB"
C_BG = "#FFFFFF"
C_PASS_BG = "#E8F5E9"
C_FAIL_BG = "#FDECEC"
C_WARN_BG = "#FEF6E4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "text.color": "#1E293B",
    "figure.facecolor": C_BG,
    "axes.facecolor": C_BG,
    "savefig.dpi": 220,
    "savefig.facecolor": C_BG,
})


def _load(name: str) -> dict:
    with open(ARTIFACTS / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


def make_takeaway1():
    """Takeaway 1: 2-stream ADR-008 zero-shot sensing, rolled control & variance check."""
    m1 = _load("mode1_pretext_fold0_summary.json")
    rs = _load("remote_sensing_fold0_summary.json")
    rl = _load("remote_sensing_tir_rolled_fold0_summary.json")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.12, 1.0]}
    )
    fig.suptitle(
        "Takeaway 1: 2-Stream Zero-Shot Thermal-to-Respiration Sensing Works Without Collapse (Fold-0, n = 27)",
        fontsize=12.8, fontweight="bold", y=0.98,
    )

    # --- Panel A: 2-stream ADR-008 conditions + pending matched ADR-007 pure-MSE ---
    m1_r = m1["subject_aggregate"]["r_masked_median"]
    rs_r = rs["subject_aggregate"]["r_masked_median"]
    rl_r = rl["subject_aggregate"]["r_masked_median"]

    labels = [
        "ADR-007\nPure MSE\n(2-stream)",
        "ADR-008\nTIR-200\n(Mode 1)",
        "ADR-008\nTIR-900\n(Full TIR)",
        "ADR-008\nTIR-900-rolled\n(Shifted 2.56 s)",
    ]
    x = np.arange(len(labels))

    # Pending placeholder bar for matched 2-stream pure-MSE ADR-007 comparator
    ax1.bar(
        x[0], 0.24, width=0.56, color="#F8FAFC", edgecolor="#94A3B8",
        linewidth=1.2, linestyle="--", hatch="///", zorder=3,
    )
    ax1.text(
        x[0], 0.12, "Pending\nmatched\n2-stream\nrun",
        ha="center", va="center", fontsize=8.4, fontweight="bold", color="#475569",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CBD5E1", lw=0.8),
        zorder=5,
    )

    vals = [m1_r["median"], rs_r["median"], rl_r["median"]]
    ci_lo = [m1_r["lo"], rs_r["lo"], rl_r["lo"]]
    ci_hi = [m1_r["hi"], rs_r["hi"], rl_r["hi"]]
    colors = [C_BLUE, C_TEAL, C_CORAL]

    for j, xi in enumerate(x[1:]):
        ax1.bar(xi, vals[j], width=0.56, color=colors[j], edgecolor="#1E293B",
                linewidth=0.75, zorder=3)
        ax1.errorbar(
            xi, vals[j],
            yerr=[[vals[j] - ci_lo[j]], [ci_hi[j] - vals[j]]],
            fmt="none", ecolor="#0F172A", elinewidth=1.4, capsize=4.5, zorder=4,
        )
        ax1.text(
            xi, ci_hi[j] + 0.018, f"{vals[j]:+.3f}",
            ha="center", va="bottom", fontsize=9.4, fontweight="bold",
            color="#0F172A" if j != 2 else C_CORAL,
        )

    ax1.axhspan(0.0, 0.10, color=C_FAIL_BG, alpha=0.8, zorder=1)
    ax1.axhline(0.10, color=C_CORAL, linestyle="--", linewidth=1.1, zorder=2,
                label="ADR-008 collapse/noise floor (r < +0.10)")
    ax1.axhline(0.35, color=C_TEAL, linestyle=":", linewidth=1.3, zorder=2,
                label="ADR-008 Mode-1 target (r ≥ +0.35)")
    ax1.axvline(2.5, color="#94A3B8", linestyle="-.", linewidth=0.95, zorder=2)
    ax1.text(2.57, 0.57, "Within-model\ncontent control\n(Δ = −0.287)",
             fontsize=8.2, color=C_CORAL, va="top", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.8)
    ax1.set_ylabel("Subject-median masked Pearson r (K_resp = 0)")
    ax1.set_ylim(-0.02, 0.62)
    ax1.grid(axis="y", color=C_LGRAY, linestyle="-", linewidth=0.7, zorder=0)
    ax1.set_title("A. 2-stream zero-shot sensing vs. time-shifted control", pad=10)
    ax1.legend(loc="upper left", fontsize=8.2, frameon=True, facecolor="white", framealpha=0.94)

    # --- Panel B: Variance retention vs Correlation (why both are needed) ---
    for data, name, col, mk in [
        (m1, "ADR-008 TIR-200 (budgeted)", C_BLUE, "o"),
        (rs, "ADR-008 TIR-900 (full frame)", C_TEAL, "s"),
        (rl, "ADR-008 TIR-900-rolled (control)", C_CORAL, "^"),
    ]:
        rows = data["subject_aggregate"]["per_subject"]
        vx = [r["median_var_hat"] for r in rows]
        ry = [r["median_r_masked"] for r in rows]
        ax2.scatter(vx, ry, s=42, alpha=0.72, color=col, marker=mk,
                    edgecolors="white", linewidth=0.5, label=name, zorder=3)
        ax2.scatter(
            [data["subject_aggregate"]["var_hat_median"]["median"]],
            [data["subject_aggregate"]["r_masked_median"]["median"]],
            s=145, color=col, marker="D", edgecolors="#0F172A", linewidth=1.2, zorder=5,
        )

    ax2.axvspan(0.0, 0.10, color=C_FAIL_BG, alpha=0.75, zorder=1)
    ax2.axvline(0.10, color=C_CORAL, linestyle="--", linewidth=1.0, zorder=2)
    ax2.axvline(0.60, color=C_TEAL, linestyle=":", linewidth=1.1, zorder=2)
    ax2.axhline(0.10, color=C_CORAL, linestyle="--", linewidth=1.0, zorder=2)
    ax2.axhline(0.0, color="#94A3B8", linewidth=0.7, zorder=2)

    ax2.text(0.05, 0.77, "Mean\ncollapse\n(<0.10)", ha="center", va="top",
             fontsize=8, color=C_CORAL, fontweight="bold")
    ax2.text(0.615, 0.77, "Var target\n(≥0.60)", ha="left", va="top",
             fontsize=8, color=C_TEAL, fontweight="bold")
    ax2.annotate(
        "Rolled TIR keeps Var ≈ 0.63\nbut r collapses to +0.082\n→ variance alone ≠ sensing",
        xy=(0.628, 0.082), xytext=(0.15, -0.25),
        fontsize=8.3, color=C_CORAL, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF5F5", edgecolor=C_CORAL, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=1.1, connectionstyle="arc3,rad=0.15"),
        zorder=6,
    )

    ax2.set_xlabel("Subject-median predicted variance Var(ŷ) [z² units; target Var(y) = 1.0]")
    ax2.set_ylabel("Subject-median masked Pearson r")
    ax2.set_xlim(0.0, 0.92)
    ax2.set_ylim(-0.34, 0.82)
    ax2.grid(True, color=C_LGRAY, linestyle="-", linewidth=0.7, zorder=0)
    ax2.set_title("B. Per-subject Var(ŷ) vs. r (diamonds = medians)", pad=10)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.13, 0.99), fontsize=8.0,
               frameon=True, facecolor="white", framealpha=0.92)

    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.text(
        0.5, 0.005,
        "Source: code/notes/eval/baselines/tir_resp/artifacts/{mode1_pretext,remote_sensing,remote_sensing_tir_rolled}_fold0_summary.json (2-stream ADR-008).",
        ha="center", fontsize=7.8, color=C_GRAY,
    )
    out = OUT_DIR / "stage2_takeaway1_collapse_and_content_control.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def make_takeaway2():
    """Takeaway 2: Spectral peak rate error (>3 bpm) and wide subject/task spread."""
    m1 = _load("mode1_pretext_fold0_summary.json")
    rs = _load("remote_sensing_fold0_summary.json")

    m1_rows = {r["subject"]: r for r in m1["subject_aggregate"]["per_subject"]}
    rs_rows = {r["subject"]: r for r in rs["subject_aggregate"]["per_subject"]}

    # Sort subjects by Mode-1 median_r_masked ascending
    subjs = sorted(m1_rows.keys(), key=lambda s: m1_rows[s]["median_r_masked"])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.6, 6.2), gridspec_kw={"width_ratios": [1.15, 1.05]}
    )
    fig.suptitle(
        "Takeaway 2: Inter-Subject Spread and Spectral Rate Error Are the Main Remaining Gaps (ADR-008 Fold-0)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # --- Panel A: Per-subject horizontal spread ---
    y = np.arange(len(subjs))
    r_m1 = [m1_rows[s]["median_r_masked"] for s in subjs]
    r_rs = [rs_rows[s]["median_r_masked"] for s in subjs]
    bar_cols = [C_CORAL if v < 0.10 else (C_TEAL if v >= 0.35 else C_BLUE) for v in r_m1]

    ax1.barh(y, r_m1, height=0.68, color=bar_cols, alpha=0.85, edgecolor="none", zorder=3)
    ax1.scatter(r_rs, y, s=34, color="#0F172A", marker="D", zorder=4)

    ax1.axvspan(-0.30, 0.10, color=C_FAIL_BG, alpha=0.65, zorder=1)
    ax1.axvline(0.0, color="#64748B", linewidth=0.8, zorder=2)
    ax1.axvline(0.10, color=C_CORAL, linestyle="--", linewidth=1.0, zorder=2)
    ax1.axvline(0.35, color=C_TEAL, linestyle=":", linewidth=1.2, zorder=2)

    ax1.set_yticks(y)
    ax1.set_yticklabels(
        [f"{s} (n={m1_rows[s]['n_clips']})" for s in subjs],
        fontsize=7.6,
    )
    ax1.set_xlabel("Per-subject median masked Pearson r")
    ax1.set_xlim(-0.30, 0.80)
    ax1.grid(axis="x", color=C_LGRAY, linestyle="-", linewidth=0.7, zorder=0)
    ax1.set_title("A. 27 held-out subjects span r = −0.18 to +0.65 (Mode 1)", pad=10)

    leg_handles = [
        mpatches.Patch(color=C_TEAL, label="Mode-1 TIR-200 (r ≥ +0.35)"),
        mpatches.Patch(color=C_BLUE, label="Mode-1 TIR-200 (+0.10 to +0.35)"),
        mpatches.Patch(color=C_CORAL, label="Mode-1 TIR-200 (< +0.10)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#0F172A",
                   markersize=6.2, label="Full-TIR TIR-900 (budget-exempt)"),
    ]
    ax1.legend(handles=leg_handles, loc="lower right", fontsize=7.6,
               frameon=True, facecolor="white", framealpha=0.96)
    ax1.annotate(
        "3 subjects near\ncollapse (r < 0.10)",
        xy=(0.015, 2.0), xytext=(0.30, 6.2),
        fontsize=8.1, color=C_CORAL, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5F5", edgecolor=C_CORAL, lw=0.7),
        arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=1.0, connectionstyle="arc3,rad=0.15"),
        zorder=5,
    )

    # --- Panel B: Top = Peak Error vs r; Bottom = Task Distortion Tiers ---
    # Use gridspec inside ax2 position by replacing ax2 with 2 stacked axes
    pos = ax2.get_position()
    ax2.remove()
    gs_right = fig.add_gridspec(2, 1, left=0.54, right=0.98, top=0.87, bottom=0.10,
                                height_ratios=[1.55, 1.0], hspace=0.40)
    ax2_top = fig.add_subplot(gs_right[0, 0])
    ax2_bot = fig.add_subplot(gs_right[1, 0])

    # Top-right: per-subject spectral peak error vs r
    pe_m1 = [m1_rows[s]["median_peak_err_bpm"] for s in subjs]
    pe_rs = [rs_rows[s]["median_peak_err_bpm"] for s in subjs]

    ax2_top.axhspan(3.0, 12.5, color=C_FAIL_BG, alpha=0.65, zorder=1)
    ax2_top.axhspan(0.0, 1.5, color=C_PASS_BG, alpha=0.75, zorder=1)
    ax2_top.axhline(3.0, color=C_CORAL, linestyle="--", linewidth=1.0, zorder=2)
    ax2_top.axhline(1.5, color=C_TEAL, linestyle=":", linewidth=1.1, zorder=2)

    ax2_top.scatter(r_m1, pe_m1, s=40, color=C_BLUE, alpha=0.8, edgecolors="white",
                    lw=0.5, label="TIR-200 (med 4.92 bpm)", zorder=3)
    ax2_top.scatter(r_rs, pe_rs, s=36, color=C_TEAL, marker="s", alpha=0.75, edgecolors="white",
                    lw=0.5, label="TIR-900 (med 3.66 bpm)", zorder=3)

    n_fail_m1 = sum(v > 3.0 for v in pe_m1)
    n_fail_rs = sum(v > 3.0 for v in pe_rs)
    ax2_top.text(-0.22, 11.4, f"Failure band (> 3.0 bpm): {n_fail_m1}/27 (TIR-200), {n_fail_rs}/27 (TIR-900)",
                 fontsize=8.2, color=C_CORAL, fontweight="bold")
    ax2_top.text(0.76, 0.55, "Target ≤ 1.5 bpm", ha="right", fontsize=8.0,
                 color=C_TEAL, fontweight="bold")

    ax2_top.set_xlabel("Per-subject median masked Pearson r")
    ax2_top.set_ylabel("Spectral peak error (bpm)")
    ax2_top.set_xlim(-0.25, 0.78)
    ax2_top.set_ylim(0.0, 12.2)
    ax2_top.grid(True, color=C_LGRAY, linestyle="-", linewidth=0.7, zorder=0)
    ax2_top.set_title("B. Dominant breathing-rate error stays above 3.0 bpm", pad=8)
    ax2_top.legend(loc="upper right", bbox_to_anchor=(0.99, 0.88), fontsize=8.0,
                   frameon=True, facecolor="white", framealpha=0.92)

    # Bottom-right: Task-level Mode-1 medians (from mode1_pretext_fold0.md)
    tasks = ["T3", "T2", "T5", "T9", "T8", "T10", "T7", "T6", "T1", "T4"]
    t_vals = [0.56, 0.39, 0.34, 0.36, 0.32, 0.30, 0.29, 0.26, 0.25, 0.18]
    t_cols = [C_TEAL]*3 + [C_AMBER]*5 + [C_CORAL]*2
    tx = np.arange(len(tasks))
    ax2_bot.bar(tx, t_vals, width=0.62, color=t_cols, edgecolor="#1E293B", lw=0.6, zorder=3)
    for i, v in enumerate(t_vals):
        ax2_bot.text(tx[i], v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.8)
    ax2_bot.set_xticks(tx)
    ax2_bot.set_xticklabels(tasks, fontsize=8.5)
    ax2_bot.set_ylabel("Clip-median r")
    ax2_bot.set_ylim(0.0, 0.68)
    ax2_bot.grid(axis="y", color=C_LGRAY, linestyle="-", linewidth=0.7, zorder=0)
    ax2_bot.set_title("C. Task tier spread (passive ~0.43 > high-dist. ~0.31 > startle ~0.21)", pad=8)

    p_patches = [
        mpatches.Patch(color=C_TEAL, label="Passive (T2,T3,T5)"),
        mpatches.Patch(color=C_AMBER, label="High distortion (T6–T10)"),
        mpatches.Patch(color=C_CORAL, label="Expression/startle (T1,T4)"),
    ]
    ax2_bot.legend(handles=p_patches, loc="upper right", fontsize=7.6,
                   frameon=True, facecolor="white", framealpha=0.92)

    # Adjust left panel to align nicely with right gridspec
    ax1.set_position([0.11, 0.10, 0.36, 0.77])

    fig.text(
        0.5, 0.012,
        "Source: mode1_pretext_fold0_summary.json, remote_sensing_fold0_summary.json, and task decomposition in mode1_pretext_fold0.md (n = 1,742 clips, 27 subjects).",
        ha="center", fontsize=7.8, color=C_GRAY,
    )
    out = OUT_DIR / "stage2_takeaway2_subject_spread_and_rate_error.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def make_takeaway3():
    """Takeaway 3: ADR-017 closes Mode-2 shortcut (G0/G2 PASS) but ties on remote sensing."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.4, 5.4), gridspec_kw={"width_ratios": [1.05, 1.15]}
    )
    fig.suptitle(
        "Takeaway 3: Masked-Span Placement (ADR-017) Closes the Inpainting Shortcut, But Ties on Remote Sensing",
        fontsize=12.8, fontweight="bold", y=0.98,
    )

    # --- Panel A: Mode-2 Inpainting — Scattered vs Masked Span (Model vs np.interp) ---
    # Values from adr017_floor_probe_fold0_val.md & adr017_pretext_fold0.md & multimae_style_panel.md
    cats = [
        "Scattered\n100/100\n(ADR-008)",
        "Scattered\n125/75\n(ADR-008)",
        "Masked-Span\n100/100\n(ADR-017)",
        "Masked-Span\n125/75\n(ADR-017)",
    ]
    interp_r = [0.9991, 0.9979, 0.4408, 0.3446]
    model_r = [0.9989, 0.9986, 0.8595, 0.8063]

    x = np.arange(len(cats))
    w = 0.34
    b1 = ax1.bar(x - w/2, interp_r, width=w, color="#94A3B8", edgecolor="#1E293B", lw=0.7,
                 label="No-video floor (np.interp)", zorder=3)
    b2 = ax1.bar(x + w/2, model_r, width=w, color=C_TEAL, edgecolor="#1E293B", lw=0.7,
                 label="Stage-2 model (subject median)", zorder=3)

    for i in range(len(cats)):
        ax1.text(x[i] - w/2, interp_r[i] + 0.018, f"{interp_r[i]:.3f}",
                 ha="center", va="bottom", fontsize=8.1, color="#475569")
        ax1.text(x[i] + w/2, model_r[i] + 0.018, f"{model_r[i]:.3f}",
                 ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=C_NAVY)

    ax1.axvline(1.5, color="#94A3B8", linestyle="--", linewidth=1.0, zorder=2)
    ax1.text(0.5, 0.22, "Shortcut regime:\nnp.interp alone gets r ≈ 0.999\n(MSE 0.002–0.005)\n→ no video needed",
             ha="center", va="center", fontsize=8.1, color="#475569",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"))
    ax1.annotate(
        "G2 PASS: paired Δ\n+0.357 / +0.403\n(95% CI > 0)\nnp.interp MSE > const",
        xy=(2.5, 0.62), xytext=(2.5, 0.14), ha="center",
        fontsize=8.1, color=C_TEAL, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=C_PASS_BG, edgecolor=C_TEAL, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.0),
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, fontsize=8.5)
    ax1.set_ylabel("Masked Pearson r")
    ax1.set_ylim(0.0, 1.14)
    ax1.grid(axis="y", color=C_LGRAY, linestyle="-", linewidth=0.7, zorder=0)
    ax1.set_title("A. Mode-2 inpainting: shortcut closed by 2–4 s masked spans", pad=10)
    ax1.legend(loc="upper right", fontsize=8.1, frameon=True, facecolor="white")

    # --- Panel B: Paired Clip-Aligned Delta (ADR-017 − ADR-008) on Zero-Resp Sensing ---
    # Exact paired values from adr017_pretext_fold0.md (1742/1742 matched clips, 27 subjects)
    rows = [
        ("TIR-900  r_masked\n(49.3% clip wins)", +0.0078, -0.0219, +0.0511, C_GRAY, "Tie (CI spans 0)"),
        ("Mode-1 TIR-200  r_masked\n(43.7% clip wins)", -0.0283, -0.0501, -0.0113, C_CORAL, "Slightly worse"),
        ("TIR-900-rolled  r_masked\n(47.7% clip wins)", -0.0062, -0.0485, +0.0368, C_GRAY, "Tie (control)"),
        ("TIR-900  Var(ŷ)\n(32.5% clip wins)", -0.1053, -0.1318, -0.0961, C_AMBER, "Amplitude shrinkage"),
        ("TIR-900-rolled  Var(ŷ)\n(35.2% clip wins)", -0.0988, -0.1293, -0.0559, C_AMBER, "Same shrinkage\nin rolled control"),
    ]
    y2 = np.arange(len(rows))[::-1]

    ax2.axvline(0.0, color="#0F172A", linestyle="--", linewidth=1.1, zorder=2)
    ax2.axvspan(-0.16, 0.0, color="#FFF7F5", alpha=0.45, zorder=0)
    ax2.axvspan(0.0, 0.12, color="#F0FDF9", alpha=0.45, zorder=0)

    for idx, (lbl, est, lo, hi, col, tag) in enumerate(rows):
        yy = y2[idx]
        ax2.errorbar(
            est, yy, xerr=[[est - lo], [hi - est]],
            fmt="o", color=col, ecolor=col, markersize=7.5, elinewidth=2.0, capsize=4, zorder=4,
        )
        if idx == 1:
            # Place left of error bar so it stays clear of the x=0 line
            ax2.text(
                lo - 0.006, yy + 0.12, f"{est:+.4f}",
                ha="right", va="center", fontsize=8.4, fontweight="bold", color=col,
            )
            ax2.text(
                lo - 0.006, yy - 0.18, tag,
                ha="right", va="center", fontsize=7.8, color="#475569",
            )
        else:
            ax2.text(
                hi + 0.006, yy + 0.12, f"{est:+.4f}",
                ha="left", va="center", fontsize=8.4, fontweight="bold", color=col,
            )
            ax2.text(
                hi + 0.006, yy - 0.18, tag,
                ha="left", va="center", fontsize=7.8, color="#475569",
            )

    ax2.axhline(1.5, color="#CBD5E1", linestyle=":", linewidth=1.0, zorder=1)
    ax2.text(-0.155, 4.35, "← ADR-008 better / higher", fontsize=8.0, color="#64748B", ha="left")
    ax2.text(0.125, 4.35, "ADR-017 better / higher →", fontsize=8.0, color="#64748B", ha="right")

    ax2.set_yticks(y2)
    ax2.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax2.set_xlabel("Paired subject-median Δ (ADR-017 − ADR-008) with 95% bootstrap CI")
    ax2.set_xlim(-0.16, 0.13)
    ax2.set_ylim(-0.6, 4.65)
    ax2.grid(axis="x", color=C_LGRAY, linestyle="-", linewidth=0.7, zorder=0)
    ax2.set_title("B. Zero-respiration (K_resp = 0): paired contrast", pad=10)

    fig.tight_layout(rect=[0, 0.025, 1, 0.93])
    fig.text(
        0.5, 0.006,
        "Source: adr017_floor_probe_fold0_val.md (val n=1,194 clips) & adr017_pretext_fold0.md (test n=1,742 matched clips, 27 subjects, 5 draws, seed 0).",
        ha="center", fontsize=7.8, color=C_GRAY,
    )
    out = OUT_DIR / "stage2_takeaway3_adr017_inpainting_vs_sensing.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def _box(ax, x, y, w, h, title, body, fc, ec, title_color="#0F172A"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h*0.72, title, ha="center", va="center",
            fontsize=9.3, fontweight="bold", color=title_color, zorder=4)
    ax.text(x + w/2, y + h*0.30, body, ha="center", va="center",
            fontsize=8.0, color="#334155", linespacing=1.28, zorder=4)


def make_takeaway4():
    """Takeaway 4: Stage-2 Clip Diagnostic vs Stage-3 Normative ADR-009 Gate + Readiness."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.6, 5.6), gridspec_kw={"width_ratios": [1.35, 0.95]}
    )
    fig.suptitle(
        "Takeaway 4: Current Stage-2 Numbers Are Fold-0 Diagnostics — Not a Scientific Go/No-Go Claim",
        fontsize=12.8, fontweight="bold", y=0.98,
    )

    # --- Panel A: Side-by-side pipeline diagram ---
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title("A. Evaluation Protocol Gap: What Exists Today vs. What Gates Adoption", pad=6)

    # Column headers
    ax1.text(0.24, 0.95, "CURRENT: Stage-2 Clip Probe", ha="center", va="center",
             fontsize=10.2, fontweight="bold", color=C_BLUE)
    ax1.text(0.24, 0.90, "(runners/run_eval_recon.py — Non-gating)", ha="center", va="center",
             fontsize=8.2, color=C_GRAY)

    ax1.text(0.76, 0.95, "REQUIRED: Stage-3 ADR-009 Gate", ha="center", va="center",
             fontsize=10.2, fontweight="bold", color=C_TEAL)
    ax1.text(0.76, 0.90, "(run_waveform.py → run_evaluate_kfold.py)", ha="center", va="center",
             fontsize=8.2, color=C_GRAY)

    # Divider
    ax1.plot([0.50, 0.50], [0.04, 0.96], color="#CBD5E1", linestyle="--", lw=1.0)

    # Left column boxes (Stage-2)
    left_steps = [
        ("1. Data Scope", "Fold 0 test only (27 subjects, 1,742 clips)\nFrozen Stage-2 encoder + decoder", "#EFF6FF", C_BLUE),
        ("2. Unit of Scoring", "Independent 8 s clips (no session stitching)\nNo resp_cache hole-mask alignment", "#EFF6FF", C_BLUE),
        ("3. Metrics", "Unfiltered r_masked, Var(ŷ), peak_err_bpm\nNo 0.10–0.60 Hz bandpass or single-tone floor", "#EFF6FF", C_BLUE),
        ("4. Decision Power", "Diagnostic health check ONLY\n(collapse check, content control, inpainting)", C_WARN_BG, C_AMBER),
    ]
    ys = [0.70, 0.49, 0.28, 0.07]
    for i, (t, b, fc, ec) in enumerate(left_steps):
        _box(ax1, 0.03, ys[i], 0.42, 0.145, t, b, fc, ec, title_color=C_NAVY if i < 3 else "#92400E")
        if i < 3:
            ax1.annotate("", xy=(0.24, ys[i+1] + 0.162), xytext=(0.24, ys[i] - 0.015),
                         arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.4))

    # Right column boxes (Stage-3 ADR-009)
    right_steps = [
        ("1. Data Scope", "All 5 subject-disjoint folds (~139 subjects)\nIdentical Stage-3 fine-tune vs. ADR-007 pure-MSE", "#ECFDF5", C_TEAL),
        ("2. Session Assembly", "Hann overlap-add into continuous sessions\nANDed with ADR-003 resp_cache validity mask", "#ECFDF5", C_TEAL),
        ("3. Normative Metrics", "Headline: band-limited Pearson (0.10–0.60 Hz)\n+ BPM MAE, PSD MAE, beat single-tone (>50% subj)", "#ECFDF5", C_TEAL),
        ("4. Decision Power", "Scientific Go/No-Go Gate (ADR-008 §3.3)\nAdopt iff pooled Δ ≥ +0.02 (95% CI > 0)", C_PASS_BG, C_TEAL),
    ]
    for i, (t, b, fc, ec) in enumerate(right_steps):
        _box(ax1, 0.55, ys[i], 0.42, 0.145, t, b, fc, ec, title_color="#065F46")
        if i < 3:
            ax1.annotate("", xy=(0.76, ys[i+1] + 0.162), xytext=(0.76, ys[i] - 0.015),
                         arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.4))

    # --- Panel B: Readiness Matrix ---
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("B. Evidence Checklist Toward Adoption Decision", pad=6)

    items = [
        ("ADR-008 Fold-0 Stage-2 pretext probes", "DONE", C_TEAL, C_PASS_BG,
         "Mode-1 +0.326, TIR-900 +0.382, rolled +0.082"),
        ("ADR-017 Fold-0 floor (G0) & pretext (G2/G3)", "DONE", C_TEAL, C_PASS_BG,
         "Inpainting PASS; remote sensing tied (+0.008)"),
        ("Matched Fold-0 ADR-007 pure-MSE baseline", "MISSING", C_CORAL, C_FAIL_BG,
         "Needed as exact denominator for ADR-008 gain"),
        ("5-fold Stage-2 pre-training (both arms)", "PENDING", C_AMBER, C_WARN_BG,
         "Only Fold 0 exists; 4 folds remaining"),
        ("5-fold Stage-3 fine-tune + ADR-009 session eval", "PENDING", C_CORAL, C_FAIL_BG,
         "Required gating milestone (Δ ≥ +0.02, 95% CI > 0)"),
    ]

    row_ys = np.linspace(0.80, 0.10, len(items))
    for yy, (name, status, col, bg, detail) in zip(row_ys, items):
        card = FancyBboxPatch(
            (0.02, yy - 0.065), 0.96, 0.135,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=0.9, zorder=2,
        )
        ax2.add_patch(card)
        badge = FancyBboxPatch(
            (0.75, yy - 0.022), 0.20, 0.058,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=bg, edgecolor=col, linewidth=1.0, zorder=3,
        )
        ax2.add_patch(badge)
        ax2.text(0.85, yy + 0.007, status, ha="center", va="center",
                 fontsize=8.0, fontweight="bold", color=col, zorder=4)
        ax2.text(0.05, yy + 0.022, name, ha="left", va="center",
                 fontsize=8.7, fontweight="bold", color="#0F172A", zorder=4)
        ax2.text(0.05, yy - 0.030, detail, ha="left", va="center",
                 fontsize=7.8, color="#475569", zorder=4)

    fig.tight_layout(rect=[0, 0.025, 1, 0.93])
    fig.text(
        0.5, 0.006,
        "Source: code/notes/design/ADR-008-composite-objective-and-cross-modal-alignment.md (§3.2–§3.3), ADR-009-resp-waveform-evaluation-protocol.md, and code/runners/run_eval_recon.py.",
        ha="center", fontsize=7.8, color=C_GRAY,
    )
    out = OUT_DIR / "stage2_takeaway4_eval_pipeline_and_gates.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_takeaway1()
    make_takeaway2()
    make_takeaway3()
    make_takeaway4()


if __name__ == "__main__":
    main()
