import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict


def export_eval_summary_png(
    qa_accuracies: list,
    run_accuracies: list,
    q_pass_counts: dict,
    qa_miss_counts: dict,
    multi_part_questions: dict,
    expected_routes: dict,
    num_multi_runs: int,
    num_qa_questions: int = 10,
    output_path: str = "eval_summary.png",
    layout: str = "standard",   # "standard" or "wide"
):
    """
    Export eval results summary table to PNG.

    Parameters
    ----------
    qa_accuracies       : list of per-run QA accuracy floats
    run_accuracies      : list of per-run routing accuracy floats
    q_pass_counts       : dict {q_key: int} — routing pass counts
    qa_miss_counts      : dict {question_str: int} — QA miss counts across runs
    multi_part_questions: dict of routing question keys (for ordering)
    expected_routes     : dict {q_key: list}
    num_multi_runs      : number of routing eval runs
    num_qa_questions    : questions sampled per QA run (default 10)
    output_path         : output file path
    """

    # ------------------------------------------------------------------ #
    #  Colors                                                              #
    # ------------------------------------------------------------------ #
    GREEN  = "#1D9E75"
    AMBER  = "#EF9F27"
    RED    = "#D85A30"
    GRAY   = "#B4B2A9"
    BG     = "#FFFFFF"
    CARD   = "#F1EFE8"
    TEXT   = "#2C2C2A"
    MUTED  = "#5F5E5A"

    # ------------------------------------------------------------------ #
    #  Compute metrics                                                     #
    # ------------------------------------------------------------------ #
    num_qa_runs = len(qa_accuracies)

    qa_metrics = {
        "Mean":    f"{np.mean(qa_accuracies):.1%}",
        "Std dev": f"±{np.std(qa_accuracies):.1%}",
        "Min":     f"{np.min(qa_accuracies):.1%}",
        "Max":     f"{np.max(qa_accuracies):.1%}",
    }
    rt_metrics = {
        "Mean":    f"{np.mean(run_accuracies):.1%}",
        "Std dev": f"±{np.std(run_accuracies):.1%}",
        "Min":     f"{np.min(run_accuracies):.1%}",
        "Max":     f"{np.max(run_accuracies):.1%}",
    }

    # Routing rows
    q_rows = []
    for q_key in sorted(multi_part_questions.keys()):
        count  = q_pass_counts.get(q_key, 0)
        rate   = count / num_multi_runs
        status = "PASS" if count == num_multi_runs else ("MISS" if count == 0 else "PART")
        color  = GREEN if status == "PASS" else (RED if status == "MISS" else AMBER)
        q_rows.append((q_key, f"{count}/{num_multi_runs}", f"{rate:.0%}", status, color, rate))

    # QA miss rows — sorted by miss count descending
    qa_miss_rows = sorted(qa_miss_counts.items(), key=lambda x: -x[1]) if qa_miss_counts else []

    missed_routing = [q for q in sorted(multi_part_questions.keys())
                      if q_pass_counts.get(q, 0) == 0]

    # ------------------------------------------------------------------ #
    #  Layout constants (all in inches)                                    #
    # ------------------------------------------------------------------ #
    if layout == "wide":
        FIG_W = 20.0   # 16:9 PowerPoint aspect (~10" tall at this width)
        DPI   = 150
    else:
        FIG_W = 13.0   # original
        DPI   = 180
    MARGIN_L  = 0.35
    MARGIN_R  = 0.35
    CONTENT_W = FIG_W - MARGIN_L - MARGIN_R

    TITLE_H   = 0.45
    SUBTITLE_H= 0.30
    GAP       = 0.18
    LABEL_H   = 0.22
    CARD_H    = 0.65
    TH_H      = 0.28
    TR_H      = 0.26
    FOOTER_H  = 0.20
    PAD_TOP   = 0.25
    PAD_BOT   = 0.35

    n_routing_rows = len(q_rows)
    n_qa_miss_rows = len(qa_miss_rows)

    # QA miss table block height (only added if there are misses)
    qa_miss_block_h = (GAP + LABEL_H + TH_H + n_qa_miss_rows * TR_H) if n_qa_miss_rows else 0

    FIG_H = (PAD_TOP + TITLE_H + SUBTITLE_H
             + GAP + LABEL_H + CARD_H          # QA metric cards
             + GAP + LABEL_H + CARD_H          # routing metric cards
             + GAP + LABEL_H + TH_H + n_routing_rows * TR_H   # routing table
             + qa_miss_block_h                  # QA miss table
             + FOOTER_H + PAD_BOT)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    ax.set_facecolor(BG)

    y = FIG_H - PAD_TOP

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def rect(x, y_bottom, w, h, color, radius=0.06):
        ax.add_patch(FancyBboxPatch(
            (x, y_bottom), w, h,
            boxstyle="round,pad=0" if radius == 0 else f"round,pad={radius}",
            facecolor=color, edgecolor="none", clip_on=False
        ))

    def hline(y_pos, lw=0.5, color=GRAY, alpha=1.0):
        ax.plot([MARGIN_L, FIG_W - MARGIN_R], [y_pos, y_pos],
                color=color, linewidth=lw, alpha=alpha, clip_on=False)

    def section_label(text):
        nonlocal y
        y -= GAP
        ax.text(MARGIN_L, y, text.upper(), fontsize=8, color=MUTED, va="top")
        y -= LABEL_H

    def draw_metric_cards(metrics: dict):
        nonlocal y
        n   = len(metrics)
        gap = 0.12
        w   = (CONTENT_W - gap * (n - 1)) / n
        y_card_bottom = y - CARD_H
        for i, (label, val) in enumerate(metrics.items()):
            cx = MARGIN_L + i * (w + gap)
            rect(cx, y_card_bottom, w, CARD_H, CARD, radius=0.08)
            ax.text(cx + w / 2, y_card_bottom + CARD_H * 0.72, label,
                    fontsize=9, color=MUTED, ha="center", va="center")
            ax.text(cx + w / 2, y_card_bottom + CARD_H * 0.28, val,
                    fontsize=18, color=TEXT, ha="center", va="center", fontweight="bold")
        y = y_card_bottom

    def draw_table_header(col_labels, col_x):
        nonlocal y
        y -= TH_H
        rect(MARGIN_L, y, CONTENT_W, TH_H, CARD, radius=0)
        for label, cx in zip(col_labels, col_x):
            ax.text(cx + 0.08, y + TH_H * 0.5, label,
                    fontsize=8.5, color=MUTED, va="center", fontweight="normal")
        hline(y, lw=0.8)

    # ------------------------------------------------------------------ #
    #  Title                                                               #
    # ------------------------------------------------------------------ #
    y -= TITLE_H
    ax.text(FIG_W / 2, y + TITLE_H * 0.35, "Eval results summary",
            fontsize=16, color=TEXT, ha="center", va="center", fontweight="bold")

    y -= SUBTITLE_H
    ax.text(FIG_W / 2, y + SUBTITLE_H * 0.5,
            f"QA: {num_qa_runs} runs × {num_qa_questions} questions   |   "
            f"Routing: {num_multi_runs} runs × {len(multi_part_questions)} questions",
            fontsize=9.5, color=MUTED, ha="center", va="center")

    # ------------------------------------------------------------------ #
    #  Metric cards                                                        #
    # ------------------------------------------------------------------ #
    section_label("QA accuracy")
    draw_metric_cards(qa_metrics)

    section_label("Routing accuracy")
    draw_metric_cards(rt_metrics)

    # ------------------------------------------------------------------ #
    #  Routing pass rate table                                             #
    # ------------------------------------------------------------------ #
    section_label(f"Per-question routing pass rate  (out of {num_multi_runs} runs)")

    CW_r  = [CONTENT_W * f for f in [0.55, 0.09, 0.08, 0.08, 0.20]]
    col_x_r = [MARGIN_L + sum(CW_r[:i]) for i in range(len(CW_r))]
    draw_table_header(["Question", "Passes", "Rate", "Status", ""], col_x_r)

    for i, (q_key, passes, rate_str, status, color, rate_val) in enumerate(q_rows):
        y -= TR_H
        if i % 2 == 1:
            rect(MARGIN_L, y, CONTENT_W, TR_H, CARD, radius=0)
        row_mid = y + TR_H * 0.5

        ax.text(col_x_r[0] + 0.08, row_mid, q_key,    fontsize=8.5, color=TEXT, va="center")
        ax.text(col_x_r[1] + 0.08, row_mid, passes,   fontsize=8.5, color=TEXT, va="center")
        ax.text(col_x_r[2] + 0.08, row_mid, rate_str, fontsize=8.5, color=TEXT, va="center")

        badge_w, badge_h = 0.55, 0.18
        badge_x = col_x_r[3] + 0.06
        rect(badge_x, row_mid - badge_h / 2, badge_w, badge_h, color, radius=0.04)
        ax.text(badge_x + badge_w / 2, row_mid, status,
                fontsize=7.5, color=BG, ha="center", va="center", fontweight="bold")

        bar_x     = col_x_r[4] + 0.08
        bar_y     = row_mid - 0.055
        bar_track = CW_r[4] - 0.16
        rect(bar_x, bar_y, bar_track, 0.11, CARD, radius=0.02)
        if rate_val > 0:
            rect(bar_x, bar_y, bar_track * rate_val, 0.11, color, radius=0.02)

        hline(y, lw=0.3, alpha=0.5)

    # ------------------------------------------------------------------ #
    #  QA missed questions table                                           #
    # ------------------------------------------------------------------ #
    if qa_miss_rows:
        section_label(f"Most missed QA questions  (out of {num_qa_runs} runs)")

        max_miss = qa_miss_rows[0][1]   # for bar scaling
        CW_q  = [CONTENT_W * f for f in [0.72, 0.08, 0.20]]
        col_x_q = [MARGIN_L + sum(CW_q[:i]) for i in range(len(CW_q))]
        draw_table_header(["Question", "Missed", ""], col_x_q)

        for i, (question, count) in enumerate(qa_miss_rows):
            y -= TR_H
            if i % 2 == 1:
                rect(MARGIN_L, y, CONTENT_W, TR_H, CARD, radius=0)
            row_mid = y + TR_H * 0.5

            # Truncate long question strings to fit column
            max_chars = 90
            q_display = question if len(question) <= max_chars else question[:max_chars - 1] + "…"
            miss_rate = count / num_qa_runs
            color     = RED if miss_rate >= 0.6 else (AMBER if miss_rate >= 0.3 else GRAY)

            ax.text(col_x_q[0] + 0.08, row_mid, q_display, fontsize=7.5, color=TEXT, va="center")
            ax.text(col_x_q[1] + 0.08, row_mid, f"{count}×",
                    fontsize=8.5, color=color, va="center", fontweight="bold")

            bar_x     = col_x_q[2] + 0.08
            bar_y     = row_mid - 0.055
            bar_track = CW_q[2] - 0.16
            fill_w    = bar_track * (count / max_miss)
            rect(bar_x, bar_y, bar_track, 0.11, CARD, radius=0.02)
            rect(bar_x, bar_y, fill_w,    0.11, color, radius=0.02)

            hline(y, lw=0.3, alpha=0.5)

    # ------------------------------------------------------------------ #
    #  Footer                                                              #
    # ------------------------------------------------------------------ #
    hline(y, lw=0.8)
    y -= 0.06
    if missed_routing:
        ax.text(MARGIN_L, y,
                f"Routing never passed: {', '.join(missed_routing)}",
                fontsize=8, color=RED, va="top")

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved → {output_path}")
