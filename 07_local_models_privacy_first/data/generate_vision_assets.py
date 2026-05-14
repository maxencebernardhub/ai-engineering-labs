"""Generate synthetic vision assets for notebook 04.

Run with:  uv run python data/generate_vision_assets.py
Outputs:   data/vision/invoice_sample.jpg
           data/vision/org_chart.png
           data/vision/dashboard_screenshot.png

Ground-truth values embedded in the images:
  invoice_sample.jpg      vendor: Acme Cloud Ltd, total: £4,832.00, VAT: £805.33
  org_chart.png           CTO direct reports: Alice Chen, Bob Patel, Carol Osei, Dan Müller
  dashboard_screenshot.png  attrition KPI: 4.2%
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = Path(__file__).parent / "vision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Vendor invoice
# ---------------------------------------------------------------------------


def generate_invoice() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Header bar
    ax.add_patch(
        FancyBboxPatch(
            (0.3, 9.8),
            7.9,
            0.9,
            boxstyle="round,pad=0.05",
            facecolor="#1a3a5c",
            edgecolor="none",
        )
    )
    ax.text(
        4.25,
        10.25,
        "INVOICE",
        fontsize=26,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    # Vendor block
    ax.text(0.5, 9.55, "From:", fontsize=8, color="#666666")
    ax.text(
        0.5, 9.30, "Acme Cloud Ltd", fontsize=11, fontweight="bold", color="#1a3a5c"
    )
    ax.text(0.5, 9.10, "14 Fintech Square, London EC2V 8RF", fontsize=8)
    ax.text(0.5, 8.92, "VAT Reg: GB 198 4321 77", fontsize=8, color="#666666")

    # Invoice meta
    ax.text(5.5, 9.55, "Invoice No:", fontsize=8, color="#666666")
    ax.text(7.8, 9.55, "INV-2024-0847", fontsize=8, ha="right")
    ax.text(5.5, 9.30, "Issue date:", fontsize=8, color="#666666")
    ax.text(7.8, 9.30, "28 November 2024", fontsize=8, ha="right")
    ax.text(5.5, 9.10, "Due date:", fontsize=8, color="#666666")
    ax.text(7.8, 9.10, "28 December 2024", fontsize=8, ha="right")

    # Bill to block
    ax.text(0.5, 8.60, "Bill To:", fontsize=8, color="#666666")
    ax.text(0.5, 8.40, "Contoso Corp", fontsize=10, fontweight="bold")
    ax.text(0.5, 8.22, "1 Canary Wharf Tower, London E14 5AB", fontsize=8)
    ax.text(
        0.5, 8.04, "Accounts payable: ap@contoso.co.uk", fontsize=8, color="#666666"
    )

    # Line items table header
    y_header = 7.60
    ax.add_patch(
        FancyBboxPatch(
            (0.3, y_header - 0.02),
            7.9,
            0.30,
            boxstyle="round,pad=0.02",
            facecolor="#e8eff7",
            edgecolor="none",
        )
    )
    for x, label in [
        (0.45, "Description"),
        (4.8, "Qty"),
        (5.8, "Unit Price"),
        (7.1, "Amount"),
    ]:
        ax.text(
            x, y_header + 0.10, label, fontsize=9, fontweight="bold", color="#1a3a5c"
        )

    # Line items
    items = [
        ("Cloud Compute — Standard tier (Nov 2024)", "1", "£2,400.00", "£2,400.00"),
        ("Managed Database — PostgreSQL (Nov 2024)", "1", "£980.00", "£980.00"),
        ("Object Storage — 4.8 TB (Nov 2024)", "1", "£312.00", "£312.00"),
        ("Support — Business tier (Nov 2024)", "1", "£335.00", "£335.00"),
    ]
    y = y_header - 0.30
    for i, (desc, qty, unit, amount) in enumerate(items):
        bg = "#f9fbfd" if i % 2 == 0 else "white"
        ax.add_patch(
            FancyBboxPatch(
                (0.3, y - 0.05),
                7.9,
                0.28,
                boxstyle="round,pad=0.01",
                facecolor=bg,
                edgecolor="none",
            )
        )
        ax.text(0.45, y + 0.07, desc, fontsize=8)
        ax.text(4.95, y + 0.07, qty, fontsize=8, ha="center")
        ax.text(6.45, y + 0.07, unit, fontsize=8, ha="right")
        ax.text(7.85, y + 0.07, amount, fontsize=8, ha="right")
        y -= 0.30

    # Totals section
    y_totals = y - 0.20
    ax.plot(
        [4.5, 8.1], [y_totals + 0.35, y_totals + 0.35], color="#cccccc", linewidth=0.8
    )

    for label, value, bold in [
        ("Subtotal (excl. VAT):", "£4,027.00", False),
        ("VAT (20%):", "£805.33", False),
        ("TOTAL DUE:", "£4,832.00", True),
    ]:
        ax.text(
            5.5,
            y_totals,
            label,
            fontsize=9 if bold else 8,
            fontweight="bold" if bold else "normal",
            color="#1a3a5c" if bold else "#333333",
        )
        ax.text(
            7.85,
            y_totals,
            value,
            fontsize=9 if bold else 8,
            fontweight="bold" if bold else "normal",
            ha="right",
            color="#1a3a5c" if bold else "#333333",
        )
        y_totals -= 0.30

    # Payment info
    ax.add_patch(
        FancyBboxPatch(
            (0.3, 2.6),
            7.9,
            1.20,
            boxstyle="round,pad=0.05",
            facecolor="#f0f4fa",
            edgecolor="none",
        )
    )
    ax.text(
        0.55,
        3.65,
        "Payment Instructions",
        fontsize=9,
        fontweight="bold",
        color="#1a3a5c",
    )
    for i, line in enumerate(
        [
            "Bank: Barclays Business  |  Sort code: 20-45-67  |  Account: 83920145",
            "Reference: INV-2024-0847",
            "Please pay within 30 days of invoice date. Late payments subject to 2% monthly fee.",
        ]
    ):
        ax.text(0.55, 3.40 - i * 0.22, line, fontsize=7.5, color="#555555")

    # Footer
    ax.text(
        4.25,
        2.25,
        "Thank you for your business — Acme Cloud Ltd",
        fontsize=8,
        ha="center",
        color="#888888",
        style="italic",
    )

    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0)
    fig.savefig(
        OUTPUT_DIR / "invoice_sample.jpg",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("invoice_sample.jpg generated")


# ---------------------------------------------------------------------------
# 2. Org chart — Engineering department
# ---------------------------------------------------------------------------


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    name: str,
    title: str,
    color: str = "#1a3a5c",
    text_color: str = "white",
    width: float = 1.6,
    height: float = 0.55,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#0d2240",
            linewidth=1.2,
        )
    )
    ax.text(
        x,
        y + 0.08,
        name,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        color=text_color,
    )
    ax.text(
        x,
        y - 0.12,
        title,
        fontsize=6.5,
        ha="center",
        va="center",
        color=text_color if text_color == "white" else "#555555",
    )


def _connect(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    mid_y = (y1 + y2) / 2
    ax.plot([x1, x1], [y1, mid_y], color="#0d2240", linewidth=1)
    ax.plot([x1, x2], [mid_y, mid_y], color="#0d2240", linewidth=1)
    ax.plot([x2, x2], [mid_y, y2], color="#0d2240", linewidth=1)


def generate_org_chart() -> None:
    # 17-unit wide canvas so 8 L2 boxes (width=1.5, gap=0.5) fit without overlap.
    fig, ax = plt.subplots(figsize=(17, 8))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 8)
    ax.axis("off")

    cx = 8.5  # horizontal centre of the figure

    # Title
    ax.text(
        cx,
        7.65,
        "Contoso Corp — Engineering Department",
        fontsize=14,
        fontweight="bold",
        ha="center",
        color="#1a3a5c",
    )
    ax.text(
        cx,
        7.35,
        "Organisation Chart · December 2024",
        fontsize=9,
        ha="center",
        color="#666666",
    )

    # Level 0 — CTO
    cto_x, cto_y = cx, 6.80
    _draw_box(
        ax,
        cto_x,
        cto_y,
        "Sarah Whitmore",
        "CTO",
        color="#0d2240",
        width=2.0,
        height=0.60,
    )

    # Level 1 — 4 direct reports, centred above their two L2 children each.
    # L2 are at x = 1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5 (step 2.0).
    # L1 midpoints: (1.5+3.5)/2=2.5, (5.5+7.5)/2=6.5, (9.5+11.5)/2=10.5, (13.5+15.5)/2=14.5
    l1 = [
        (2.5, 5.5, "Alice Chen", "VP Engineering – Platform"),
        (6.5, 5.5, "Bob Patel", "VP Engineering – Product"),
        (10.5, 5.5, "Carol Osei", "Head of ML & Data"),
        (14.5, 5.5, "Dan Müller", "Head of DevOps & SRE"),
    ]
    for x, y, name, title in l1:
        _connect(ax, cto_x, cto_y - 0.30, x, y + 0.28)
        _draw_box(ax, x, y, name, title, color="#1a3a5c", width=2.2, height=0.56)

    # Level 2 — 2 reports per L1, evenly spaced at step 2.0 across the full width.
    l2 = [
        # Under Alice Chen (2.5)
        (1.5, 4.1, "Priya Sharma", "Staff Eng – Infra"),
        (3.5, 4.1, "James Liu", "Lead Eng – API"),
        # Under Bob Patel (6.5)
        (5.5, 4.1, "Fatima Idrissi", "Lead Eng – Frontend"),
        (7.5, 4.1, "Luca Romano", "Senior Eng – iOS"),
        # Under Carol Osei (10.5)
        (9.5, 4.1, "Noah Bernstein", "ML Engineer"),
        (11.5, 4.1, "Yuki Tanaka", "Data Engineer"),
        # Under Dan Müller (14.5)
        (13.5, 4.1, "Anya Kowalski", "SRE Lead"),
        (15.5, 4.1, "Tariq Hassan", "DevOps Engineer"),
    ]
    l1_parents = [2.5, 2.5, 6.5, 6.5, 10.5, 10.5, 14.5, 14.5]
    for i, (x, y, name, title) in enumerate(l2):
        _connect(ax, l1_parents[i], 5.5 - 0.28, x, y + 0.28)
        _draw_box(ax, x, y, name, title, color="#2e6da4", width=1.7, height=0.56)

    # Legend
    handles = [
        mpatches.Patch(color="#0d2240", label="C-Suite / VP"),
        mpatches.Patch(color="#1a3a5c", label="Head of / VP"),
        mpatches.Patch(color="#2e6da4", label="Lead / Staff / Senior"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)

    # Footer
    ax.text(
        cx,
        0.25,
        "Confidential — Contoso Corp People & HR · Q4 2024",
        fontsize=7,
        ha="center",
        color="#aaaaaa",
        style="italic",
    )

    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0.3)
    fig.savefig(
        OUTPUT_DIR / "org_chart.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print("org_chart.png generated")


# ---------------------------------------------------------------------------
# 3. HR dashboard screenshot
# ---------------------------------------------------------------------------


def generate_dashboard() -> None:
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#f0f2f5")

    # ── Top header bar ────────────────────────────────────────────────────
    header = fig.add_axes([0, 0.88, 1, 0.12])
    header.set_facecolor("#1a3a5c")
    header.axis("off")
    header.text(
        0.03,
        0.55,
        "Contoso Corp",
        fontsize=18,
        fontweight="bold",
        color="white",
        transform=header.transAxes,
    )
    header.text(
        0.03,
        0.15,
        "People & HR Dashboard — Q4 2024",
        fontsize=10,
        color="#aac8e8",
        transform=header.transAxes,
    )
    header.text(
        0.97,
        0.55,
        "Last updated: 31 Dec 2024",
        fontsize=8,
        color="#aac8e8",
        ha="right",
        transform=header.transAxes,
    )

    # ── KPI cards (top row) ───────────────────────────────────────────────
    kpis = [
        ("Total Headcount", "214", "+12 vs Q3", "#27ae60"),
        ("Q4 Attrition Rate", "4.2%", "↓ 1.1pp vs Q3", "#27ae60"),
        ("Avg Performance", "3.11", "Stable", "#2e86c1"),
        ("Open Roles", "24", "Q1 2025 target", "#e67e22"),
        ("Employees on PIP", "5", "Requires action", "#e74c3c"),
    ]
    card_w, card_gap = 0.175, 0.01
    for i, (label, value, note, note_color) in enumerate(kpis):
        x = 0.02 + i * (card_w + card_gap)
        card = fig.add_axes([x, 0.64, card_w, 0.22])
        card.set_facecolor("white")
        card.set_xlim(0, 1)
        card.set_ylim(0, 1)
        card.axis("off")
        for spine in card.spines.values():
            spine.set_visible(False)
        card.text(
            0.5,
            0.78,
            label,
            fontsize=8,
            ha="center",
            color="#666666",
            transform=card.transAxes,
        )
        card.text(
            0.5,
            0.46,
            value,
            fontsize=22,
            fontweight="bold",
            ha="center",
            color="#1a3a5c",
            transform=card.transAxes,
        )
        card.text(
            0.5,
            0.14,
            note,
            fontsize=7.5,
            ha="center",
            color=note_color,
            transform=card.transAxes,
        )

    # ── Headcount by department (bar chart) ───────────────────────────────
    ax_bar = fig.add_axes([0.04, 0.10, 0.38, 0.48])
    ax_bar.set_facecolor("white")
    departments = [
        "Engineering",
        "Sales",
        "Cust. Success",
        "Product",
        "Marketing",
        "Finance & Ops",
        "People & HR",
        "Executive",
    ]
    counts = [72, 38, 31, 22, 19, 17, 9, 6]
    bars = ax_bar.barh(departments, counts, color="#2e6da4", height=0.6)
    for bar, count in zip(bars, counts):
        ax_bar.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=8,
            color="#333333",
        )
    ax_bar.set_xlabel("Headcount", fontsize=8)
    ax_bar.set_title(
        "Headcount by Department",
        fontsize=10,
        fontweight="bold",
        color="#1a3a5c",
        pad=8,
    )
    ax_bar.tick_params(labelsize=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ── Performance rating distribution (pie) ─────────────────────────────
    ax_pie = fig.add_axes([0.48, 0.10, 0.24, 0.48])
    ax_pie.set_facecolor("white")
    sizes = [12, 41, 108, 28, 7]
    labels = [
        "Exceptional\n(5)",
        "Exceeds\n(4)",
        "Meets\n(3)",
        "Partially\nMeets (2)",
        "Below\n(1)",
    ]
    colors = ["#1a9641", "#74c476", "#fed976", "#fd8d3c", "#d7191c"]
    ax_pie.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        pctdistance=0.75,
        startangle=90,
        textprops={"fontsize": 7.5},
    )
    ax_pie.set_title(
        "Performance Distribution\n(196 reviewed)",
        fontsize=10,
        fontweight="bold",
        color="#1a3a5c",
        pad=8,
    )

    # ── Attrition trend (line) ────────────────────────────────────────────
    ax_trend = fig.add_axes([0.76, 0.10, 0.22, 0.48])
    ax_trend.set_facecolor("white")
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    attrition = [3.1, 4.0, 5.3, 4.2]
    ax_trend.plot(
        quarters,
        attrition,
        marker="o",
        linewidth=2,
        color="#2e6da4",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2,
    )
    for q, v in zip(quarters, attrition):
        ax_trend.text(
            q,
            v + 0.15,
            f"{v}%",
            ha="center",
            fontsize=8,
            color="#1a3a5c",
            fontweight="bold",
        )
    ax_trend.axhline(
        y=4.2, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.6, label="Q4: 4.2%"
    )
    ax_trend.set_ylim(0, 7)
    ax_trend.set_title(
        "Attrition Rate by Quarter\n(2024)",
        fontsize=10,
        fontweight="bold",
        color="#1a3a5c",
        pad=8,
    )
    ax_trend.set_ylabel("Attrition %", fontsize=8)
    ax_trend.tick_params(labelsize=8)
    ax_trend.spines["top"].set_visible(False)
    ax_trend.spines["right"].set_visible(False)
    ax_trend.legend(fontsize=7, loc="upper left")

    fig.savefig(
        OUTPUT_DIR / "dashboard_screenshot.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print("dashboard_screenshot.png generated")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_invoice()
    generate_org_chart()
    generate_dashboard()
    print(f"\nAll assets written to {OUTPUT_DIR.resolve()}")
