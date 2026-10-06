"""Console summary and plots for :class:`~fefet_reservoir.tracks.rc.RCScores`."""

from __future__ import annotations

from pathlib import Path

from .rc import RCScores

__all__ = ["format_rc_summary", "save_rc_plots"]


def format_rc_summary(scores: RCScores) -> str:
    """Render *scores* as human-readable console text.

    Always includes the two headline numbers, the protocol, the full
    settings echo (every value that determined the score -- see
    :class:`~fefet_reservoir.tracks.rc.RCScores`), and, for ``kfold`` runs,
    the per-fold capacities.
    """
    tasks = scores.settings.get("tasks", ("stm", "pc"))

    lines = ["STM / PC scores"]
    if "stm" in tasks:
        lines.append(f"  STM capacity      : {scores.stm_capacity:.4f}")
    else:
        lines.append("  STM capacity      : not scored")
    if "pc" in tasks:
        lines.append(f"  PC capacity       : {scores.pc_capacity:.4f}")
    else:
        lines.append("  PC capacity       : not scored")
    bits_used = scores.settings.get("readout_bits_used")
    if bits_used == "ideal":
        lines.append("  Readout           : ideal (no quantization)")
    elif bits_used is not None:
        full_scale = scores.settings.get("readout_full_scale")
        lines.append(
            f"  Readout           : {bits_used}-bit, range 0..{full_scale:.3g} A "
            f"(readout_bits: {scores.settings.get('readout_bits')})"
        )
    lines.append(f"Protocol: {scores.protocol}")
    lines.append("Settings:")
    for key, value in scores.settings.items():
        lines.append(f"  {key}: {value}")

    if scores.protocol == "kfold" and scores.per_fold:
        lines.append("Per-fold scores:")
        for task, values in scores.per_fold.items():
            formatted = ", ".join(f"{v:.4f}" for v in values)
            lines.append(f"  {task}: [{formatted}]")

    return "\n".join(lines)


def save_rc_plots(scores: RCScores, out_dir) -> list[Path]:
    """Save the STM / PC summary plots for *scores* into *out_dir*.

    Writes two PNGs and returns their paths:

    1. ``rc_delay_curves.png`` -- STM and PC delay-vs-r^2 curves on one
       figure, with a legend.
    2. ``rc_summary_bar.png`` -- a bar chart of the two headline numbers
       (STM capacity, PC capacity).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig1 = Figure()
    FigureCanvasAgg(fig1)
    ax1 = fig1.subplots()
    ax1.plot(range(len(scores.stm_curve)), scores.stm_curve, marker="o", label="STM")
    ax1.plot(range(len(scores.pc_curve)), scores.pc_curve, marker="s", label="PC")
    ax1.set_xlabel("delay")
    ax1.set_ylabel("r^2")
    ax1.set_title("STM / PC: r^2 vs delay")
    ax1.legend()
    curves_path = out_dir / "rc_delay_curves.png"
    fig1.savefig(curves_path)
    paths.append(curves_path)

    fig2 = Figure()
    FigureCanvasAgg(fig2)
    ax2 = fig2.subplots()
    labels = ["STM capacity", "PC capacity"]
    values = [scores.stm_capacity, scores.pc_capacity]
    ax2.bar(labels, values)
    ax2.set_title("STM / PC capacity")
    bar_path = out_dir / "rc_summary_bar.png"
    fig2.savefig(bar_path)
    paths.append(bar_path)

    return paths
