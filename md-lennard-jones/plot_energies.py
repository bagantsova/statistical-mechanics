#!/usr/bin/env python3
"""Plot the energy traces written by md_lennard_jones.

    python3 plot_energies.py results/energies.csv --output ../docs/figures/energy.png
"""

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        [int(r["step"]) for r in rows],
        [float(r["kinetic"]) for r in rows],
        [float(r["potential"]) for r in rows],
        [float(r["total"]) for r in rows],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("energy.png"))
    args = parser.parse_args()

    step, kinetic, potential, total = read(args.csv)

    figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(step, kinetic, label="kinetic", linewidth=1)
    axes[0].plot(step, potential, label="potential", linewidth=1)
    axes[0].plot(step, total, label="total", linewidth=1.6, color="black")
    axes[0].set_ylabel("energy (reduced units)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)

    # The conserved quantity, plotted against its own scale: this is where an
    # integrator bug shows up as a visible slope.
    reference = total[0]
    drift = [(value - reference) / abs(reference) for value in total]
    axes[1].plot(step, drift, linewidth=1, color="black")
    axes[1].set_ylabel("relative drift\nin total energy")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.3)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
