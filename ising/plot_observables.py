#!/usr/bin/env python3
"""Plot the finite-size scan written by ising_2d.

    python3 plot_observables.py results/observables.csv \
            --output ../docs/figures/ising_finite_size_scaling.png
"""

import argparse
import collections
import csv
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXACT_TC = 2.0 / math.log(1.0 + math.sqrt(2.0))


def read(path):
    by_size = collections.defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_size[int(row["L"])].append(
                {key: float(row[key]) for key in row if key != "L"}
            )
    for rows in by_size.values():
        rows.sort(key=lambda r: r["T"])
    return dict(sorted(by_size.items()))


def series(rows, key):
    return [r["T"] for r in rows], [r[key] for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("ising_finite_size_scaling.png"))
    args = parser.parse_args()

    by_size = read(args.csv)

    panels = [
        ("abs_magnetisation", r"$\langle |m| \rangle$", "magnetisation per site"),
        ("specific_heat", r"$C$", "specific heat"),
        ("susceptibility", r"$\chi$", "susceptibility"),
        ("binder", r"$U_4$", "Binder cumulant"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    for axis, (key, symbol, title) in zip(axes.flat, panels):
        for side, rows in by_size.items():
            temperature, values = series(rows, key)
            axis.plot(temperature, values, marker="o", markersize=2.5,
                      linewidth=1, label=f"L = {side}")

        # The exact critical temperature is the reference every panel is read
        # against: the order parameter collapses here, C and chi peak here,
        # and the Binder curves cross here.
        axis.axvline(EXACT_TC, color="black", linestyle="--", linewidth=1,
                     label=r"exact $T_c$" if key == "abs_magnetisation" else None)
        axis.set_xlabel("temperature")
        axis.set_ylabel(symbol)
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3)

    axes.flat[0].legend(frameon=False, fontsize=8)

    # Zoom the Binder panel on the crossing region, which is the whole point
    # of the plot and is invisible at full scale.
    axes.flat[3].set_xlim(EXACT_TC - 0.12, EXACT_TC + 0.12)

    figure.suptitle(
        "2D Ising model, Wolff cluster updates: finite-size scaling near "
        rf"$T_c = 2/\ln(1+\sqrt{{2}}) = {EXACT_TC:.4f}$",
        fontsize=11)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
