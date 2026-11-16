"""Generate publication-friendly SVG/PNG from a measured run. Requires matplotlib."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/retrieval-results"))
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                        "svg.fonttype": "none", "axes.spines.top": False,
                        "axes.spines.right": False, "axes.spines.left": False})
    fig, (scores_ax, delta_ax) = plt.subplots(1, 2, figsize=(12.8, 5.4),
                                             gridspec_kw={"width_ratios": [1, 1.2]})
    fig.set_facecolor("#f7f5ef")
    colors = ["#667380", "#20776c", "#c06a37"]
    names = list(result["systems"])
    scores = [result["systems"][name]["aggregate"]["ndcg@10"] for name in names]
    for ax in (scores_ax, delta_ax):
        ax.set_facecolor("#f7f5ef")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#d9d9d4", linewidth=0.7)
        ax.set_axisbelow(True)
    scores_ax.barh(names, scores, color=colors, height=0.5)
    scores_ax.invert_yaxis()
    scores_ax.set_xlim(0, 1)
    scores_ax.set_xlabel("nDCG@10 · higher is better")
    scores_ax.set_title("Ranking quality", loc="left", pad=18, weight="bold")
    for i, score in enumerate(scores):
        scores_ax.text(score + .016, i, f"{score:.4f}", va="center", weight="bold")
    comparisons = result["comparisons"]
    for i, comp in enumerate(comparisons):
        mean = comp["mean_delta"]
        lo, hi = comp["ci95"]
        delta_ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o",
                          capsize=5, color="#20776c", markersize=7, linewidth=2)
    delta_ax.axvline(0, color="#424950", linewidth=1, linestyle="--")
    delta_ax.set_yticks(range(len(comparisons)),
                        [f"{c['candidate']} − {c['baseline']}" for c in comparisons])
    delta_ax.set_ylim(-.6, len(comparisons) - .4)
    delta_ax.invert_yaxis()
    delta_ax.set_xlabel("Δ nDCG@10 · paired 95% bootstrap interval")
    delta_ax.set_title("Does the difference persist across queries?", loc="left", pad=18,
                       weight="bold", fontsize=11)
    fig.suptitle("EVIDENCEFORGE  /  Measured retrieval, inspectable tradeoffs", x=.055,
                 ha="left", fontsize=17, weight="bold", y=.98)
    fig.text(.055, .87, "BEIR SciFact  •  5,183 documents  •  300 queries  •  local CPU inference",
             color="#555d60", fontsize=11)
    fig.text(.055, .055, "One prespecified experiment. Public benchmark; original SciFact dev split.\n"
             "Retrieval relevance is measured here. Generated-answer quality is not.",
             color="#555d60", fontsize=10)
    fig.subplots_adjust(left=.08, right=.975, top=.73, bottom=.23, wspace=.65)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".svg"), facecolor=fig.get_facecolor())
    svg = args.output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()).rstrip() + "\n",
                   encoding="utf-8", newline="\n")
    fig.savefig(args.output.with_suffix(".png"), dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
