"""The one command: python main.py

Downloads the stimuli, runs the four networks, and writes every number and
figure into results/.
"""

import csv
from pathlib import Path

import numpy as np

import categories
import data
import metrics
import models
import paper
import plots

RESULTS = Path(__file__).resolve().parent / "results"

# The aggregation that reproduces the paper's released decisions: with "max",
# torchvision ResNet-50 (the same weights as the paper's) gives the paper's
# decision on all 1,280 images. The authors' published function defaults to
# "mean", which does not (see the README).
AGGREGATION = "max"


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def fmt(x):
    return f"{x:.4f}"


def main():
    data.download()
    stimuli = data.list_stimuli()
    paths = [p for p, _, _ in stimuli]
    names = [p.name for p in paths]
    shapes = [s for _, s, _ in stimuli]
    textures = [t for _, _, t in stimuli]
    conflict = metrics.conflict_mask(shapes, textures)
    RESULTS.mkdir(exist_ok=True)

    # The released human decisions, pooled over the ten observers as the
    # paper's 95.9% is (pooling gives 0.9587; the mean of per-observer values
    # would give 0.9576).
    humans = [paper.read_decisions(f) for f in paper.human_files()]
    human_summary = metrics.summary(shapes * len(humans), textures * len(humans),
                                    [h[n] for h in humans for n in names])

    # Each network is run once on all 1,280 images; the outputs are kept for
    # the mean-against-max table below.
    softmax = {}
    ours, theirs, summary_rows = {}, {}, []
    for name in models.MODELS:
        print(f"Running {name} on {len(paths)} images (CPU)")
        probs = models.softmax_outputs(models.load(name), paths)
        softmax[name] = probs
        scores = categories.aggregate(probs, AGGREGATION)
        decisions = categories.decide(probs, AGGREGATION)
        ours[name] = metrics.summary(shapes, textures, decisions)
        weighted = metrics.weighted_shape_bias(shapes, textures, scores)

        released = paper.read_decisions(paper.PAPER_MODEL_FILES[name])
        released = [released[n] for n in names]
        theirs[name] = metrics.summary(shapes, textures, released)
        # Image-by-image agreement with the paper's network, conflict images only.
        agree = np.mean([a == b for a, b, c in zip(decisions, released, conflict) if c])

        write_csv(RESULTS / f"decisions_{name}.csv",
                  ["image", "shape", "texture", "decision", "paper_decision"] + categories.CATEGORIES,
                  [[n, s, t, d, r] + [fmt(v) for v in row]
                   for n, s, t, d, r, row in zip(names, shapes, textures, decisions, released, scores)])

        summary_rows.append([
            name, paper.PAPER_SHAPE_BIAS[name], paper.README_TORCHVISION_SHAPE_BIAS.get(name, ""),
            fmt(theirs[name]["shape_bias"]), fmt(ours[name]["shape_bias"]),
            fmt(ours[name]["shape_fraction"]), fmt(ours[name]["texture_fraction"]),
            fmt(ours[name]["neither_fraction"]), fmt(theirs[name]["neither_fraction"]),
            fmt(weighted), fmt(agree)])

    summary_rows.append([
        "humans", paper.PAPER_SHAPE_BIAS["humans"], "", fmt(human_summary["shape_bias"]), "",
        "", "", "", fmt(human_summary["neither_fraction"]), "", ""])

    header = ["model", "paper_figure4", "authors_readme_torchvision", "recomputed_from_released",
              "ours", "ours_shape_frac", "ours_texture_frac", "ours_neither_frac",
              "released_neither_frac", "ours_prob_weighted", "agreement_with_released"]
    write_csv(RESULTS / "summary.csv", header, summary_rows)

    write_csv(RESULTS / "per_category_shape_bias.csv",
              ["category", "humans_released"] + [f"{m}_ours" for m in models.MODELS]
              + [f"{m}_released" for m in models.MODELS],
              [[c, fmt(human_summary["per_category_shape_bias"][c])]
               + [fmt(ours[m]["per_category_shape_bias"][c]) for m in models.MODELS]
               + [fmt(theirs[m]["per_category_shape_bias"][c]) for m in models.MODELS]
               for c in categories.CATEGORIES])

    # Shape bias and agreement per network under each variant, for the agreement figure.
    variants = {"resize_crop": {}}

    # Ablation: the same networks with resize-256 and centre-crop-224 first.
    ablation_rows = []
    for name in models.MODELS:
        print(f"Running {name} with resize and crop")
        probs = models.softmax_outputs(models.load(name), paths, models.PREPROCESS_RESIZE_CROP)
        decisions = categories.decide(probs, AGGREGATION)
        released = paper.read_decisions(paper.PAPER_MODEL_FILES[name])
        agree = np.mean([d == released[n] for d, n, c in zip(decisions, names, conflict) if c])
        bias = metrics.summary(shapes, textures, decisions)["shape_bias"]
        variants["resize_crop"][name] = (bias, agree)
        ablation_rows.append([name, fmt(ours[name]["shape_bias"]), fmt(bias), fmt(agree)])
    write_csv(RESULTS / "ablation_resize_crop.csv",
              ["model", "ours_no_resize", "ours_resize256_crop224", "agreement_with_released"],
              ablation_rows)

    # Mean against max: the published function's default against the rule that
    # reproduces the released decisions.
    aggregation_rows = []
    for name in models.MODELS:
        probs = softmax[name]
        released = paper.read_decisions(paper.PAPER_MODEL_FILES[name])
        row = [name, paper.PAPER_SHAPE_BIAS[name]]
        for how in categories.AGGREGATIONS:
            decisions = categories.decide(probs, how)
            agree = np.mean([d == released[n] for d, n, c in zip(decisions, names, conflict) if c])
            bias = metrics.summary(shapes, textures, decisions)["shape_bias"]
            variants.setdefault(how, {})[name] = (bias, agree)
            row += [fmt(bias), fmt(agree)]
        aggregation_rows.append(row)
    write_csv(RESULTS / "aggregation_mean_vs_max.csv",
              ["model", "paper_figure4"]
              + [f"{how}_{col}" for how in categories.AGGREGATIONS
                 for col in ("shape_bias", "agreement_with_released")],
              aggregation_rows)

    plots.figure4(ours, theirs, human_summary, RESULTS / "figure4_reproduction.png")
    plots.agreement_figure(variants, RESULTS / "figure_agreement.png")

    widths = [max(len(str(r[i])) for r in [header] + summary_rows) for i in range(len(header))]
    for r in [header] + summary_rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    print(f"\nWritten to {RESULTS}")


if __name__ == "__main__":
    main()
