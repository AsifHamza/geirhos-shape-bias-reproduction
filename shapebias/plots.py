"""Figure 4 of the paper, redrawn with our numbers next to the released ones."""

import matplotlib

matplotlib.use("Agg")  # write files only; no window, so it runs headless
import matplotlib.pyplot as plt

# Drawn at the size it is printed: the full text width of an IEEE two-column
# page (7.16 in), with 8 pt text, so LaTeX does not have to shrink it.
WIDTH_IN = 7.16
FONT_PT = 8
DPI = 300

TITLES = {"alexnet": "AlexNet", "vgg16": "VGG-16", "googlenet": "GoogLeNet", "resnet50": "ResNet-50"}


def figure4(ours, theirs, humans, path):
    """One panel per network. As in the paper, each row is a shape category,
    sorted by human shape bias, and the x axis is the fraction of texture
    decisions among shape-or-texture decisions (1 minus shape bias), with
    the overall value as a vertical line."""
    cats = sorted(humans["per_category_shape_bias"],
                  key=lambda c: humans["per_category_shape_bias"][c])
    y = range(len(cats))
    with plt.rc_context({"font.size": FONT_PT}):
        _draw(ours, theirs, humans, cats, y, path)


def _draw(ours, theirs, humans, cats, y, path):
    fig, axes = plt.subplots(1, len(ours), figsize=(WIDTH_IN, 4.2), sharey=True)
    for ax, name in zip(axes, ours):
        for summary, style, label in [
            (humans, dict(color="#a51c30", marker="o", s=14), "humans (released)"),
            (theirs[name], dict(facecolors="none", edgecolors="black", marker="s", s=16, linewidths=0.8),
             "paper network (released)"),
            (ours[name], dict(color="#1f77b4", marker="D", s=11), "ours (torchvision)"),
        ]:
            x = [1 - summary["per_category_shape_bias"][c] for c in cats]
            ax.scatter(x, y, label=label, zorder=3, **style)
            colour = style.get("color", "black")
            ax.axvline(1 - summary["shape_bias"], color=colour, alpha=0.5, lw=1)
        ax.set_title(f"{TITLES[name]}\nours {ours[name]['shape_bias']:.1%}\n"
                     f"paper {theirs[name]['shape_bias']:.1%}")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="y", ls=":", alpha=0.5)
    axes[0].set_yticks(list(y), cats)
    axes[0].set_ylabel("shape category")
    fig.supxlabel("fraction of texture decisions (1 minus shape bias)", y=0.06)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)  # leave room for the legend
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
