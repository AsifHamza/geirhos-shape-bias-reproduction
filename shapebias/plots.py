"""Figure 4 of the paper, redrawn with our numbers next to the released ones."""

import matplotlib

matplotlib.use("Agg")  # write files only; no window, so it runs headless
import matplotlib.pyplot as plt

TITLES = {"alexnet": "AlexNet", "vgg16": "VGG-16", "googlenet": "GoogLeNet", "resnet50": "ResNet-50"}


def figure4(ours, theirs, humans, path):
    """One panel per network. As in the paper, each row is a shape category,
    sorted by human shape bias, and the x axis is the fraction of texture
    decisions among shape-or-texture decisions (1 minus shape bias), with
    the overall value as a vertical line."""
    cats = sorted(humans["per_category_shape_bias"],
                  key=lambda c: humans["per_category_shape_bias"][c])
    y = range(len(cats))
    fig, axes = plt.subplots(1, len(ours), figsize=(3.2 * len(ours), 5.5), sharey=True)
    for ax, name in zip(axes, ours):
        for summary, style, label in [
            (humans, dict(color="#a51c30", marker="o", s=40), "humans (released)"),
            (theirs[name], dict(facecolors="none", edgecolors="black", marker="s", s=45),
             "paper network (released)"),
            (ours[name], dict(color="#1f77b4", marker="D", s=30), "ours (torchvision)"),
        ]:
            x = [1 - summary["per_category_shape_bias"][c] for c in cats]
            ax.scatter(x, y, label=label, zorder=3, **style)
            colour = style.get("color", "black")
            ax.axvline(1 - summary["shape_bias"], color=colour, alpha=0.5, lw=1.5)
        ax.set_title(f"{TITLES[name]}\nours {ours[name]['shape_bias']:.1%}, "
                     f"paper {theirs[name]['shape_bias']:.1%}", fontsize=10)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("fraction of texture decisions")
        ax.grid(axis="y", ls=":", alpha=0.5)
    axes[0].set_yticks(list(y), cats)
    axes[0].set_ylabel("shape category")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))  # leave room for the legend
    fig.savefig(path, dpi=150)
    plt.close(fig)
