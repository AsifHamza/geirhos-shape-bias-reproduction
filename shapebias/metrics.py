"""Shape bias and the related numbers, computed from 16-class decisions.

Definitions follow the authors' R code (data-analysis/data-analysis-helper.R,
get.fractions.of.responses, and the "PAPER NUMBERS" block of data-analysis.R):

1. Drop the 80 images whose shape and texture are the same category; they
   cannot tell a shape decision from a texture decision.
2. Within each shape category, take the fraction of decisions that name the
   shape, the texture, or neither.
3. Average each fraction over the 16 categories, then
   shape bias = shape / (shape + texture).

Every category has 75 conflict images, so step 3 gives the same value as
pooling all 1,200 images; the category average is kept to match the authors.
"""

import numpy as np

from .categories import CATEGORIES


def conflict_mask(shapes, textures):
    return np.array([s != t for s, t in zip(shapes, textures)])


def fractions_by_category(shapes, textures, decisions):
    """{category: (shape fraction, texture fraction, neither fraction)},
    over the conflict images whose shape is that category."""
    result = {}
    for c in CATEGORIES:
        rows = [(t, d) for s, t, d in zip(shapes, textures, decisions) if s == c and s != t]
        n = len(rows)
        shape = sum(d == c for t, d in rows) / n
        texture = sum(d == t for t, d in rows) / n
        result[c] = (shape, texture, 1.0 - shape - texture)
    return result


def summary(shapes, textures, decisions):
    """Shape bias, mean fractions and per-category shape bias for one model."""
    by_cat = fractions_by_category(shapes, textures, decisions)
    shape = np.mean([v[0] for v in by_cat.values()])
    texture = np.mean([v[1] for v in by_cat.values()])
    per_category = {c: (v[0] / (v[0] + v[1]) if v[0] + v[1] > 0 else float("nan"))
                    for c, v in by_cat.items()}
    return {
        "shape_bias": shape / (shape + texture),
        "shape_fraction": shape,
        "texture_fraction": texture,
        "neither_fraction": 1.0 - shape - texture,
        "per_category_shape_bias": per_category,
    }


def weighted_shape_bias(shapes, textures, scores):
    """Probability-weighted shape bias, a number the paper does not report.

    scores: (n_images, 16) aggregated probabilities, columns in CATEGORIES
    order (categories.aggregate, with the same aggregation as the decisions).
    Instead of counting only the winning category, each conflict image
    contributes the score of its shape category and of its texture category:

        sum over images of p_shape / sum over images of (p_shape + p_texture)

    It uses the same category scores as the decision rule, so it measures the
    same thing without the all-or-nothing argmax.
    """
    col = {c: i for i, c in enumerate(CATEGORIES)}
    mask = conflict_mask(shapes, textures)
    rows = np.flatnonzero(mask)
    p_shape = scores[rows, [col[shapes[i]] for i in rows]]
    p_texture = scores[rows, [col[textures[i]] for i in rows]]
    return p_shape.sum() / (p_shape.sum() + p_texture.sum())
