"""The 16 entry-level categories and the rule that maps 1,000 ImageNet
probabilities to one of them.

The index lists are copied from bethgelab/model-vs-human,
modelvshuman/helper/human_categories.py (MIT licence, copyright Robert Geirhos
2021; see LICENSES/model-vs-human-MIT.txt). They are identical to the lists in
rgeirhos/texture-vs-shape, the code the paper used. The decision rule below is
written from scratch; tests/test_decision_rule.py checks it against the
authors' own function.
"""

import numpy as np

# ImageNet class indices (0 to 999, torchvision order) that count as each
# category. Only 207 of the 1,000 classes appear; a prediction on any other
# class has no direct say in the decision.
CATEGORY_INDICES = {
    "airplane": [404],
    "bear": [294, 295, 296, 297],
    "bicycle": [444, 671],
    "bird": [8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23,
             24, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93,
             94, 95, 96, 98, 99, 100, 127, 128, 129, 130, 131,
             132, 133, 135, 136, 137, 138, 139, 140, 141, 142,
             143, 144, 145],
    "boat": [472, 554, 625, 814, 914],
    "bottle": [440, 720, 737, 898, 899, 901, 907],
    "car": [436, 511, 817],
    "cat": [281, 282, 283, 284, 285, 286],
    "chair": [423, 559, 765, 857],
    "clock": [409, 530, 892],
    "dog": [152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
            162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
            172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
            182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
            193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
            203, 205, 206, 207, 208, 209, 210, 211, 212, 213,
            214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
            224, 225, 226, 228, 229, 230, 231, 232, 233, 234,
            235, 236, 237, 238, 239, 240, 241, 243, 244, 245,
            246, 247, 248, 249, 250, 252, 253, 254, 255, 256,
            257, 259, 261, 262, 263, 265, 266, 267, 268],
    "elephant": [385, 386],
    "keyboard": [508, 878],
    "knife": [499],
    "oven": [766],
    "truck": [555, 569, 656, 675, 717, 734, 864, 867],
}

# Alphabetical, as in the authors' get_human_object_recognition_categories().
# The order matters only for ties: np.argmax returns the first maximum, and the
# authors' loop keeps the first category that reaches the maximum (strict ">").
CATEGORIES = sorted(CATEGORY_INDICES)

# How the probabilities inside one category are combined into one score.
# "mean" is the default of the authors' published function
# (aggregation_function=np.mean, in both texture-vs-shape and model-vs-human).
# "max" takes the most probable ImageNet class inside the category, so the
# decision is the category of the top-1 class among the 207 listed.
AGGREGATIONS = {"mean": np.mean, "max": np.max}


def aggregate(probabilities, how="mean"):
    """Combine the probabilities inside each category into one score.

    probabilities: array of shape (n_images, 1000), softmax outputs.
    how: "mean" or "max", see AGGREGATIONS.
    Returns an array of shape (n_images, 16), columns in CATEGORIES order.
    """
    function = AGGREGATIONS[how]
    probabilities = np.asarray(probabilities)
    columns = [function(probabilities[:, CATEGORY_INDICES[c]], axis=1) for c in CATEGORIES]
    return np.stack(columns, axis=1)


def decide(probabilities, how="mean"):
    """Return the 16-class decision (a category name) for each image."""
    scores = aggregate(probabilities, how)
    return [CATEGORIES[i] for i in scores.argmax(axis=1)]
