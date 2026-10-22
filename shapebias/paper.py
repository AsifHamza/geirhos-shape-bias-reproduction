"""Read the authors' released per-image decisions.

These are the decisions behind Figure 4: Caffe AlexNet, GoogLeNet and VGG-16,
torchvision ResNet-50, and ten human observers. Recomputing the shape bias from
them gives the paper's numbers exactly, which shows the metric code here is
the paper's metric; comparing them image by image with our own runs shows
where our models disagree with theirs.
"""

import csv

from .data import SOURCE_DIR

RAW = SOURCE_DIR / "raw-data"
EXPT = "style-transfer-512-nomask-experiment"

# Our model name -> the CSV of the network the paper reported under that name.
PAPER_MODEL_FILES = {
    "alexnet": RAW / EXPT / f"{EXPT}_alexnet_session_1.csv",
    "vgg16": RAW / EXPT / f"{EXPT}_vgg_session_1.csv",
    "googlenet": RAW / EXPT / f"{EXPT}_googlenet_session_1.csv",
    "resnet50": RAW / f"{EXPT}-trained-CNNs" / f"{EXPT}_resnet50_session-1.csv",
}

# Figure 4 as printed, for the comparison table.
PAPER_SHAPE_BIAS = {"alexnet": 0.429, "vgg16": 0.172, "googlenet": 0.312,
                    "resnet50": 0.221, "humans": 0.959}

# The texture-vs-shape README (section data-analysis) reports these for
# torchvision AlexNet and VGG-16, citing the repository's issue #7.
README_TORCHVISION_SHAPE_BIAS = {"alexnet": 0.253, "vgg16": 0.092}


def read_decisions(path):
    """{stimulus filename: decision} from one released CSV.

    The CSV's imagename looks like
    "0001_s5n_dnn_0_airplane_00_airplane1-bicycle2.png"; the part after the
    last underscore is the stimulus filename. A human may answer "na" (no
    response in time), which counts as neither shape nor texture.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    decisions = {row["imagename"].rsplit("_", 1)[1]: row["object_response"] for row in rows}
    assert len(decisions) == len(rows) == 1280, path
    return decisions


def human_files():
    return sorted((RAW / EXPT).glob(f"{EXPT}_subject-*_session_1.csv"))
