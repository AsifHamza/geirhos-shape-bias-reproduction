"""Shared test fixture: each network's softmax outputs on all 1,280 stimuli.

Computed once per test session, for the main preprocessing and for the
resize-and-crop ablation (about five minutes on a laptop CPU), and used by the
tests that need real network outputs.
"""

import pytest

import data
import models


@pytest.fixture(scope="session")
def softmax():
    """{(network name, variant): (1280, 1000) softmax outputs}, where variant is
    "main" (models.PREPROCESS) or "resize_crop" (models.PREPROCESS_RESIZE_CROP)."""
    data.download()
    paths = [p for p, _, _ in data.list_stimuli()]
    variants = {"main": models.PREPROCESS, "resize_crop": models.PREPROCESS_RESIZE_CROP}
    return {(name, variant): models.softmax_outputs(models.load(name), paths, preprocess)
            for name in models.MODELS for variant, preprocess in variants.items()}
