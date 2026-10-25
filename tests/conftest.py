"""Shared test fixture: ResNet-50's softmax outputs on all 1,280 stimuli.

Computed once per test session (about 30 seconds on a laptop CPU) and used by
the tests that need real network outputs.
"""

import pytest

import data
import models


@pytest.fixture(scope="session")
def resnet50_softmax():
    data.download()
    paths = [p for p, _, _ in data.list_stimuli()]
    return models.softmax_outputs(models.load("resnet50"), paths)
