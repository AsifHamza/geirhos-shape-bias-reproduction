"""Shared test fixture: each network's softmax outputs on all 1,280 stimuli.

Computed once per test session (about four minutes on a laptop CPU for the
four networks) and used by the tests that need real network outputs.
"""

import pytest

import data
import models


@pytest.fixture(scope="session")
def softmax():
    """{network name: (1280, 1000) softmax outputs}, for all four networks."""
    data.download()
    paths = [p for p, _, _ in data.list_stimuli()]
    return {name: models.softmax_outputs(models.load(name), paths) for name in models.MODELS}
