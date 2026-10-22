"""Check our 16-class decision rule against the authors' own code.

The authors' probabilities_to_decision.py (texture-vs-shape, pinned commit)
is downloaded with the stimuli into data/ and imported from there; it is not
part of this repository because that code carries no licence.
"""

import sys

import numpy as np
import pytest

from shapebias import categories, data

AUTHORS_CODE = data.SOURCE_DIR / "code"


@pytest.fixture(scope="module")
def authors_mapping():
    data.download()
    # Their module imports "helper.human_categories", relative to code/.
    sys.path.insert(0, str(AUTHORS_CODE))
    try:
        import probabilities_to_decision as ptd
    finally:
        sys.path.remove(str(AUTHORS_CODE))
    return ptd.ImageNetProbabilitiesTo16ClassesMapping()


def random_softmax(rng, n, concentration):
    """Softmax-like vectors. A small concentration gives peaked vectors, as a
    confident network does; a large one gives flat vectors, where near-ties
    between categories are likely."""
    return rng.dirichlet(np.full(1000, concentration), size=n)


@pytest.mark.parametrize("concentration", [0.01, 0.1, 1.0, 100.0])
def test_matches_authors_on_random_inputs(authors_mapping, concentration):
    rng = np.random.default_rng(0)
    probs = random_softmax(rng, 2000, concentration)
    ours = categories.decide(probs)
    theirs = [authors_mapping.probabilities_to_decision(p) for p in probs]
    assert ours == theirs


def test_matches_authors_on_model_outputs(authors_mapping):
    """The real softmax outputs, if reproduce has cached them."""
    caches = sorted((data.DATA_DIR / "softmax").glob("*.npy"))
    if not caches:
        pytest.skip("run python -m shapebias.reproduce first")
    for cache in caches:
        probs = np.load(cache)
        theirs = [authors_mapping.probabilities_to_decision(p) for p in probs]
        assert categories.decide(probs) == theirs, cache.name


def test_ties_go_to_the_first_category(authors_mapping):
    """With every probability equal, all 16 means tie and both rules must pick
    the alphabetically first category.

    2**-10 rather than 1/1000: a power of two is stored exactly, so every
    category mean is exactly equal. With 1/1000, rounding inside np.mean makes
    the means differ in the last bit and "dog" wins (both rules agree on that
    too, but it is no longer a tie)."""
    probs = np.full((1, 1000), 2.0 ** -10)
    assert categories.decide(probs) == [authors_mapping.probabilities_to_decision(probs[0])] == ["airplane"]


def test_index_lists_match_authors(authors_mapping):
    import helper.human_categories as hc  # cached by the fixture's import
    c = hc.HumanCategories()
    for name in categories.CATEGORIES:
        assert categories.CATEGORY_INDICES[name] == c.get_imagenet_indices_for_category(name)
    assert categories.CATEGORIES == hc.get_human_object_recognition_categories()
