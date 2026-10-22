"""Check the shape-bias code against the paper's own numbers.

Fed the authors' released per-image decisions, our metric must give exactly
the values printed in Figure 4.
"""

import pytest

from shapebias import data, metrics, paper


@pytest.fixture(scope="module")
def stimuli():
    data.download()
    st = data.list_stimuli()
    return [p.name for p, _, _ in st], [s for _, s, _ in st], [t for _, _, t in st]


def test_stimulus_counts(stimuli):
    names, shapes, textures = stimuli
    assert len(names) == 1280
    assert sum(s == t for s, t in zip(shapes, textures)) == 80


@pytest.mark.parametrize("model", list(paper.PAPER_MODEL_FILES))
def test_released_decisions_give_figure4(stimuli, model):
    names, shapes, textures = stimuli
    released = paper.read_decisions(paper.PAPER_MODEL_FILES[model])
    bias = metrics.summary(shapes, textures, [released[n] for n in names])["shape_bias"]
    assert round(bias, 3) == paper.PAPER_SHAPE_BIAS[model]


def test_released_humans_give_figure4(stimuli):
    names, shapes, textures = stimuli
    humans = [paper.read_decisions(f) for f in paper.human_files()]
    assert len(humans) == 10
    bias = metrics.summary(shapes * 10, textures * 10,
                           [h[n] for h in humans for n in names])["shape_bias"]
    assert round(bias, 3) == paper.PAPER_SHAPE_BIAS["humans"]
