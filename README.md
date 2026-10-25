# Reproducing the texture bias of ImageNet CNNs (Geirhos et al., ICLR 2019)

## The claim

"A pretrained classifier follows texture over shape on cue-conflict images": Figure 4 of R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann and W. Brendel, *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness*, ICLR 2019. On the 1,280 released cue-conflict images the paper reports a shape bias of 42.9% for AlexNet, 17.2% for VGG-16, 31.2% for GoogLeNet and 22.1% for ResNet-50, against 95.9% for humans.

Shape bias is the fraction of shape decisions among the decisions that name either the shape or the texture category, over the 1,200 images whose shape and texture differ.

## Route

The pipeline is my own code: loading the stimuli, the forward passes, the 1,000-to-16 class decision rule, the metric and the figures. The 16 lists of ImageNet indices are copied from the authors' MIT-licensed [model-vs-human](https://github.com/bethgelab/model-vs-human) (see `LICENSES/`). The authors' own decision function from [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape) is downloaded at run time and used only by the tests, to check that my rule gives the same decision on every input.

All four networks are torchvision's original ImageNet weights (`IMAGENET1K_V1`), run on the CPU. The paper used Caffe weights for AlexNet, VGG-16 and GoogLeNet, and torchvision for ResNet-50.

The decision rule takes, for each of the 16 categories, the largest softmax probability among that category's ImageNet classes, and picks the category with the largest value. The authors' published function averages instead (`aggregation_function=np.mean` by default), but the mean does not reproduce their released decisions: for torchvision ResNet-50, the paper's own network, the max gives the released decision on all 1,280 images and the mean on 85% of the 1,200 conflict images. The max also reproduces the torchvision AlexNet and VGG-16 figures in the texture-vs-shape README (25.3% and 9.2%). The authors' README acknowledges the difference without naming the rule: in its section on the Stylized-ImageNet models it says those values "are computed using a slightly different probability aggregation method as reported in the paper. We here used the average", and lists 21.39% for a standard ResNet-50, which is what the mean gives here. Their earlier paper ([Geirhos et al., NeurIPS 2018](https://arxiv.org/abs/1808.08750), updated appendix) used summation. No source states the max; it is identified by the exact match with the released decisions. The rule is set by `AGGREGATION` in `main.py`.

## Re-running

```
conda env create -f environment.yml
conda activate shape-bias
python main.py
```

This downloads the stimuli and the authors' released per-image decisions from commit `ccd0c01` of texture-vs-shape (a 163 MB archive, of which about 140 MB is kept; the stimuli are checked against a SHA-256), downloads the four torchvision weight files (about 910 MB, into torch's cache), and writes everything in `results/`. It takes about five minutes on a laptop CPU (an Apple M4 Pro; 4 min 49 s and 5 min 26 s in the last two runs). Then

```
python -m pytest
```

takes about 70 seconds (it runs ResNet-50 once itself), checks the decision rule against the authors' code with both aggregations, checks that the metric, fed the authors' released decisions, gives the Figure 4 numbers exactly, and checks that max aggregation on ResNet-50 gives the released decisions.

## Outputs

- `results/summary.csv`: per network, the paper's number, the number recomputed from the authors' released decisions, ours, our shape, texture and "neither" fractions, the released "neither" fraction, a probability-weighted shape bias, and the image-by-image agreement with the paper's network.
- `results/per_category_shape_bias.csv`: shape bias per shape category, ours and released.
- `results/ablation_resize_crop.csv`: shape bias when the usual resize-256, centre-crop-224 evaluation transform is applied first.
- `results/aggregation_mean_vs_max.csv`: shape bias and agreement with the released decisions under each aggregation, mean (the published function's default) and max (used for every other result).
- `results/figure4_reproduction.png`: Figure 4 redrawn, ours next to the released decisions.
- `results/figure_agreement.png`: per network, the fraction of conflict images on which our decision equals the paper's released decision, with max aggregation, with mean aggregation, and with max after resize and crop.
- `results/decisions_<network>.csv`: per image, our decision, the paper's decision and the 16 category scores.

## Layout

- `main.py`: the one command; runs the steps below in order and writes `results/`.
- `data.py`: download, checksum and filename parsing of the stimuli.
- `categories.py`: the 16 categories and the decision rule.
- `models.py`: the networks and preprocessing.
- `metrics.py`: shape bias, per-category values and the probability-weighted variant.
- `paper.py`: reading the authors' released decisions.
- `plots.py`: the two figures.

## Licences

The stimuli and released decisions are by Geirhos et al. under CC BY 4.0 (texture-vs-shape `DATASET_LICENSE`, downloaded alongside them). The category index lists are from model-vs-human, MIT, copyright Robert Geirhos 2021.
