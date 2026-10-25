"""The four ImageNet classifiers of Figure 4 and the forward pass.

The paper used Caffe weights for AlexNet, VGG-16 and GoogLeNet and torchvision
for ResNet-50. Here all four are torchvision's original ImageNet weights
(IMAGENET1K_V1). That is a deliberate change, stated in the write-up.
"""

import numpy as np
import torch
import torchvision.models as tvm
from PIL import Image
from torchvision import transforms

# IMAGENET1K_V1 is named explicitly: for ResNet-50, torchvision's DEFAULT is
# IMAGENET1K_V2, a 2021 retraining with a different recipe, not the 2018
# weights the paper used.
MODELS = {
    "alexnet": (tvm.alexnet, tvm.AlexNet_Weights.IMAGENET1K_V1),
    "vgg16": (tvm.vgg16, tvm.VGG16_Weights.IMAGENET1K_V1),
    "googlenet": (tvm.googlenet, tvm.GoogLeNet_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V1),
}

# The stimuli are already 224 by 224, so there is no resize or crop. The
# weights' own transforms() would resize to 256 and centre-crop 224, which
# zooms in and cuts off part of the object outline (the shape cue).
PREPROCESS = transforms.Compose([
    transforms.ToTensor(),  # uint8 HWC in 0..255 -> float CHW in 0..1
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# An ablation, not the reproduction: how far shape bias moves under the usual
# ImageNet evaluation transform (resize the short side to 256, keep the central
# 224 by 224), applied to images that are already 224 by 224.
PREPROCESS_RESIZE_CROP = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    PREPROCESS,
])


def load(name):
    builder, weights = MODELS[name]
    model = builder(weights=weights)
    # eval() switches dropout off and makes batch norm use its stored
    # statistics; without it the outputs change from batch to batch.
    return model.eval()


def softmax_outputs(model, paths, preprocess=PREPROCESS, batch_size=64):
    """Return an (n_images, 1000) array of softmax probabilities.

    Runs on the CPU so the numbers do not depend on which GPU backend, if any,
    the machine has.
    """
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([preprocess(Image.open(p).convert("RGB"))
                                 for p in paths[start:start + batch_size]])
            logits = model(batch)
            outputs.append(torch.softmax(logits, dim=1).double().numpy())
    return np.concatenate(outputs)
