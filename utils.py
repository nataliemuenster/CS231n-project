import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.nn import init

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per Assignment 3 of CS231N Spr1718 and https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def xavier_initialize(model):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'fc' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters()
    ]

    for p in parameters:
        if p.dim() >= 2:
            init.xavier_normal(p)
        else:
            init.constant(p, 0)


def gaussian_initialize(model, std=.01):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'fc' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters()
    ]

    for p in parameters:
        if p.dim() >= 2:
            init.normal_(p, std=std)
        else:
            init.constant_(p, 0)