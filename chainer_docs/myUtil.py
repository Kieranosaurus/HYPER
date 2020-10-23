from HYPER.progressive_grown_gan_chainer import PGGC
import HYPER.constants as constants
import matplotlib.pyplot as plt
import numpy as np
import chainer
import h5py
from PIL import Image
import HYPER.paths as paths
import tensorflow as tf
import pickle


def load_weights_from_tf(model_chainer):

    with open('pggan_weights', 'rb') as f:
        wb = pickle.load(f)

    weights = []
    biases = []
    for i in range(0, 54, 2):
        weights.append(wb[i])
        biases.append(wb[i + 1])

    xp = np
    for idx in range(len(constants.tf_to_chainer_names_order)):
        i = constants.tf_to_chainer_names_order[idx]

        if idx == 0:
            name = 'fc1'  # bias is already set to 'nobias'
            w = weights[idx][:].astype(xp.float32) * constants.SCALARS[name]
            w = w.transpose((1, 0))
            w = chainer.variable.Parameter(w)
            assert(w.shape == model_chainer.__getattribute__(name).W.shape)
            model_chainer.__getattribute__(name).__setattr__('W', w)

            # set weights to ones
            name = 'fc2'
            b = biases[idx][:].astype(xp.float32)
            b = chainer.variable.Parameter(b)
            assert (b.shape == model_chainer.__getattribute__(name).b.shape)
            model_chainer.__getattribute__(name).__setattr__('b', b)
        else:
            w = weights[idx][:].astype(xp.float32) * constants.SCALARS[i]
            w = w.transpose((3, 2, 1, 0))
            w = chainer.variable.Parameter(w)
            assert (w.shape == model_chainer.__getattribute__(i).W.shape)
            model_chainer.__getattribute__(i).__setattr__('W', w)

            if i.__contains__('torgb'):
                name = i
            else:
                name = i + '_b'

            b = biases[idx][:].astype(xp.float32)
            b = chainer.variable.Parameter(b)
            assert (b.shape == model_chainer.__getattribute__(name).b.shape)
            model_chainer.__getattribute__(name).__setattr__('b', b)

    return model_chainer

