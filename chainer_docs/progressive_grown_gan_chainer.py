import chainer
from chainer.links import Convolution2D, Linear, Deconvolution2D, Bias
from chainer.functions import leaky_relu, linear, resize_images, matmul
# from chainercv.transforms.image.resize import resize
import numpy as np
import cv2
import tensorflow as tf
# from upsample import upsample
from chainer import Variable
from chainer.backends import cuda
import HYPER.constants as constants
import cupy


class PGGC(chainer.Chain):
    @staticmethod
    def upscale2d(x, f=2):
        x = chainer.functions.repeat(x, f, axis=2)
        x = chainer.functions.repeat(x, f, axis=3)
        return x

    @staticmethod
    def pixel_norm(x, epsilon=1e-8):
        # print(type(x))
        tmp = chainer.functions.square(x)
        tmp = chainer.functions.mean(tmp, axis=1, keepdims=True) + epsilon
        tmp = chainer.functions.rsqrt(tmp)
        tmp = x * chainer.functions.broadcast_to(tmp, (x.shape[0], x.shape[1], x.shape[-1], x.shape[-1]))
        return tmp
    
    @staticmethod
    def normal_norm(x):
        tmp = chainer.functions.broadcast_to(chainer.functions.min(x), x.shape)
        tmp2 = chainer.functions.broadcast_to(chainer.functions.max(x) - chainer.functions.min(x), x.shape)
        x = (x - tmp) / tmp2
        return x

    @staticmethod
    def reshape(x):
        x = chainer.functions.reshape(x, (x.shape[0], 512, 4, 4))
        # x = chainer.functions.expand_dims(x, 0)
        return x

    @staticmethod
    def grow(x1, x2, a):

        x = (x2 - x1) * (1-a) + x2

        # b = 1 - a
        # x = (a * x1) + (b * x2)
        #
        # x = (x2 - x1) * (1-a)
        # x = x + x1
        return x

    @staticmethod
    def clip(x):
        x = chainer.functions.clip(x, 0.0, 1.0)
        return x

    def __init__(self):
        super(PGGC, self).__init__()
        with self.init_scope():
            self.fc1 = Linear(in_size=512, out_size=8192, nobias=True)
            self.fc2 = Bias(shape=(512,))
            self.conv2 = Convolution2D(in_channels=512, out_channels=512, stride=1, ksize=3, pad=1, nobias=True)
            self.conv2_b = Bias(shape=(512,))
            # -- 2
            self.torgb8 = Convolution2D(in_channels=512, out_channels=3, stride=1, ksize=1, pad=0)
            self.conv3 = Convolution2D(in_channels=512, out_channels=512, stride=1, ksize=3, pad=1, nobias=True)
            self.conv3_b = Bias(shape=(512,))
            self.conv4 = Convolution2D(in_channels=512, out_channels=512, stride=1, ksize=3, pad=1, nobias=True)
            self.conv4_b = Bias(shape=(512,))
            # -- 3
            self.torgb7 = Convolution2D(in_channels=512, out_channels=3, stride=1, ksize=1, pad=0)
            self.conv5 = Convolution2D(in_channels=512, out_channels=512, stride=1, ksize=3, pad=1, nobias=True)
            self.conv5_b = Bias(shape=(512,))
            self.conv6 = Convolution2D(in_channels=512, out_channels=512, stride=1, ksize=3, pad=1, nobias=True)
            self.conv6_b = Bias(shape=(512,))
            # -- 4
            self.torgb6 = Convolution2D(in_channels=512, out_channels=3, stride=1, ksize=1, pad=0)
            self.conv7 = Convolution2D(in_channels=512, out_channels=512, stride=1, ksize=3, pad=1, nobias=True)
            self.conv7_b = Bias(shape=(512,))
            self.conv8 = Convolution2D(in_channels=512, out_channels=512, stride=1, ksize=3, pad=1, nobias=True)
            self.conv8_b = Bias(shape=(512,))
            # -- 5
            self.torgb5 = Convolution2D(in_channels=512, out_channels=3, stride=1, ksize=1, pad=0)
            self.conv9 = Convolution2D(in_channels=512, out_channels=256, stride=1, ksize=3, pad=1, nobias=True)
            self.conv9_b = Bias(shape=(256,))
            self.conv10 = Convolution2D(in_channels=256, out_channels=256, stride=1, ksize=3, pad=1, nobias=True)
            self.conv10_b = Bias(shape=(256,))
            # -- 6
            self.torgb4 = Convolution2D(in_channels=256, out_channels=3, stride=1, ksize=1, pad=0)
            self.conv11 = Convolution2D(in_channels=256, out_channels=128, stride=1, ksize=3, pad=1, nobias=True)
            self.conv11_b = Bias(shape=(128,))
            self.conv12 = Convolution2D(in_channels=128, out_channels=128, stride=1, ksize=3, pad=1, nobias=True)
            self.conv12_b = Bias(shape=(128,))
            # -- 7
            self.torgb3 = Convolution2D(in_channels=128, out_channels=3, stride=1, ksize=1, pad=0)
            self.conv13 = Convolution2D(in_channels=128, out_channels=64, stride=1, ksize=3, pad=1, nobias=True)
            self.conv13_b = Bias(shape=(64,))
            self.conv14 = Convolution2D(in_channels=64, out_channels=64, stride=1, ksize=3, pad=1, nobias=True)
            self.conv14_b = Bias(shape=(64,))
            # -- 8
            self.torgb2 = Convolution2D(in_channels=64, out_channels=3, stride=1, ksize=1, pad=0)
            self.conv15 = Convolution2D(in_channels=64, out_channels=32, stride=1, ksize=3, pad=1, nobias=True)
            self.conv15_b = Bias(shape=(32,))
            self.conv16 = Convolution2D(in_channels=32, out_channels=32, stride=1, ksize=3, pad=1, nobias=True)
            self.conv16_b = Bias(shape=(32,))
            # -- 9
            self.torgb1 = Convolution2D(in_channels=32, out_channels=3, stride=1, ksize=1, pad=0)
            self.conv17 = Convolution2D(in_channels=32, out_channels=16, stride=1, ksize=3, pad=1, nobias=True)
            self.conv17_b = Bias(shape=(16,))
            self.conv18 = Convolution2D(in_channels=16, out_channels=16, stride=1, ksize=3, pad=1, nobias=True)
            self.conv18_b = Bias(shape=(16,))
            self.torgb0 = Convolution2D(in_channels=16, out_channels=3, stride=1, ksize=1, pad=0)

    def __call__(self, x, layer, n_in=0):
        # pixel norm after each conv doesn't matter. only at the end and after each block

        # -- 1
        h = self.pixel_norm(x)  # 1 x 512
        h = self.fc1(h)  # 1 x 8192, only multiply with weights here
        h = self.reshape(h)
        h = self.fc2(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)
        h = self.conv2(h)
        h = self.conv2_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)  # 512 x 4 x 4
        p = self.torgb8(h)
        p = self.upscale2d(p) # <--- L1 output
        # if layer == 1:
        #     return p[:, :, 2:6, 2:6]   # 8 --> 4
        # -- 2
        h = self.upscale2d(h)  # 512 x 8 x 8
        h = self.conv3(h)
        h = self.conv3_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)
        h = self.conv4(h)
        h = self.conv4_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)  # 512 x 8 x 8
        p_ = self.torgb7(h)
        p = self.grow(p, p_, constants.ALPHAS[0])
        p = self.upscale2d(p)  # < ---
        # if layer == 2:
        #     return p[:, :, 4:12, 4:12]    # 16 --> 8
        # -- 3
        h = self.upscale2d(h)  # 512 x 16 x 16
        h = self.conv5(h)
        h = self.conv5_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)
        h = self.conv6(h)
        h = self.conv6_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)  # 512 x 16 x 16
        p_ = self.torgb6(h)
        p = self.grow(p, p_, constants.ALPHAS[1])
        p = self.upscale2d(p)
        # if layer == 3:
        #     return p[:, :, 8:24, 8:24]   # 32 --> 16
        # -- 4
        h = self.upscale2d(h)  # 512 x 32 x 32
        h = self.conv7(h)
        h = self.conv7_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)
        h = self.conv8(h)
        h = self.conv8_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)  # 512 x 32 x 32
        p_ = self.torgb5(h)
        p = self.grow(p, p_, constants.ALPHAS[2])
        p = self.upscale2d(p)
        # if layer == 4:
        #     return p[:, :, 16:48, 16:48]
        # -- 5
        h = self.upscale2d(h)  # 256 x 64 x 64
        h = self.conv9(h)
        h = self.conv9_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)
        h = self.conv10(h)
        h = self.conv10_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)  # 256 x 64 x 64
        p_ = self.torgb4(h)
        p = self.grow(p, p_, constants.ALPHAS[3])
        p = self.upscale2d(p)
        # if layer == 5:
        #     return p[:, :, 32:96, 32:96]
        # -- 6
        # print('l6')
        h = self.upscale2d(h)  # 128 x 128 x 128
        h = self.conv11(h)
        h = self.conv11_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)
        h = self.conv12(h)
        h = self.conv12_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)  # 128 x 128 x 128
        p_ = self.torgb3(h)
        p = self.grow(p, p_, constants.ALPHAS[4])
        p = self.upscale2d(p)
        # if layer == 6:
        #     return p
        # -- 7
        # print('l7')
        h = self.upscale2d(h)  # 64 x 256 x 256
        h = self.conv13(h)
        h = self.conv13_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)
        h = self.conv14(h)
        h = self.conv14_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)  # 64 x 256 x 256
        p_ = self.torgb2(h)
        p = self.grow(p, p_, constants.ALPHAS[5])
        p = self.upscale2d(p)
        # if layer == 7:
        #     return h
        # -- 8
        h = self.upscale2d(h)  # 32 x 512 x 512
        h = self.conv15(h)
        h = self.conv15_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)
        h = self.conv16(h)
        h = self.conv16_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)  # 32 x 512 x 512
        p_ = self.torgb1(h)
        p = self.grow(p, p_, constants.ALPHAS[6])
        p = self.upscale2d(p)
        # if layer == 8:
        #     return h
        # -- 9
        # print('l9')
        h = self.upscale2d(h)  # 16 x 1024 x 1024
        # print('-9- h mean after upscale: %s' % str(chainer.functions.mean(h)))
        h = self.conv17(h)
        h = self.conv17_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)
        h = self.conv18(h)
        h = self.conv18_b(h)
        h = leaky_relu(h)
        h = self.pixel_norm(h)  # 16 x 1024 x 1024
        # if layer == 9:
        #     return h
        # -- last
        p_ = self.torgb0(h)
        return p_
