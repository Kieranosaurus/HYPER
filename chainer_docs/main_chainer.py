from HYPER.progressive_grown_gan_chainer import *
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
from HYPER.myUtil import *
import numpy as np
import pickle


def ch_face_from_latent(model, latents, my_path):
    latents = np.expand_dims(np.expand_dims(latents, -1), -1).astype(np.float32)
    # activations = np.zeros((latents.shape[0], 512, 4, 4))
    for i in range(latents.shape[0]):
        latent = np.expand_dims(latents[i], 0)

        face, l1_output = model(np.float32(latent))
        face = face.data
        # activations[i] = l1_output.data

        face = np.clip(np.rint((face + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
        face = face.transpose((0, 3, 2, 1))
        Image.fromarray(face[0], 'RGB').save(my_path + '/%d.png' % i)

    return face


model = PGGC()
model = load_weights_from_tf(model)
my_path = '/home/tdado/PycharmProjects/HYPER/faces_chainer/'

# predict s1
with open('data_sub1_4096.dat', 'rb') as fp:
    X_train, T_train, X_test, T_test = pickle.load(fp)

# new ground truth labels
test_activations = face_from_latent(model, T_test, my_path + 'test_hyper_1')
