ITERATIONS = 100
SEED = 1000
DEVICE = 1
SAVE_EVERY = 100
ON_GPU = True
EPOCHS = 4000
BATCH_SIZE = 16
BIG_ENOUGH = 1000

# tf_to_chainer_names_order = ['fc1', 'fc1','conv2', 'conv2',
#                              'torgb8', 'torgb8','conv3', 'conv3',
#                              'conv4', 'conv4','torgb7', 'torgb7',
#                              'conv5', 'conv5','conv6', 'conv6',
#                              'torgb6','torgb6','conv7','conv7',
#                              'conv8','conv8','torgb5','torgb5',
#                              'conv9','conv9','conv10','conv10',
#                              'torgb4','torgb4','conv11','conv11',
#                              'conv12','conv12','torgb3','torgb3',
#                              'conv13','conv13','conv14','conv14',
#                              'torgb2','torgb2','conv15','conv15',
#                              'conv16','conv16','torgb1','torgb1',
#                              'conv17','conv17','conv18','conv18',
#                              'torgb0','torgb0']

tf_to_chainer_names_order = ['fc1', 'conv2','torgb8','conv3',
                             'conv4','torgb7','conv5','conv6',
                             'torgb6','conv7','conv8','torgb5',
                             'conv9','conv10','torgb4','conv11',
                             'conv12','torgb3','conv13','conv14',
                             'torgb2','conv15','conv16','torgb1',
                             'conv17','conv18','torgb0']

SCALARS = {'fc1': 0.015625,
           'conv2': 0.02083333395,
           'conv3': 0.02083333395,
           'conv4': 0.02083333395,
           'conv5': 0.02083333395,
           'conv6': 0.02083333395,
           'conv7': 0.02083333395,
           'conv8': 0.02083333395,
           'conv9': 0.02083333395,
           'conv10': 0.02946278267,
           'conv11': 0.02946278267,
           'conv12': 0.04166666791,
           'conv13': 0.04166666791,
           'conv14': 0.05892556533,
           'conv15': 0.05892556533,
           'conv16': 0.08333333582,
           'conv17': 0.08333333582,
           'conv18': 0.1178511307,
           'torgb8': 0.04419417307,
           'torgb7': 0.04419417307,
           'torgb6': 0.04419417307,
           'torgb5': 0.04419417307,
           'torgb4': 0.0625,
           'torgb3': 0.08838834614,
           'torgb2': 0.125,
           'torgb1': 0.1767766923,
           'torgb0': 0.25}


ALPHAS = [0.1, 0.2, 0.3, 0.4,
          0.5, 0.6, 0.7, 0.8]