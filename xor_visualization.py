# -*- coding: UTF-8 -*-
import sys
sys.path.append("../thunet")
import numpy as np
from time import time
np.set_printoptions(precision=8)
from thunet.neural_nets.activations import ReLU, Sigmoid, Tanh
from thunet.neural_nets.layers import Softmax, FullyConnected, Embedding
from thunet.neural_nets.losses import SquaredError, CrossEntropy
from thunet.neural_nets.schedulers import ExponentialScheduler, NoamScheduler, ConstantScheduler
from thunet.neural_nets.optimizers import SGD, Adam
from thunet.neural_nets.utils import minibatch
from thunet.neural_nets.utils import save, load

loss_func = CrossEntropy()
sch = ConstantScheduler(lr=0.0001)
fc1_adam = Adam(lr=0.0001, lr_scheduler=sch)
fc2_adam = Adam(lr=0.0001, lr_scheduler=sch)
fc3_adam = Adam(lr=0.0001, lr_scheduler=sch)
fc_layer1 = FullyConnected(8, act_fn=Tanh(), optimizer=fc1_adam)
fc_layer2 = FullyConnected(4, act_fn=Tanh(), optimizer=fc2_adam)
fc_layer3 = FullyConnected(2, act_fn=Tanh(), optimizer=fc3_adam)
sm_layer = Softmax()
PRETRAIN=True

model = {
    "fc_layer1": fc_layer1,
    "fc_layer2": fc_layer2,
    "fc_layer3": fc_layer3,
    "sm_layer": sm_layer
}

if PRETRAIN:
    model = load("model_lake/03_xor_model_io.thu")
    fc_layer1 = model["fc_layer1"]
    fc_layer2 = model["fc_layer2"]
    fc_layer3 = model["fc_layer3"]
    sm_layer = model["sm_layer"]

w1 = fc_layer1.parameters['W']
print(w1.shape)
w2 = fc_layer2.parameters['W']
print(w2.shape)
w3 = fc_layer3.parameters['W']
print(w3.shape)
print(sm_layer.n_in)

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

# Create data: 200 points
#pre-training model.
data = np.random.randn(2,28).T

#after-trained model.
w = np.concatenate([w1, w2.reshape((2,-1)), w3.reshape((2, -1))], axis=1)
data = w.T

x, y = data.T
print(x)
print(y)

# Create a figure with 6 plot areas
fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(21, 5))

# Everything starts with a Scatterplot
axes[0].set_title('Scatterplot')
axes[0].plot(x, y, 'ko')
# As you can see there is a lot of overlapping here!

# Thus we can cut the plotting window in several hexbins
nbins = 16
axes[1].set_title('Hexbin')
axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)

# 2D Histogram
axes[2].set_title('2D Histogram')
axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
k = kde.gaussian_kde(data.T)
xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# plot a density
axes[3].set_title('Calculate Gaussian KDE')
axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.BuGn_r)

# add shading
axes[4].set_title('2D Density with shading')
axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

# contour
axes[5].set_title('Contour')
axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
axes[5].contour(xi, yi, zi.reshape(xi.shape))

plt.show()