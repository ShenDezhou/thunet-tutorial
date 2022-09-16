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
# save(model, "model.pt")
if PRETRAIN:
    model = load("model_lake/03_xor_model_io.thu")
    print(model)
    fc_layer1 = model["fc_layer1"]
    fc_layer2 = model["fc_layer2"]
    fc_layer3 = model["fc_layer3"]
    sm_layer = model["sm_layer"]

X = [[0, 1],
     [1, 0],
     [0, 0],
     [1, 1]]
Y = [1,
     1,
     0,
     0]
X_train = np.array(X)
y_train = np.eye(len([0, 1]))[Y]
prev_loss = np.inf
VERBOSE = False

for i in range(1000):
    loss, estart = 0.0, time()
    batch_size = 1
    batch_generator, nb = minibatch(X_train, batch_size, shuffle=True)

    for j, b_ix in enumerate(batch_generator):
        bsize, bstart = len(b_ix), time()  #

        X_batch = X_train[b_ix]
        y_real_batch = y_train[b_ix]

        y_pred_batch = fc_layer1.forward(X_batch)
        y_pred_batch = fc_layer2.forward(y_pred_batch)
        y_pred_batch = fc_layer3.forward(y_pred_batch)

        y_pred_batch = sm_layer.forward(y_pred_batch)

        print(y_real_batch, y_pred_batch)
        batch_loss = loss_func(y_real_batch, y_pred_batch)
        y_grad = loss_func.grad(y_real_batch, y_pred_batch)
        y_grad = sm_layer.backward(y_grad)
        y_grad = fc_layer3.backward(y_grad)
        y_grad = fc_layer2.backward(y_grad)
        y_grad = fc_layer1.backward(y_grad)

        sm_layer.update()
        fc_layer3.update()
        fc_layer2.update()
        fc_layer1.update()

        loss += batch_loss

        if VERBOSE:
            fstr = "\t[Batch {}/{}] Train loss: {:.3f} ({:.1f}s/batch)"
            print(fstr.format(j + 1, nb, batch_loss, time() - bstart))

    loss /= nb
    fstr = "[Epoch {}] Avg. loss: {:.6f}  Delta: {:.6f} ({:.2f}m/epoch)"
    print(fstr.format(i + 1, loss, prev_loss - loss, (time() - estart) / 60.0))
    prev_loss = loss

save(model, "model_lake/03_xor_model_io.thu")
print(model)