from time import time
import numpy as np
import sys
sys.path.append("../thunet")
np.set_printoptions(precision=8)
from thunet.neural_nets.modules import MultiHeadedAttentionModule
from thunet.neural_nets.activations import ReLU, Sigmoid, Tanh
from thunet.neural_nets.layers import Softmax, FullyConnected, Embedding, Flatten, LayerNorm1D
from thunet.neural_nets.losses import SquaredError, CrossEntropy
from thunet.neural_nets.schedulers import ExponentialScheduler, NoamScheduler, ConstantScheduler
from thunet.neural_nets.optimizers import SGD, Adam
from thunet.neural_nets.utils import minibatch
from thunet.neural_nets.utils import save, load

vocab_size = 21128
seq_len = 512
latent_dim = 64
n_heads = 12
feature = 768
batchsize = 30

loss_func = CrossEntropy()
sch = NoamScheduler(lr=0.0001)
adam_q = Adam(lr=0.0001, lr_scheduler=sch)
adam_k = Adam(lr=0.0001, lr_scheduler=sch)
adam_v = Adam(lr=0.0001, lr_scheduler=sch)
adam_m = Adam(lr=0.0001, lr_scheduler=sch)
adam_flat = Adam(lr=0.0001, lr_scheduler=sch)
adam_fc1 = Adam(lr=0.0001, lr_scheduler=sch)
adam_ln1 = Adam(lr=0.0001, lr_scheduler=sch)
adam_fc2 = Adam(lr=0.0001, lr_scheduler=sch)
adam_ln2 = Adam(lr=0.0001, lr_scheduler=sch)

wq_layer = FullyConnected(feature, optimizer=adam_q, init="glorot_normal")
wk_layer = FullyConnected(feature, optimizer=adam_k, init="glorot_normal")
wv_layer = FullyConnected(feature, optimizer=adam_v, init="glorot_normal")
multihead = MultiHeadedAttentionModule(
        n_heads=n_heads,
        dropout_p=0,
        optimizer=adam_m,
        init="glorot_normal"
    )
flat = Flatten(optimizer=adam_flat)
Layernorm1d1 = LayerNorm1D(optimizer=adam_ln1)
FC1 = FullyConnected(n_heads * latent_dim,optimizer=adam_fc1, init="glorot_normal")
Layernorm1d2 = LayerNorm1D(optimizer=adam_ln2)
FC2 = FullyConnected(2, act_fn=ReLU(),optimizer=adam_fc2, init="glorot_normal")
sm_layer = Softmax()

PRETRAIN = True

model = {
    "q":wq_layer,
    "k":wk_layer,
    "v":wv_layer,
    "multi_head": multihead,
    "fc_layer1": FC1,
    "ln_layer1": Layernorm1d1,
    "fc_layer2": FC2,
    "ln_layer2": Layernorm1d2,
    "sm_layer": sm_layer
}
# save(model, "model.pt")
if PRETRAIN:
    model = load("model_lake/05_xor_model_io.thu")
    print(model)
    wq_layer = model["q"]
    wk_layer = model["k"]
    wv_layer = model["v"]
    multihead = model["multi_head"]
    fc_layer1 = model["fc_layer1"]
    Layernorm1d1 = model["ln_layer1"]
    fc_layer2 = model["fc_layer2"]
    Layernorm1d2 = model["ln_layer2"]
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

for i in range(100):
    loss, estart = 0.0, time()
    batch_size = 4
    batch_generator, nb = minibatch(X_train, batch_size, shuffle=True)

    for j, b_ix in enumerate(batch_generator):
        bsize, bstart = len(b_ix), time()  #

        X_batch = X_train[b_ix]
        y_real_batch = y_train[b_ix]

        K = wk_layer.forward(X_batch)
        Q = wq_layer.forward(X_batch)
        V = wv_layer.forward(X_batch)
        y_pred = multihead.forward(Q, K, V)
        y_pred = flat.forward(y_pred)
        y_pred = Layernorm1d1.forward(y_pred)
        y_pred = FC1.forward(y_pred)
        y_pred = Layernorm1d2.forward(y_pred)
        y_pred = FC2.forward(y_pred)
        y_pred = sm_layer.forward(y_pred)

        print(y_real_batch, y_pred)
        batch_loss = loss_func(y_real_batch, y_pred)
        y_grad = loss_func.grad(y_real_batch, y_pred)

        y_grad = sm_layer.backward(y_grad)
        y_grad = FC2.backward(y_grad)
        y_grad = Layernorm1d2.backward(y_grad)
        y_grad = FC1.backward(y_grad)
        y_grad = Layernorm1d1.backward(y_grad)
        y_grad = flat.backward(y_grad)
        dLdQ, dLdK, dLdV = multihead.backward(y_grad)
        wq_layer.backward(dLdQ)
        wk_layer.backward(dLdK)
        wv_layer.backward(dLdV)

        sm_layer.update()
        FC2.update()
        Layernorm1d2.update()
        FC1.update()
        Layernorm1d1.update()
        flat.update()
        multihead.update()
        wq_layer.update()
        wk_layer.update()
        wv_layer.update()


        loss += batch_loss

        if VERBOSE:
            fstr = "\t[Batch {}/{}] Train loss: {:.3f} ({:.1f}s/batch)"
            print(fstr.format(j + 1, nb, batch_loss, time() - bstart))

    loss /= nb
    fstr = "[Epoch {}] Avg. loss: {:.6f}  Delta: {:.6f} ({:.2f}m/epoch)"
    print(fstr.format(i + 1, loss, prev_loss - loss, (time() - estart) / 60.0))
    prev_loss = loss

save(model, "model_lake/05_xor_model_io.thu")
print(model)
