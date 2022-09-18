# XOR learning with BERT architecture
Today, I am going to introduce building BERT architecture in XOR learning.
The pre-training and tokenization steps for BERT are too complicated, hence I would take a simple classification task to show that how to build and train a BERT model using THUNET.


# What You Can Learn?
In this tutorial, I am telling how to build a model to act as 'XOR' operator, which is a common problem in Computer Science.
After finishing the lesson, readers should be able to create Activation, Loss, Layer, Scheduler and Optimizer objects, which are key components of the deep learning network.
Readers will learn how to prepare data and train the model.
And in the end, readers have a clear understanding that the model layers' Forward, Backward and Update process.

And audience can learn the famous BERT architecture to make classification task in Machine Learning.

# Introduction to THUNET
If you are familiar with THUNET already, you can skip this part to save time.

## What is THUNET?
A deep learning net/framework named "TsingHua University NET", short for "THUNET", is for non-commercial, educational, scientific purpose for the deep learning community.
Today, I am proud to announce that all the mandatory component of the framework, THUNET, have been completed. Students, teachers and scientiests are free and welcome to use this deep learning framework.

## How to build a neural network with THUNET?
Next, I will explain how to use THUNET to build a model to compute XOR operator.
And I will explain the steps in detail: forward, loss, backward grad, backward.

## Tutorial-5: XOR operand
The XOR operand is that: it tells two operands if they are different or not.
Specifically, 1 xor 0 makes 1, and 0 xor 0 makes 0.

### Preparation
We should make a python environment ready. According to [THUNET's guideline](https://pypi.org/project/thunet), the following python versions are required: 2.7, 3.5, 3.6, 3.7, 3.8, 3.9, or 3.10.

### Python Packages Install
Install the package by pip command:
`pip install thunet`

### Network Detail
If you are not interested in details, you can skip this part.

#### Loss function
loss_func = CrossEntropy()

#### Scheduler
sch = NoamScheduler(lr=0.0001)

#### Optimizer
adam_q = Adam(lr=0.0001, lr_scheduler=sch)
adam_k = Adam(lr=0.0001, lr_scheduler=sch)
adam_v = Adam(lr=0.0001, lr_scheduler=sch)
adam_m = Adam(lr=0.0001, lr_scheduler=sch)
adam_flat = Adam(lr=0.0001, lr_scheduler=sch)
adam_fc1 = Adam(lr=0.0001, lr_scheduler=sch)
adam_ln1 = Adam(lr=0.0001, lr_scheduler=sch)
adam_fc2 = Adam(lr=0.0001, lr_scheduler=sch)
adam_ln2 = Adam(lr=0.0001, lr_scheduler=sch)

#### Neural Layer
wq_layer = FullyConnected(feature, optimizer=adam_q, init="glorot_normal")
wk_layer = FullyConnected(feature, optimizer=adam_k, init="glorot_normal")
wv_layer = FullyConnected(feature, optimizer=adam_v, init="glorot_normal")
multihead = MultiHeadedAttentionModule(n_heads=n_heads,dropout_p=0,optimizer=adam_m,init="glorot_normal")
flat = Flatten(optimizer=adam_flat)
Layernorm1d1 = LayerNorm1D(optimizer=adam_ln1)
FC1 = FullyConnected(n_heads * latent_dim,optimizer=adam_fc1, init="glorot_normal")
Layernorm1d2 = LayerNorm1D(optimizer=adam_ln2)
FC2 = FullyConnected(2, act_fn=ReLU(),optimizer=adam_fc2, init="glorot_normal")
sm_layer = Softmax()

#### Data preparation
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

#### Pretrained Model Loading
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


#### Make Batch using minibatch
batch_generator, nb = minibatch(X_train, batch_size, shuffle=True)
for j, b_ix in enumerate(batch_generator):


#### Forward
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

#### Compute the loss
batch_loss = loss_func(y_real_batch, y_pred)


#### Compute the loss grad
y_grad = loss_func.grad(y_real_batch, y_pred_batch)

#### Backward
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

#### Weight Update
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

#### Trained Model Saving
save(model, "model_lake/05_xor_model_io.thu")


### Training Result
By use model checkpoints, you will be able to get constant results. The saving and loading step in the tutorial save the training time in getting the correct model.
There is a confidential level of 100% to reproduce the result if you use a pretrained correct model, e.g. [xor-model](https://transformers-model.obs.cn-north-4.myhuaweicloud.com/thunet/xor/05_xor_model_io.thu). You will constantly get the same correct model using the pretrain-finetune pattern. 
We print the actual value followed by the predicted value.
The model cannot predict exact same as the actual value, but using `argmax` function, we are able to tell the correct class.

The actual value is represented by one-hot vector. Thus, [1, 0] shows it belongs to class-0, and [0, 1] shows that it belongs to class-1.
And the predicted value in epoch-10, [0.516938 0.483062] shows it is class-0, and [0.48051901 0.51948099] shows it is class-1.
```
[[1. 0.]
 [0. 1.]
 [1. 0.]
 [0. 1.]] [[3.76915309e-01 6.23084691e-01]
 [3.84601065e-01 6.15398935e-01]
 [9.99405130e-01 5.94869811e-04]
 [4.10812293e-01 5.89187707e-01]]
[Epoch 100] Avg. loss: 1.990825  Delta: 0.000654 (0.00m/epoch)
```

### Test Result
From the output of epoch-100, we can compute the accuracy of the model, it has accuracy of 100%.

### Space Advantage of using THUNET
The [model](https://transformers-model.obs.cn-north-4.myhuaweicloud.com/thunet/xor/05_xor_model_io.thu) file size is 49.21M, however the same model size is 59.6M, if saved in zip format. Thus, the THUNET has a storage saving advantage of 17.45%.
This is a notable feature if you train a large model, it will save your disk space by 17%+.

### Supplimentary
Readers can use this tutorial to get a general picture of a deep learning model.
Github: [thunet-tutorial](https://github.com/ShenDezhou/thunet-tutorial)
