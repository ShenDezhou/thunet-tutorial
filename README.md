# What is THUNET?
A deep learning net/framework named "TsingHua University NET", short for "THUNET", is for non-commercial, educational, scientific purpose for the deep learning community.

# How to build a neural network with THUNET?
Next, I will explain how to use THUNET to build a model to compute XOR operator.

## Tutorial-1: XOR operand
The XOR operand is that: it tells two operands if they are different or not.
Specifically, 1 xor 0 makes 1, and 0 xor 0 makes 0.

## What You Can Learn?
In this tutorial, I am telling how to build a model to act as 'XOR' operator, which is a common problem in Computer Science.
After finishing the lesson, readers should be able to create Activation, Loss, Layer, Scheduler and Optimizer objects, which are key components of the deep learning network.
Readers will learn how to prepare data and train the model.
And in the end, readers have a clear understanding that the model layers' Forward, Backward and Update process.

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
sch = ConstantScheduler(lr=0.01)

#### Optimizer
fc1_adam = Adam(lr=0.01, lr_scheduler=sch)
fc2_adam = Adam(lr=0.01, lr_scheduler=sch)

#### Neural Layer
fc_layer1 = FullyConnected(16, act_fn=Tanh(), optimizer=fc1_adam)
fc_layer2 = FullyConnected(2, act_fn=Tanh(), optimizer=fc2_adam)
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

#### Make Batch using minibatch

batch_generator, nb = minibatch(X_train, batch_size, shuffle=True)
for j, b_ix in enumerate(batch_generator):

#### Layer Weight Update
y_pred_batch = fc_layer1.forward(X_batch)
y_pred_batch = fc_layer2.forward(y_pred_batch)
y_pred_batch = sm_layer.forward(y_pred_batch)

y_pred_batch = sm_layer.backward(y_pred_batch)
y_pred_batch = fc_layer2.backward(y_pred_batch)
y_pred_batch = fc_layer1.backward(y_pred_batch)

sm_layer.update()
fc_layer2.update()
fc_layer1.update()

#### Compute the loss
batch_loss = loss_func(y_real_batch, y_pred_batch)

### Training Result
*This part changes according to different experiment.* There is a confidential level of 60%(3/5) to reproduce the result. 
We print the actual value followed by the predicted value.
The model cannot predict exact same as the actual value, but using `argmax` function, we are able to tell the correct class.

The actual value is represented by one-hot vector. Thus, [1, 0] shows it belongs to class-0, and [0, 1] shows that it belongs to class-1.
And the predicted value in epoch-10, [0.516938 0.483062] shows it is class-0, and [0.48051901 0.51948099] shows it is class-1(counting from 1, the first class).
```
[[1. 0.]] [[0.5 0.5]]
[[0. 1.]] [[0.64168317 0.35831683]]
[[1. 0.]] [[0.63063128 0.36936872]]
[[0. 1.]] [[0.36888144 0.63111856]]
[Epoch 1] Avg. loss: 0.660195  Delta: inf (0.00m/epoch)
[[1. 0.]] [[0.5806954 0.4193046]]
[[1. 0.]] [[0.45599229 0.54400771]]
[[0. 1.]] [[0.35737045 0.64262955]]
[[0. 1.]] [[0.52543727 0.47456273]]
[Epoch 2] Avg. loss: 0.629089  Delta: 0.031106 (0.00m/epoch)
[[1. 0.]] [[0.46678321 0.53321679]]
[[0. 1.]] [[0.50995437 0.49004563]]
[[0. 1.]] [[0.37220265 0.62779735]]
[[1. 0.]] [[0.50661093 0.49338907]]
[Epoch 3] Avg. loss: 0.655174  Delta: -0.026085 (0.00m/epoch)
[[0. 1.]] [[0.49805414 0.50194586]]
[[1. 0.]] [[0.50337561 0.49662439]]
[[1. 0.]] [[0.50563422 0.49436578]]
[[0. 1.]] [[0.40342127 0.59657873]]
[Epoch 4] Avg. loss: 0.643542  Delta: 0.011632 (0.00m/epoch)
[[0. 1.]] [[0.41142339 0.58857661]]
[[1. 0.]] [[0.5083185 0.4916815]]
[[1. 0.]] [[0.53216966 0.46783034]]
[[0. 1.]] [[0.49655871 0.50344129]]
[Epoch 5] Avg. loss: 0.630944  Delta: 0.012598 (0.00m/epoch)
[[0. 1.]] [[0.49723218 0.50276782]]
[[0. 1.]] [[0.45071719 0.54928281]]
[[1. 0.]] [[0.52734092 0.47265908]]
[[1. 0.]] [[0.54994751 0.45005249]]
[Epoch 6] Avg. loss: 0.631152  Delta: -0.000208 (0.00m/epoch)
[[0. 1.]] [[0.49810482 0.50189518]]
[[1. 0.]] [[0.54824217 0.45175783]]
[[0. 1.]] [[0.47170895 0.52829105]]
[[1. 0.]] [[0.53109125 0.46890875]]
[Epoch 7] Avg. loss: 0.640333  Delta: -0.009181 (0.00m/epoch)
[[1. 0.]] [[0.52875858 0.47124142]]
[[0. 1.]] [[0.48093632 0.51906368]]
[[0. 1.]] [[0.47534703 0.52465297]]
[[1. 0.]] [[0.53049834 0.46950166]]
[Epoch 8] Avg. loss: 0.642977  Delta: -0.002644 (0.00m/epoch)
[[0. 1.]] [[0.47683366 0.52316634]]
[[0. 1.]] [[0.46818057 0.53181943]]
[[1. 0.]] [[0.52276494 0.47723506]]
[[1. 0.]] [[0.51853856 0.48146144]]
[Epoch 9] Avg. loss: 0.646168  Delta: -0.003191 (0.00m/epoch)
[[1. 0.]] [[0.516938 0.483062]]
[[1. 0.]] [[0.51642774 0.48357226]]
[[0. 1.]] [[0.46152795 0.53847205]]
[[0. 1.]] [[0.48051901 0.51948099]]
[Epoch 10] Avg. loss: 0.648649  Delta: -0.002481 (0.00m/epoch)
```

### Test Result
From the output of epoch-10, we can compute the accuracy of the model, it has accuracy of 100%.

### Supplimentary
Readers can use this tutorial to get a general picture of a deep learning model.
Github: [thunet-tutorial](https://github.com/ShenDezhou/thunet-tutorial)

# Changelog

> 2022.9.9 202209_v1 modify Constant scheduler with Noam scheduler to have more chance to convergence.