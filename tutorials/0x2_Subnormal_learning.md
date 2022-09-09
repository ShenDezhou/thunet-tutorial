# What is THUNET?
A deep learning net/framework named "TsingHua University NET", short for "THUNET", is for non-commercial, educational, scientific purpose for the deep learning community.

# How to build a neural network with THUNET?
Next, I will explain how to use THUNET to build a model to tell which one is subnormal number from two operands.

## Tutorial-2: Subnormal picking operand
**Subnormal background** 
In Computer Science, floating value are stored with precision loss in storage. Thus with limit bytes for a float number, there is a minimal number that a float can represent. For example, normally a number smaller than 1.1754944e-38 are considered as zero, however, with software technology, even smaller number can be represented, however, there is still a limit for this, that is called subnormal number, which is 
1.4012985e-45 for x86_64 CPU in most case.


The XOR operand is that: it tells two small float number operands which is not subnormal float number.
Specifically, 0 tells the first is subnormal number, and 1 tells the second.

## What You Can Learn?
In this tutorial, I am telling how to build a model to tell subnormal float number, which is a basic concept in representing float value using bytes in Computer Science.
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
sch = NoamScheduler(lr=1e-5)

#### Optimizer
fc1_adam = Adam(lr=1e-5, lr_scheduler=sch)
fc2_adam = Adam(lr=1e-5, lr_scheduler=sch)

#### Neural Layer
fc_layer1 = FullyConnected(8, act_fn=Tanh(), optimizer=fc1_adam, init="he_normal")
fc_layer2 = FullyConnected(2, act_fn=Tanh(), optimizer=fc2_adam, init="he_normal")
sm_layer = Softmax()

#### Data preparation
X = [[1.4-44, 2e-30],
    [1.4-44, 1e-30],
    [1e-30, 1.4-44],
     [2e-30, 1.4-44]]
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
*This part changes according to different experiment.* There is a confidential level of 40%(2/5) to reproduce the result. 
We print the actual value followed by the predicted value.
The model cannot predict exact same as the actual value, but using `argmax` function, we are able to tell the correct class.

The actual value is represented by one-hot vector. Thus, [1, 0] shows it belongs to class-0, and [0, 1] shows that it belongs to class-1.
And the predicted value in epoch-1000, [0.5000000527 0.4999999473] shows it is class-0, and [0.4999978552 0.5000021448] shows it is class-1.
```
[[1. 0.]] [[0.5000000527 0.4999999473]]
[[0. 1.]] [[0.4999978552 0.5000021448]]
[[0. 1.]] [[0.4999978716 0.5000021284]]
[[1. 0.]] [[0.50000008 0.49999992]]
[Epoch 1000] Avg. loss: 0.693145  Delta: -0.000000 (0.00m/epoch)
```

### Test Result
From the output of epoch-1000, we can compute the accuracy of the model, it has accuracy of 100%.

### Supplimentary
Readers can use this tutorial to get a general picture of a deep learning model.
Github: [thunet-tutorial](https://github.com/ShenDezhou/thunet-tutorial)