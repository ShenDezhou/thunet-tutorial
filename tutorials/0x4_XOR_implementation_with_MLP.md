# What is THUNET?
A deep learning net/framework named "TsingHua University NET", short for "THUNET", is for non-commercial, educational, scientific purpose for the deep learning community.
Today, I am proud to announce that all the manditory component of the framework, THUNET, have been completed. Students, teachers and scientiests are free and welcome to use this deep learning framework.

# How to build a neural network with THUNET?
Next, I will explain how to use THUNET to build a model to compute XOR operator.
And I will explain the steps in detail: forward, loss, backward grad, backward.

## Tutorial-4: XOR operand, with MLP layers
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
sch = ConstantScheduler(lr=0.0001)

#### Optimizer
fc1_adam = Adam(lr=0.0001, lr_scheduler=sch)
fc2_adam = Adam(lr=0.0001, lr_scheduler=sch)
fc3_adam = Adam(lr=0.0001, lr_scheduler=sch)

#### Neural Layer
fc_layer1 = FullyConnected(8, act_fn=Tanh(), optimizer=fc1_adam)
fc_layer2 = FullyConnected(4, act_fn=Tanh(), optimizer=fc2_adam)
fc_layer3 = FullyConnected(2, act_fn=Tanh(), optimizer=fc3_adam)
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
    "fc_layer1": fc_layer1,
    "fc_layer2": fc_layer2,
    "fc_layer3": fc_layer3,
    "sm_layer": sm_layer
}
model = load("model_lake/03_xor_model_io.thu")
print(model)
fc_layer1 = model["fc_layer1"]
fc_layer2 = model["fc_layer2"]
fc_layer3 = model["fc_layer3"]
sm_layer = model["sm_layer"]

#### Make Batch using minibatch
batch_generator, nb = minibatch(X_train, batch_size, shuffle=True)
for j, b_ix in enumerate(batch_generator):


#### Forward
y_pred_batch = fc_layer1.forward(X_batch)
y_pred_batch = fc_layer2.forward(y_pred_batch)
y_pred_batch = fc_layer3.forward(y_pred_batch)
y_pred_batch = sm_layer.forward(y_pred_batch)

#### Compute the loss
batch_loss = loss_func(y_real_batch, y_pred_batch)

#### Compute the loss grad
y_grad = loss_func.grad(y_real_batch, y_pred_batch)

#### Backward
y_grad = sm_layer.backward(y_grad)
y_grad = fc_layer3.backward(y_grad)
y_grad = fc_layer2.backward(y_grad)
y_grad = fc_layer1.backward(y_grad)

#### Weight Update
sm_layer.update()
fc_layer3.update()
fc_layer2.update()
fc_layer1.update()

#### Trained Model Saving
save(model, "model_lake/03_xor_model_io.thu")


### Training Result
By use model checkpoints, you will be able to get constant results. The saving and loading step in the tutorial save the training time in getting the correct model.
There is a confidential level of 100% to reproduce the result if you use a pretrained correct model. You will constantly get the same correct model using the pretrain-finetune pattern. 
We print the actual value followed by the predicted value.
The model cannot predict exact same as the actual value, but using `argmax` function, we are able to tell the correct class.

The actual value is represented by one-hot vector. Thus, [1, 0] shows it belongs to class-0, and [0, 1] shows that it belongs to class-1.
And the predicted value in epoch-10, [0.516938 0.483062] shows it is class-0, and [0.48051901 0.51948099] shows it is class-1(counting from 1, the first class).
```
[[1. 0.]] [[0.86992463 0.13007537]]
[[0. 1.]] [[0.12770694 0.87229306]]
[[1. 0.]] [[0.87563392 0.12436608]]
[[0. 1.]] [[0.12621451 0.87378549]]
[Epoch 1000] Avg. loss: 0.135927  Delta: 0.000008 (0.00m/epoch)
```

### Test Result
From the output of epoch-1000, we can compute the accuracy of the model, it has accuracy of 100%.

### Supplimentary
Readers can use this tutorial to get a general picture of a deep learning model.
Github: [thunet-tutorial](https://github.com/ShenDezhou/thunet-tutorial)
