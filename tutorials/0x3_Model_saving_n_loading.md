# What is THUNET?
A deep learning net/framework named "TsingHua University NET", short for "THUNET", is for non-commercial, educational, scientific purpose for the deep learning community.

# How to build a neural network with THUNET?
Next, I will explain how to use THUNET to save and load a model. Models are saved in 7z format, thus gain higher compression rate than zip format by 10%+.[Zip / 7zip Compression Differences](https://stackoverflow.com/questions/21992551/zip-7zip-compression-differences)

## Tutorial-3: Model Saving and Loading
Model serialization makes use of the 7z format instead of the legacy zip format for higher compression rate.

## Referred from Wiki article on comparison of zip and 7z
`In 2011, TopTenReviews found that the 7z compression was at least 17% better than ZIP,[15] and 7-Zip's own site has since 2002 reported that while compression ratio results are very dependent upon the data used for the tests, "Usually, 7-Zip compresses to 7z format 30–70% better than to zip format, and 7-Zip compresses to zip format 2–10% better than most other zip-compatible programs.[16]"`

## What You Can Learn?
In this tutorial, I am telling how to build a model, save it, load it, which is a fundamental operation.
After finishing the lesson, readers should be able to create Activation, Loss, Layer, Scheduler and Optimizer objects, which are key components of the deep learning network.
Readers will learn how to prepare data and train the model.
And in the end, readers have a clear understanding that the model layers' Forward, Backward and Update process.

### Preparation
We should make a python environment ready. According to [THUNET's guideline](https://pypi.org/project/thunet), the following python versions are required: 2.7, 3.5, 3.6, 3.7, 3.8, 3.9, or 3.10.

### Python Packages Install
Install the package by pip command:
`pip install thunet`

### Model Saving
from thunet.neural_nets.utils.disks import save, load

model = {
    "fc_layer1": fc_layer1,
    "fc_layer2": fc_layer2,
    "sm_layer": sm_layer
}

save(model, "../model_lake/01_xor_model.thu")

### Model Loading
from thunet.neural_nets.utils.disks import save, load

model = load("../model_lake/01_xor_model.thu")
fc_layer1 = model["fc_layer1"]
fc_layer2 = model["fc_layer2"]
sm_layer = model["sm_layer"]


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
By saving the trained model, users are able to share funny models with the community.

### Supplimentary
Readers can use this tutorial to get a general picture of a deep learning model.
Github: [thunet-tutorial](https://github.com/ShenDezhou/thunet-tutorial)