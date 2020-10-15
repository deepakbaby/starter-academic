+++
title = "Tracking Multiple Losses with Keras"
date = 2019-03-04T11:59:56+01:00
draft = false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["admin"]

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = ["deep learning", "keras"]
categories = []

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["deep-learning"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
# projects = ["internal-project"]

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
[image]
  # Caption (optional)
  caption = ""

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""
+++

Often we deal with networks that are optimized for multiple losses (e.g., VAE). In such scenarios, it is useful to keep track of each loss independently, for fine-tuning its contribution to the overall loss. This post details an example on how to do this with keras.

Let us look at an example model which needs to trained to minimize the sum of two losses, say mean square error (MSE) and mean absolute error (MAE). Let $\lambda\_{mse}$ be the hyperparameter that controls the contribution of MSE to the toal loss. i.e., the total loss is MAE + $\lambda\_{mse}$ * MSE. This loss can be implemented using:  
```python
import keras.backend as K

lambda_mse = 10 # hyperparameter to be adjusted

def joint_loss (y_true, y_pred):
    # mse
    mse_loss = K.mean(K.square(y_true - y_pred))
    # mae
    mae_loss = K.mean(K.abs(y_true - y_pred))
    return mae_loss + (lambda_mse * mse_loss)
```

with the model compiled as:
```python
model.compile(loss = joint_loss, optimizer='Adam')
```

However, when we run ```model.fit(...)``` keras shows the progress something like this.. 
```
Epoch 1/30
 19488/144615 [===>..........................] - ETA: 1:52:37 - loss: 0.4103
```

Keras shows only the joint loss and does not give the individual MSE and MAE losses which makes it difficult to track how they evolve over epochs and to adjust $\lambda\_{mae}$ accordingly. 

In order to track them, we will need to define individual losses as below. 

```python
import keras.backend as K

lambda_mse = 10 # hyperparameter to be adjusted

def joint_loss (y_true, y_pred):
    # mse
    mse_loss = K.mean(K.square(y_true - y_pred))
    # mae
    mae_loss = K.mean(K.abs(y_true - y_pred))
    return mae_loss + (lambda_mse * mse_loss)

def mse_loss (y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

def mae_loss (y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))
```

Then we can use the ```metrics``` parameter in the ```model.compile``` to also track the MAE and MSE. This can be done by compiling the model using

```python
model.compile(loss = joint_loss, optimizer='Adam', metrics=[mse_loss, mae_loss])
```

Notice that the model is still compiled to optimize for the joint loss, but it also returns the MAE and MSE losses. Executing ```model.metrics_names``` will return three values, ```['loss', 'mae_loss', 'mse_loss']```.  Now the ```model.fit(...)``` will show something like this

```
Epoch 1/30
 26336/144615 [====>.........................] - ETA: 1:46:54 - loss: 0.4078 - mae_loss: 0.1891 - mse_loss: 0.0219
```

Now we can see the joint loss and the individual losses that contributed to it. We can also verify that the joint loss indeed is mae\_loss + 10 * mse_loss, where 10 was the value chosen for $\lambda\_{mse}$.

Similiarly, you can define your own loss terms and use the ```metrics``` parameter in ```model.compile``` to track them independently.
