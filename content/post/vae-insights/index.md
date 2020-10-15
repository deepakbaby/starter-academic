+++
title = "Implementing Variational Autoencoders: Insights and some tricks"
date = 2019-07-03T15:53:58+02:00
draft = false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = []

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = []
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

This post is a summary of some of the main hurdles I encountered in implementing a VAE on a custom dataset and the tricks I used to solve them. The keras code snippets are also provided. Understanding VAEs and its basic implementation in Keras can be found in the [previous post](https://deepakbaby.github.io/post/vae-keras/).

## Posterior collapse in VAEs
The Goal of VAE is to train a generative model $\mathbb{P}(\mathbf{X}, z)$ to maximize the marginal likelihood $\mathbb{\mathbf{X}}$ of the dataset. The cost function used in training a VAE is comprised of a reconstruction loss and a KL loss as given below.

$$\begin{equation}
\mathcal{L} = -\mathbb{E}\left\[ \log\mathbb{P}(\mathbf{X} \vert z)  \right\] + D\left(~\mathbb{Q}(z \vert \mathbf{X})~ \Vert~ \mathbb{P}(z)   ~\right)
\end{equation}$$

The main implementation issue in this case is that the two losses are kind of opposing each other. The problem of mode collapse is that the second loss term $D\left(~\mathbb{Q}(z \vert \mathbf{X})~ \Vert~ \mathbb{P}(z)   ~\right)$ reduces to $0$. i.e., the approximate posterior $\mathbb{Q}(z \vert \mathbf{X})$ becomes equal to the prior $\mathbb{P}(z)$. Thus the latent variable do not carry any information about the input $\mathbf{X}$.

In addition, there is always a mismatch between the dimensions of the data and the latent space. If our data is $N$ dimensional and the latent space has a dimension of $D$, the first cost term involves summation over $N$ values and the KL loss is a summation over $D$ values. This scaling difference introduces additional weightage on one loss term over the other and it converges faster than the other.

If the reconstruction loss converges faster, it leads to the latent space not learning any meaningful representations. On the other hand, if the KL loss converges faster, it leads decoder generating meaniningless outputs. So there is always this problem of balancing these two losses. After some reading, I came across the following three approaches to mitigate this cost balancing problem. The solutions are given in Keras terms.

```
import keras.backend as K
```
## Using `K.sum` instead of `K.mean`
Many standard implementations (for example, [Keras VAE tutorial](https://keras.io/examples/variational_autoencoder/)) either use `K.sum` instead of `K.mean` or if you are using a standard loss term such as `mse` scale it by $N$ which is the data dimension.

```python
import keras.backend as K
from keras.losses import mse

N=x_train.shape[1] # dimension of the data

def vae_loss(y_true, y_pred):
    # mse loss
    reconstruction_loss = mse(y_true, y_pred)
    reconstruction_loss *= N
    # kl loss
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(reconstruction_loss + kl_loss)
```

OR

```python
import keras.backend as K

def vae_loss(y_true, y_pred):
    # mse loss
    reconstruction_loss = K.sum(K.square(y_true - y_pred), axis=-1)
    # kl loss
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(reconstruction_loss + kl_loss)
```

However, this does not always solve the problem. There will be scaling mismatches introduced by $N$ and $D$, and what happened in my experiment was the KL loss converged faster.

## KL annealing 

This is a more popular approach and it worked in my case too. This approach is to first train the VAE using the reconstruction loss only for a few epochs and then slowly introduce the KL loss term. This approach works better due to the fact that the KL cost is initially very large and the optimizer will focus on the KL loss only which leads to a local minimum. 

So what I did was to train the VAE with reconstruction loss only by scaling the KL loss by 0 for a few epochs and then gradually increase the scaling on KL loss from 0 to 1 over the next few epochs and let it train using the actual VAE loss for the remaining epochs. I used a callback for updating the weight on KL loss. Assuming the VAE model in keras is compiled as `vae`. 

In the following snippet the VAE is trained only on reconstruction loss for the first 40 epochs and then the KL loss scale is increased from 0 to 1 linearly over the next 20 epochs.

```python
import keras.backend as K
from keras.callbacks import Callback

# total number of epochs
n_epochs = 500 
# The number of epochs at which KL loss should be included
klstart = 40
# number of epochs over which KL scaling is increased from 0 to 1
kl_annealtime = 20

class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight
    def on_epoch_end (self, epoch, logs={}):
        if epoch > klstart :
            new_weight = min(K.get_value(self.weight) + (1./ annealtime), 1.)
            K.set_value(self.weight, new_weight)
        print ("Current KL Weight is " + str(K.get_value(self.weight)))


# the starting value of weight is 0
# define it as a keras backend variable
weight = K.variable(0.)
# wrap the loss as a function of weight
def vae_loss(weight):
    def loss (y_true, y_pred):
        # mse loss
        reconstruction_loss = K.sum(K.square(y_true - y_pred), axis=-1)
        # kl loss
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return reconstruction_loss + (weight * kl_loss)
    return loss

# compile vae with the weighted vae loss
vae.compile(optimizer='adam', loss=vae_loss(weight))

# train VAE with annealing callback
vae.fit(X_train, X_train, epochs=n_epochs,
        callbacks=[AnnealingCallback(weight)])
   
```

This will train the VAE with the new weight scheduling on the KL loss. This lets the network to first learn to reconstruct the data and gradually learn how the latent space is distributed.
