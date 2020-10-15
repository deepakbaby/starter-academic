+++
title = "Understanding Variational Autoencoders and Implementation in Keras"
date = 2019-07-02T16:44:25+02:00
draft = false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["admin"]

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = ["deep learning", "vae", "variational autoencoder", "keras"]
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

Variational Autoencoders (VAEs)[[Kingma, et.al (2013)]](https://arxiv.org/abs/1312.6114) let us design complex generative models of data that can be trained on large datasets. This post is about understanding the VAE concepts, its loss functions and how we can implement it in keras.

## Generating data from a latent space
VAEs, in terms of probabilistic terms, assume that the data-points in a large dataset are generated from a latent space. For e.g., let us assume we want to generate the image of an animal. First we imagine that it has four legs, a head and a tail. This is analogous to the latent space and from this set of characteristics that are defined in the latent space, the model will learn to generate the image of an animal.

Before we dive into the math and intuitions, let us define some notations:

1. $\mathbf{X}$: The type of data we want to generate (say, a large dataset containing images of animals)     
1. $z$: The latent variable, the set of characteristics we want in the image     
1. $\mathbb{P}(\mathbf{X})$: probability distribution of the data     
1. $\mathbb{P}(z)$: probability distribution of the latent space
1. $\mathbb{P}(\mathbf{X} \vert z)$: probability distribution of generating data from the latent variable 

We assume that every data-point $x$ is a random sample from the _unknown underlying process_ whose true distribution $\mathbb{P}(\mathbf{X})$ is unknown.
VAEs make use of a specific probability model that captures the joint probability between the data $\mathbf{X}$ and latent variables $z$. This joint probability can be written as $\mathbb{P}(\mathbf{X}, z) = \mathbb{P}(\mathbf{X} \vert z) \cdot \mathbf{P}(z)$. The generative model assumed in VAE can be described as:    

1. Draw one latent variable $ z_{i} \sim \mathbb{P}(z) $: similar to defining a set of characteristics that defines an animal   
1. Generate the data-point such that $x \sim \mathbb{P}(\mathbf{X} \vert z) $: similar to generating the image of an animal that satisfies the characteristics specified in the latent variable

## VAE formulation and cost function

From the probability model perspective, the latent variables are drawn from a prior $\mathbb{P}(z)$ and the generated data $x$ has a likelihood of $\mathbb{P}(\mathbf{X} \vert z)$ that is conditioned on the latent variables $z$. The objective here is to model the data distribution $\mathbb{P}(\mathbf{X})$ by marginalizing out the latent variable $z$ from the joint-distribution $\mathbb{P}(\mathbf{X}, z)$.

$$\begin{equation}
\mathbb{P}(\mathbf{X}) = \int_{z} \mathbb{P}(\mathbf{X} \vert z) \mathbb{P}(z) ~ dz 
\end{equation}$$

However, this integral is very difficult to compute as it requires to be computed over all possibilities of the latent variable $z$. In order to overcome this, VAEs first try to infer the distribution $\mathbb{P}(z)$ from the data using $\mathbb{P}(z \vert \mathbf{X})$. i.e., rather looking at all possibilities of $z$, we want to infer the distribution of $z$ that describes our data reasonably well. For example, if we want to generate an animal, we only need to specify the characteristics that describe an animal. We do not need to include things like glass, table, ... as it is unlikely that those characteristics contribute to generating the image of an animal. Thus, this inference phase $\mathbb{P}(z \vert \mathbf{X})$ limits our imagination space to focus on characteristics that are required for generating the image of animal.

Again, we do not know the best set of characteristics $\mathbb{P}(z \vert \mathbf{X})$ yet. VAEs make use of variational inference to infer $\mathbb{P}(z \vert \mathbf{X})$. Variational inference approximate the true distribuation $\mathbb{P}(z \vert \mathbf{X})$ using a simpler distribution that is easy to evaluate. A popular choice is Gaussian distribution.

Further, a parametric inference model $\mathbb{Q}(z \vert \mathbf{X})$ that maps the data to the underlying latent space and the difference between $\mathbb{P}(z \vert \mathbf{X})$  and $\mathbb{Q}(z \vert \mathbf{X})$ is quantified using [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between them.

$$\begin{align}
D_{KL} \left(~ \mathbb{Q}(z \vert \mathbf{X})~ \Vert~ \mathbb{P}(z \vert \mathbf{X})~ \right) &= \sum  \mathbb{Q}(z \vert \mathbf{X}) \log \dfrac{ \mathbb{Q}(z \vert \mathbf{X})}{ \mathbb{P}(z \vert \mathbf{X})} \\\\\\
&=\mathbb{E}\left\[ \log \dfrac{ \mathbb{Q}(z \vert \mathbf{X})}{ \mathbb{P}(z \vert \mathbf{X})}  \right\] \\\\\\
&= \mathbb{E}\left\[ \log\mathbb{Q}(z \vert \mathbf{X}) - \log \mathbb{P}(z \vert \mathbf{X})  \right\]
\end{align}$$

where, $\mathbb{E}$ is the expectation with respect to $\mathbb{Q}(z \vert \mathbf{X})$. Using $\mathbb{P}(z \vert \mathbf{X}) = \dfrac{\mathbb{P}(\mathbf{X} \vert z) \mathbb{P}(z)}{\mathbb{P}(\mathbf{X})}$ we can rewrite the above expression as:

$$\begin{align}
D_{KL} \left(~ \mathbb{Q}(z \vert \mathbf{X})~ \Vert~ \mathbb{P}(z \vert \mathbf{X})~ \right) &= \mathbb{E}\left\[ \log\mathbb{Q}(z \vert \mathbf{X}) - \log \dfrac{\mathbb{P}(\mathbf{X} \vert z) \mathbb{P}(z)}{\mathbb{P}(\mathbf{X})}  \right\] \\\\\\
&= \mathbb{E}\left\[ \log\mathbb{Q}(z \vert \mathbf{X}) - \log\mathbb{P}(\mathbf{X} \vert z) - \log \mathbb{P}(z) + \log \mathbb{P}(\mathbf{X}) \right\]
\end{align}$$

Notice that $\mathbb{P}(\mathbf{X})$ does not depend on $z$ and hence it can be taken outside the expectation operation over $z$. We will denote $D_{KL}$ as $D$.
$$\begin{align}
D \left(~ \mathbb{Q}(z \vert \mathbf{X})~ \Vert~ \mathbb{P}(z \vert \mathbf{X})~ \right) &=  \mathbb{E}\left\[ \log\mathbb{Q}(z \vert \mathbf{X}) - \log\mathbb{P}(\mathbf{X} \vert z) - \log \mathbb{P}(z) \right\] + \log \mathbb{P}(\mathbf{X}) \\\\\\
\implies ~~ \log \mathbb{P}(\mathbf{X}) - D \left(~ \mathbb{Q}(z \vert \mathbf{X})~ \Vert~ \mathbb{P}(z \vert \mathbf{X})~ \right) &= \mathbb{E}\left\[ \log\mathbb{P}(\mathbf{X} \vert z)  \right\] - \mathbb{E}\left\[ \log\mathbb{Q}(z \vert \mathbf{X}) - \mathbb{P}(z)  \right\] \\\\\\
&= \mathbb{E}\left\[ \log\mathbb{P}(\mathbf{X} \vert z)  \right\] - D\left(~\mathbb{Q}(z \vert \mathbf{X})~ \Vert~ \mathbb{P}(z)   ~\right)
\end{align}$$

The right-hand side of the above equation is the objective function used by VAEs. What it says is that, we are trying to model our data which is described by $\log \mathbb{P}(\mathbf{X})$ with some error $D \left(~ \mathbb{Q}(z \vert \mathbf{X})~ \Vert~ \mathbb{P}(z \vert \mathbf{X})~ \right)$. Since $D_{KL}$ is always positive, we can write the above equation as:

$$\begin{equation}
\log \mathbb{P}(\mathbf{X}) \geq \mathbb{E}\left\[ \log\mathbb{P}(\mathbf{X} \vert z)  \right\] - D\left(~\mathbb{Q}(z \vert \mathbf{X})~ \Vert~ \mathbb{P}(z)   ~\right)
\end{equation}$$

Thus, the right-hand side (RHS) of the above inequality is the lower bound for $\log \mathbb{P}(\mathbf{X})$ which we are trying to maximize. This is known as the evidence lower bound (ELBO). Maximizing the RHS is also the same as minimizing its negative. The negative of the RHS is therefore used as a cost function to be minimized while training VAEs.

At this point, what we have is:

1. $\mathbb{P}(\mathbf{X} \vert z)$: Generating data from the given latent variable (the **decoder**)    
1. $\mathbb{Q}(z \vert \mathbf{X})$: Infering the latent code given the data (the **encoder**)      
1. $D\left(~\mathbb{Q}(z \vert \mathbf{X})~ \Vert~ \mathbb{P}(z)   ~\right)$: Making sure that the encoded representation resembles a simpler, tractable distribuation (e.g., Gaussian).

Thus a VAE first encodes the data into some latent space (mapping $x$ to $z$) and then generates (decodes: mapping $z$ to $x$) data based on samples from that latent space, and hence called variational autoencoder.

## VAE cost function and neural networks

The VAE cost function can be seen as adding an additional cost term on the traditional autoencoders. The first term is the reconstruction loss at the output, which is the same as used in an autoencoder. The second term forces the encoder to map the input data to a pre-defined tractable distribution.

Why do we need $\mathbb{P}(z)$ to be a simple distribution? Since VAE is a generative model, we would like to generate new data-points by sampling $\mathbb{P}(z)$. The easiest choice for this is a standard normal distribution $\mathcal{N}(0,1)$.

The mappings $\mathbb{P}(\mathbf{X} \vert z)$ and $\mathbb{Q}(z \vert \mathbf{X})$ are realized using deep neural networks (DNNs). Thus VAEs are designed using two DNNs: an encoder and a decoder. The cost function is to minimize the negative of the ELBO obtained above. 

## Implementing VAE cost in keras
As detailed before, the first term of the cost function is the reconstruction loss. We can use any popular loss, say mean-squared error, for this purpose. Computing the KL divergence cost term requires assuming $\mathbb{Q}(z \vert \mathbf{X})$ to be also Gaussian with parameters $\mu (\mathbf{X})$ and $\Sigma (\mathbf{X})$. This assumption enables us to compute the KL divergence between $\mathbb{Q}(z \vert \mathbf{X}) = \mathcal{N}(\mu (\mathbf{X}), \Sigma (\mathbf{X}))$ and $\mathbb{P}(z) = \mathcal{N}(0,1)$ in closed form as:

$$\begin{align}
D\left\[ \mathcal{N}(\mu (\mathbf{X}), \Sigma (\mathbf{X}))~\Vert ~ \mathcal{N}(0,1) \right\] = \dfrac{1}{2} \left\[ tr\left( \Sigma (\mathbf{X}) \right) + \mu (\mathbf{X})^{T} \mu (\mathbf{X}) - k - \log det\left( \Sigma (\mathbf{X}) \right)   \right\]
\end{align}$$
where, $tr$ and $det$ are the trace and determinant of the covariance matrix $\Sigma (\mathbf{X})$ and $k$ is the dimension of the Gaussian distribution. For details on the calculation of the above divergence, refer to this [page](https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians). We also assume that the covariance matrix is diagonal, we can compute the determinant by simpy multiplying its diagonal elements. In addition, we can also implement $\Sigma (\mathbf{X})$ as a vector since it is a diagonal matrix.

$$\begin{align}
D\left\[ \mathcal{N}(\mu (\mathbf{X}), \Sigma (\mathbf{X}))~\Vert ~ \mathcal{N}(0,1) \right\] &= \dfrac{1}{2} \left\[ \sum \Sigma (\mathbf{X}) + \sum \mu^{2} (\mathbf{X}) - \sum 1 - \log \prod \Sigma (\mathbf{X})   \right\] \\\\\\
&= \dfrac{1}{2} \left\[ \sum \Sigma (\mathbf{X}) + \sum \mu^{2} (\mathbf{X}) - \sum 1 - \sum \log \Sigma (\mathbf{X})   \right\] \\\\\\
&=  \dfrac{1}{2} \sum \left\[ \Sigma (\mathbf{X}) + \mu^{2} (\mathbf{X}) -  1 - \log \Sigma (\mathbf{X})   \right\]
\end{align}$$

In addition, typically we model the logarithm of $\Sigma (\mathbf{X})$ for numerical stability. Thus the final loss term becomes:

$$\begin{equation}
D\left\[ \mathcal{N}(\mu (\mathbf{X}), \Sigma (\mathbf{X}))~\Vert ~ \mathcal{N}(0,1) \right\] = \dfrac{1}{2} \sum \left\[ \exp(\Sigma (\mathbf{X})) + \mu^{2} (\mathbf{X}) -  1 - \Sigma (\mathbf{X})   \right\]
\end{equation}$$ 

## Keras implementation

This is mostly a copy of the example provided in [Keras VAE example](https://keras.io/examples/variational_autoencoder/), but with some edits and added comments. This post does not discuss the reparameterization trick involved in training a VAE as it is discussed in many other pages. 

**Even though the example below works really well, in practice, we will need to somehow adjust the reconstruction loss and the KL loss. The insights I gained and the tricks I used to overcome the issues will be described in the upcoming post.**

[Implementing Variational Autoencoders: Some insights and tricks](https://deepakbaby.github.io/post/vae-insights/)

```python
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    models = (encoder, decoder)
    data = (x_test, y_test)

    def vae_loss(y_true, y_pred):
        reconstruction_loss = mse(y_true, y_pred)
        reconstruction_loss *= original_dim
        z_mean = vae.get_layer('encoder').get_layer('z_mean').output
        z_log_var = vae.get_layer('encoder').get_layer('z_log_var').output
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)

    vae.compile(optimizer='adam', loss=vae_loss)
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    # train the autoencoder
    vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))
    vae.save_weights('vae_mlp_mnist.h5')

    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")
```
