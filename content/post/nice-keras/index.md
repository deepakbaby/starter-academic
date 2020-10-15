+++
title = "NICE- Non-linear Independent Components Estimation: Insights and Implementation in Keras"
date = 2019-12-03T12:55:51+01:00
draft = false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["admin"]

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

Keras implementation can be found [here](https://deepakbaby.github.io/post/nice-keras/).

Flow-based deep generative models have not gained much attention in the research community when compared to [GANs](https://arxiv.org/abs/1406.2661) or [VAEs](https://arxiv.org/abs/1312.6114). This post discusses a flow-based model called [NICE](https://arxiv.org/abs/1410.8516), its advantages over the other generative models and finally an implementation in Keras.

While VAEs use an encoder that finds only an approximation of the latent variable corresponding to a datapoint, GANs doesnt even have an encoder to infer latents. In flow-based models, the latent variables can be infered exactly without any approximation. Flow-based models make use of reversible architecture (which will be explained below) which enables accurate inference, in addition to providing optimization over the exact log-likelihood of the data instead of a lower bound of it.

### Flow-based generative models
In flow-based approaches the generative process for a datapoint $ \mathbf{x} $ is defined as:

$$\begin{equation}
\mathbf{z} \sim p (\mathbf{z}) \\\\\\ 
\mathbf{x} = \mathbf{g}(\mathbf{z})
\end{equation}$$

where $\mathbf{z}$ is the latent variable with some tractable density $p(\mathbf{z})$ such as standard normal: $p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})$. The key aspect in flow-based approaches is that the function $\mathbf{g}$ is an invertible function (also called _bijective_) such that for every datapoint $\mathbf{x}$, the latent variable can be inferred by $\mathbf{z} = \mathbf{f}(\mathbf{x})=\mathbf{g}^{-1}(\mathbf{x})$. 

The function $\mathbf{f}$ which maps the datapoint to the corresponding latent-variable is realised using a deep neural network. In addition, notice that $\mathbf{g}$ is the inverse of $\mathbf{f}$, therefore the neural network architecture should be carefully defined such that it is reversible or invertible. Therefore, we focus on functions where $\mathbf{f}$ (and $\mathbf
{g}$) is composed of a sequence of invertible transformations: $\mathbf{f} = \mathbf{f_1} \circ \mathbf{f_2} \circ \dots \circ \mathbf{f_K}$, such that the mapping from $\mathbf{x}$ to $\mathbf{z}$ can be written as: 
$$\begin{equation}
\mathbf{x}\stackrel{\mathbf{f_1}}{\longleftrightarrow} \mathbf{h_1} \stackrel{\mathrm{\mathbf{f_1}}}{\longleftrightarrow} \mathbf{h_2} \dots \stackrel{\mathrm{\mathbf{f_K}}}{\longleftrightarrow} \mathbf{z}.
\end{equation}$$
Such a sequence of invertible transformations is also called normalizing flow.

Given an observed data variable $\mathbf{x} \in \mathcal{X}$, a simple probability distribution $p(\mathbf{z})$ with $\mathbf{z} \in \mathcal{Z}$, and a bijective function $\mathbf{f} : \mathcal{X} \rightarrow \mathcal{Z}$ (with $\mathbf{g} = \mathbf{f}^{-1}$), the change of variable formula defines a model distribution on $\mathcal{X}$ by
$$\begin{align}
p(\mathbf{x}) &= p(\mathbf{z})\left\vert \det \left( \dfrac{\partial \mathbf{z}}{\partial \mathbf{x}}  \right)  \right\vert \\\\\\
\log p(\mathbf{x}) &= \log p(\mathbf{z}) + \log \left\vert \det \left( \dfrac{\partial \mathbf{z}}{\partial \mathbf{x}}  \right)  \right\vert \\\\\\
&=  \log p(\mathbf{z}) + \sum_{i=1, j=i-1}^{K} \log \left\vert \det \left( \dfrac{{\partial \mathbf{h_i}}}{{\partial \mathbf{h_j}}}  \right)  \right\vert
\end{align}$$
with $\mathbf{h_0} = \mathbf{x}$ and $\mathbf{h_K} = \mathbf{z}$.

The determinant of the Jacobian matrix $\partial \mathbf{h_i} / \partial \mathbf{h_j}$ is the change in density when $\mathbf{h_j}$ is transformed to $\mathbf{h_i}$ under the transformation $\mathbf{f_i}$. Thus, flow-based models require to compute this determinant as well. The second key aspect is to design the functions $\mathbf{f_i}$ such that the determinant of its Jacobian is easy to compute. 

Thus, flow-based models require two important design choices on $\mathbf{f_i}$:

1. Have reversible architecture
1. Design transformations whose determinant of Jacobians are easy to compute

To satisfy these two requirements, the trick is to choose the transformations whose Jacobian is a triangular matrix, such that their determinnant can be simply computed as the product of its diagonal elements. Thus,
$$\begin{equation}
\log \left\vert \det \left( \dfrac{{\partial \mathbf{h_i}}}{{\partial \mathbf{h_j}}}  \right)  \right\vert = \sum \log  \left\vert \text{diag}~ \left( \dfrac{{\partial \mathbf{h_i}}}{{\partial \mathbf{h_j}}}  \right)  \right\vert
\end{equation}$$
where, $\text{diag}(\cdot)$ takes the diagonal of the Jacobian matrix. 

These models are trained (i.e., training the neural nets $\mathbf{f}$) such that the negative log-likelihood of $\mathbf{z}$ is minimized with respect to some prior distribution (more on this below).

To generate data, we can sample from the prior distribution $p(\mathbf{z})$ and do the inverse operation.

### NICE: Non-linear Independent Components Estimation
As mentioned before, the two key main aspects of flow-based approaches is **easy determinant of the Jacobian** and **easy inverse**. In NICE, the input data is split into two blocks  $\mathbf{x} \rightarrow (\mathbf{x_1}, \mathbf{x_2})$ (say, even and odd indices). Then apply block transformation from $(\mathbf{x_1}, \mathbf{x_2})$ to $(\mathbf{h_1^1}, \mathbf{h_1^2})$ of the form:
$$\begin{align}
\mathbf{h_1^1} &= \mathbf{x_1} \\\\\\
\mathbf{h_1^2} &= \mathbf{x_2} + \mathbf{m}(\mathbf{x_1})
\end{align}$$
where, $\mathbf{m}$ is an arbitrarily complex function (neural net). They call this transformation as an *Affine Coupling layer*.  This transformation satisfies the two design choices:

* The inverse can be easily computed as:
$$\begin{align}
\mathbf{x_1} &= \mathbf{h_1^1} \\\\\\
\mathbf{x_2} &= \mathbf{h_1^2} - \mathbf{m}(\mathbf{h_1^1})
\end{align}$$

* The transformation $\mathbf{f}$ is (where $d$ and $e$ are the dimensions of $ \mathbf{x_1}$ and $\mathbf{x_2}$)
$$\begin{align}
\mathbf{h_1} = \begin{bmatrix} \mathbf{h_1^1} \\\\\ \mathbf{h_1^{2}} \end{bmatrix} = \begin{bmatrix} \mathbf{I_d} & 0 \\\\\ \mathbf{m}(\cdot)  & \mathbf{I_e} \end{bmatrix} \begin{bmatrix} \mathbf{x_1} \\\\\ \mathbf{x_2} \end{bmatrix}
\end{align}$$
resulting in a Jacobian matrix 
$$\begin{align}
\dfrac{\partial \mathbf{f}}{\partial \mathbf{x}} &=  \begin{bmatrix} \dfrac{\partial \mathbf{h_1^1}}{\partial \mathbf{x_1}} & \dfrac{\partial \mathbf{h_1^1}}{\partial \mathbf{x_2}} \\\\\\  \dfrac{\partial \mathbf{h_1^2}}{\partial \mathbf{x_1}} & \dfrac{\partial \mathbf{h_1^2}}{\partial \mathbf{x_2}}  \end{bmatrix} \\\\\\
&=  \begin{bmatrix} \mathbf{I_d} & 0 \\\\\ \dfrac{\partial \mathbf{m}(\cdot)}{\partial \mathbf{x_1}}  & \mathbf{I_e}   \end{bmatrix}
\end{align}$$
whose determinant is unity. Notice that such a design not only enables easy compuatation of the determinant, but also lets us choose arbitrarily complex $\mathbf{m}(\cdot)$ since we dont have to compute its derivative to obtain the determinant.

Similarly, inverse operation from $\mathbf{z}$ to $\mathbf{x}$ also results in a unit Jacobian determinant. Thus generating data also is easy with the NICE model.

In the NICE model, since all the transformations are volume preserving (unit Jacobian determinant), the resulting transformation will have equal weight over all dimensions, which is not desirable in practical applications. To address this, NICE also includes a scaling layer at the output that scales every dimension by a trainable weight $S_i$. This allows the model to give more weight on some dimensions and less on others.

Thus the nice criterion becomes maximizing the log-likelihood of the data distribution:
$$\begin{align}
\log p(\mathbf{x}) &= \log p(\mathbf{z}) + \log \left\vert \det \left( \dfrac{\partial \mathbf{z}}{\partial \mathbf{x}}  \right)  \right\vert \\\\\\
&=  \log p(\mathbf{z}) +\sum_{i=1}^D \log (\vert S_i  \vert)
\end{align}$$ 

Further, NICE model assumes that the prior distribution is factorial: $p(\mathbf{z}) = \prod_{i=1}^{D} p(\mathbf{z_i})$. The training criterion for NICE is to maximize its log-likelihood or minimize the negative log-likelihood: $\mathcal{L} = - \log p(\mathbf{x})$

* For standard gaussian:      
$\mathcal{L} = \sum_{i=1}^D 0.5 \cdot (\mathbf{z_i}^2 + \log 2\pi) - \log (\vert S_i  \vert)$
* For standard logistic:    
$\mathcal{L} =\sum_{i=1}^D \log\left(1+\exp(\mathbf{z_i})\right) +  \log(1+\exp(\mathbf{-z_i})) - \log (\vert S_i  \vert) $

As we stack multiple affine coupling layers to obtain more complex transformations. Since the transformation leaves one part of the data leaves unchanged, we can alternate the role of each part in subsequent coupling layers. Typically, 4 coupling layers are used so that all dimensions influence the one another. The scaling layer is paramterised exponentially $\exp(S_i)$ to have positive scaling.

In NICE the forward model maps the datapoint to the latent space and is trained to minimize the negative log-likelihood with respect to some prior distribution. And for inference, we sample from the prior distribution to get $\mathbf{h}$ and new datapoints $\mathbf{x}$ can be generated using the inverse flow. For a 4 layer architecture, the forward and inverse flow equations are,


| Forward Flow  | Inverse Flow |
|:-----------------|:----------------|
| $$\begin{align}\mathbf{h_1^1} &= \mathbf{x_1} \\\\\\ \mathbf{h_1^2} &= \mathbf{x_2} + \mathbf{m_1} ( \mathbf{x_1})  \end{align}$$ | $$\begin{align} \mathbf{h_4} = \exp(-S) \odot \mathbf{h}\end{align}$$ |
|$$\begin{align}\mathbf{h_2^2} &= \mathbf{h_1^2} \\\\\\ \mathbf{h_2^1} &= \mathbf{h_1^1} + \mathbf{m_2} ( \mathbf{h_1^2})  \end{align}$$  | $$\begin{align} \mathbf{h_3^2} &= \mathbf{h_4^2} \\\\\\ \mathbf{h_3^1} &= \mathbf{h_4^1} - \mathbf{m_4} ( \mathbf{h_4^2})  \end{align}$$ |
|$$\begin{align}\mathbf{h_3^1} &= \mathbf{h_2^1} \\\\\\ \mathbf{h_3^2} &= \mathbf{h_2^2} + \mathbf{m_3} ( \mathbf{h_2^1})  \end{align}$$  | $$\begin{align} \mathbf{h_2^1} &= \mathbf{h_3^1} \\\\\\ \mathbf{h_2^2} &= \mathbf{h_3^2} - \mathbf{m_3} ( \mathbf{h_3^1})  \end{align}$$ |
|$$\begin{align}\mathbf{h_4^2} &= \mathbf{h_3^2} \\\\\\ \mathbf{h_4^1} &= \mathbf{h_3^1} + \mathbf{m_4} ( \mathbf{h_3^2})  \end{align}$$  | $$\begin{align} \mathbf{h_1^2} &= \mathbf{h_2^2} \\\\\\ \mathbf{h_1^1} &= \mathbf{h_2^1} - \mathbf{m_2} ( \mathbf{h_2^2})  \end{align}$$ |
| $$\begin{align} \mathbf{h} = \exp(S) \odot \mathbf{h_4}\end{align}$$  |$$\begin{align} \mathbf{x_1} &= \mathbf{h_1^1} \\\\\\ \mathbf{x_2} &= \mathbf{h_1^2} - \mathbf{m_1} ( \mathbf{h_1^1})  \end{align}$$ |

----
### Implementation Notes

As mentioned above, the NICE model is trained to minimize the negative log-likelihood with respect to a standard Gaussian or logistic distribution. 

* **Getting the models correct:** The main design challenge was to get the inverse model correct with its weights tied to the corresponding forward model. We can verify if the inverse model is correct even before training the model. Just take a test image from MNIST and pass it through the forward model. Then use the resulting output as input to the inverse model. If the inverse model is created with correct weights, it should yield the test image.

* **Getting NaNs:** For implementing the logistic loss $\log\left(1+\exp(\mathbf{z_i})\right) +  \log(1+\exp(\mathbf{-z_i}))$, I initially used Keras backend as: 
```
import keras.backend as K     
logistic_negloglikelihood = K.sum ( K.log (1 + K.exp(z)) +  K.log(1 + K.exp(-z)), axis=1 )
```
However, it was resulting in NaN after a few epochs. Then this was replaced with `softplus` function which computes $\log (1 + \exp(x))$  and the NaN issue was (magically!) gone.
```
logistic_negloglikelihood = K.sum ( K.softplus(z) + K.softplus(-z), axis=1 )
```

* **Initialization:** Initially, I was using the default `glorot_uniform`  initialization of keras which was not resulting in the log-likelihoods given in the paper. [Dinh's original implementation](https://github.com/laurent-dinh/nice) in Theano used an initialization from a uniform distribution in $[-0.01,0.01]$. Using the same initialization for the Dense layers resulted in better log-likelihoods.

* **Batch-size:** Dinh used Adam optimizer with a learning rate $0.001$ and batch size of 256. In my experiments, I observed that increasing the batch size from $256$ to $2048$ consistently yielded better log-likelihoods.

* **Clip the outputs during generating data:** MNIST data is rescaled to the range $[0, 1]$ for training. But during inference, the output may not be always in $[0, 1]$. Therefore, we apply clipping on the generated outputs to bring them to the desired range.
