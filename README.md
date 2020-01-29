# tf-text2image
 
| **Work In Progress** |
| --- |

This project uses a neural network to generate images from text using TensorFlow 2.

The text is composed of two parts:

* *chars*: The characters to be displayed in the image
* *spec*: A specification for the format of the text. For instance the color (so far), font, policy etc.

The data is generated synthetically by sampling the chars and specs randomly and plotting the image using matplotlib.

The advantage of synthetically generated data is that we can gradually increase the difficulty of the task as the model gets better at it. The ultimate goal is to use a Generative Adversarial Network (GAN).

# Part 1: Training a generator directly

Since our data does not contain any noise ie the mapping from text to images is deterministic, we can train a neural network in a supervised fashion.

Our generator has a simple structure. First, two distinct RNNs read in the variable length text inputs (`chars` and `spec`) to transform them into a fixed length encoding.

This encoding is then fed through a full-collected layer to add a non-linear transformation on the one hand and to increase its size.

Finally, the encoding initializes a feature-map that is up-sampled via fractional-strides convolutions to return an image of size 128x128x3.

Since the output of the generator is later fed into the discriminator of a GAN, we would like it to be centered around 0. Therefore, we use `tanh` as the activation of the last convolution.

## A loss function for `tanh`

In what follows, we propose a loss function tailored to the `tanh` activation. Its gradient possesses the same desirable properties as when using the `sigmoid` in combination with the binary cross-entropy.

Note that since the `tanh` can be rewritten using the `sigmoid` function:

<img src="https://latex.codecogs.com/gif.latex?\tanh(h)=\frac{e^{&plus;h}-e^{-h}}{e^{&plus;h}&plus;e^{-h}}=\frac{1-e^{-2h}}{1&plus;e^{-2h}}=\frac{2-(1&plus;e^{-2h})}{1&plus;e^{-2h}}=2\sigma(2x)-1" title="\tanh(h)=\frac{e^{+h}-e^{-h}}{e^{+h}+e^{-h}}=\frac{1-e^{-2h}}{1+e^{-2h}}=\frac{2-(1+e^{-2h})}{1+e^{-2h}}=2\sigma(2x)-1" />

It would have been more convenient - and equivalent up to a constant - to use the sigmoid function and normalize that data as part of the discriminator model. Yet, since this is a personal project, let's have some fun with math :-)

By analogy of the binary cross-entropy applied to the `sigmoid`, we suggest the following loss function:

<img src="https://latex.codecogs.com/gif.latex?L=-\frac{1}{2}\Big[(1&plus;y)\log(1&plus;p)&plus;(1-y)\log(1-p)\Big]" title="L=-\frac{1}{2}\Big[(1+y)\log(1+p)+(1-y)\log(1-p)\Big]" />

Where *`y`* represents the ground truth value of the pixels, and *`p=tanh(h)`* is the prediction of the network, with *`h`* representing the output of the last hidden layer ie. the logits.

### Gradient properties

We show next that this loss function gives desirable properties for the gradient, when used with `tanh`.

Let's compute the gradient of the loss *`L`* with respect to the logits *`h`*:

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;h}=-\frac{1}{2}\left[\frac{1&plus;y}{1&plus;p}\frac{\partial&space;p}{\partial&space;h}-\frac{1-y}{1-p}\frac{\partial&space;p}{\partial&space;h}\right]" title="\frac{\partial L}{\partial h}=-\frac{1}{2}\left[\frac{1+y}{1+p}\frac{\partial p}{\partial h}-\frac{1-y}{1-p}\frac{\partial p}{\partial h}\right]" />

The derivative of the activations *`p`* with respect to the logits *`h`* takes a simple form:

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;p}{\partial&space;h}=\frac{1}{\coth^2(h)}=1-\tanh^2(h)=1-p^2=(1-p).(1&plus;p)" title="\frac{\partial p}{\partial h}=\frac{1}{\coth^2(h)}=1-\tanh^2(h)=1-p^2=(1-p).(1+p)" />

Using this in the equation just above, shows that the gradient propagated through the activations is directly proportional to the difference between activations and targets (ie no vanishing nor exploding gradients):

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;h}=-\frac{1}{2}\Big[(1&plus;y)(1-p)-(1-y)(1&plus;p)\Big]=p-y" title="\frac{\partial L}{\partial h}=-\frac{1}{2}\Big[(1+y)(1-p)-(1-y)(1+p)\Big]=p-y" />

### Numerically stable version

Inspiring from [tensorflow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_impl.py#L124), we develop numerically stable version of the loss *`L`* from the logits *`h`*:

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;L&=-\frac{1}{2}\left[&space;(1&plus;y)\log\left(\frac{(1&plus;e^{-2h})&plus;(1-e^{-2h})}{1&plus;e^{-2h}}\right)&space;&plus;(1-y)\log\left(\frac{(1&plus;e^{-2h})-(1-e^{-2h})}{1&plus;e^{-2h}}\right)&space;\right]&space;\\&space;&=-\frac{1}{2}\Big[&space;(1&plus;y)\left[\log(2)-\log(1&plus;e^{-2h})\right]&space;&plus;(1-y)\left[\log(2)-2h-\log(1&plus;e^{-2h})\right]&space;\Big]&space;\\&space;&=\log(1&plus;e^{-2h})&plus;h-yh-\log(2)&space;\end{align*}" title="\begin{align*} L&=-\frac{1}{2}\left[ (1+y)\log\left(\frac{(1+e^{-2h})+(1-e^{-2h})}{1+e^{-2h}}\right) +(1-y)\log\left(\frac{(1+e^{-2h})-(1-e^{-2h})}{1+e^{-2h}}\right) \right] \\ &=-\frac{1}{2}\Big[ (1+y)\left[\log(2)-\log(1+e^{-2h})\right] +(1-y)\left[\log(2)-2h-\log(1+e^{-2h})\right] \Big] \\ &=\log(1+e^{-2h})+h-yh-\log(2) \end{align*}" />

To avoid an exponent of a positive number, which might be too large, let's rewrite this for the case where *`h<0`*:

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;L&=\log(1-e^{-2h})&plus;\Big(\log(e^{2h})-\log(e^{2h})\Big)&plus;h-yh-\log(2)&space;\\&space;&=\log(e^{2h}&plus;1)-h-yh-\log(2)&space;\end{align*}" title="\begin{align*} L&=\log(1-e^{-2h})+\Big(\log(e^{2h})-\log(e^{2h})\Big)+h-yh-\log(2) \\ &=\log(e^{2h}+1)-h-yh-\log(2) \end{align*}" />

Both cases *`h<0`* and *`h>0`* can be rewritten as follows, where we drop the constant:

<img src="https://latex.codecogs.com/gif.latex?L&=\log(1-e^{-2|h|})&plus;|h|-yh" title="L&=\log(1-e^{-2|h|})+|h|-yh" />

## Results

### Generator

We train the generator for 500 epochs (each epoch is made of 64 batches of size 16) and get the following convergence.

<img src="media/G_train.png" alt="G_train" width="300">

Since our data is generated synthetically, it covers the whole input space ie it contains all possible combinations of `chars`and `spec`.
We can therefore safely evaluate the model on the training data (we verify indeed that the metrics on a separate validation set are indistinguishable from the training set).

A sample of generated image can be seen below:

<p float="center">
<img src="media/evaluation_1.png" alt="evaluation_1" width="300">
<img src="media/evaluation_2.png" alt="evaluation_2" width="300">
<img src="media/evaluation_3.png" alt="evaluation_3" width="300">
</p>

### Discriminator

We also train a discriminator using the converged generator. We observe that near 100% accuracy can be achieved after only a few epochs:

<img src="media/D_train.png" alt="D_train" width="300">
