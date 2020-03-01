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

<img src="media/G_train.png" alt="G_train" width="400">

Since our data is generated synthetically, it covers the whole input space ie it contains all possible combinations of `chars` and `spec`.
We can therefore safely evaluate the model on the training data (we verify indeed that the metrics on a separate validation set are indistinguishable from the training set).

A sample of generated image can be seen below:

<p float="center">
<img src="media/evaluation_1.png" alt="evaluation_1" width="250">
<img src="media/evaluation_2.png" alt="evaluation_2" width="250">
<img src="media/evaluation_3.png" alt="evaluation_3" width="250">
</p>

### Discriminator

We also train a discriminator using the converged generator. We observe that near 100% accuracy can be achieved after only a few epochs:

<img src="media/D_train.png" alt="D_train" width="400">

# Part 2: A Generative Adversarial Net

Next, we train a GAN on our synthetic data. The gan is composed of the generator and the discriminator just mentioned.

Note that our GAN differs from a standard gan and could be qualified as "fully-conditional GAN". That is, we only have conditional data as input, no latent variable z. This is similar to [Isola et al. 2016](https://arxiv.org/abs/1611.07004), who generate an image condtional on another image. They can hence use a U-net based generator that uses local featires of the conditinal iage to generate the image. In our case the input is text representation where such an approch is not feasible.

It is probably for that reason that this task turned out to be challenging as the training of the GAN is very unstable (as it is known to be).

Anothe difficulty is extrempy regular images

It is interesting to note that because the link between our text input and our images is deterministic. We can therfore define an exact evaluation metric for our GAN. This is contrary to most literature in the field. We decide to use the mean absolute error (MAE) since it seemed to be a more stable metric than RMSE.

Several improvements over a standard GAN had to be made to give satisfying results:

Beta1=.09* **Adam**: AS is common we use Adam for optimization. It turned out however, that the default values for β did not work. We had to remove the β1, which controls momentum from .9 to .5 and β2, which controls adaptiveness to the gradient norm, from .99 to .9. This indicates the the direction and norm of our gradients change significantmy over the course of the optimization.
LR.0001* **Learning rate**: The litarture advises to use low learnign rates to facilitat covergence. It turned out, that our learning rate of 5-e5 is about one matter of size lower that is common.
* **Minibatch Discrimination**: On problem ouf our and general GAN is mode collaps, where all generated images are almost identical. Once mode collaps occurs, there is no way to escape it. To help the discriminator detect this [Salimans et al. 2016](https://arxiv.org/abs/1606.03498) suggest Minibatch Discrimination (MBD). Thos technique allows the discriminator to look at multiple images from one batch and detect features common between them. Fro this technique to work, fake and real images have to be provided to the discriminator in separate batches (see next point)
* **Separate real & fake batches**: We feed batches os fake and real images separately to the discriminator. Besides simplifying the implementation a little, this has also the effect of accelerating and improving convergence. This is because doing so, effectively increases the learnign rate of the discriminator. Like most public implementation, we use the same parameters when training the discriminator and generator. One would however expect that the generator [since effectively a lot deeper from a gradient perspective] is much harder to train] would require more conservative optimization parameter than the generator. Yet it is known that the generator can only be as good as the discriminator. Allowing the discriminator to converge faster, can thus only be beficial. To our supprise and to our knowledge, using higher learnign rate for the  discriminator (and in general more suite optimization parameters) is not explored in the literature.
DiscOnlySteps* **Discriminator only steps**:
FullVocqb* **Limited vocabulary**: 
TrainRNN* **Pretrained text_rnn**: 
* **Shuffled images**: Our GAN manages quicly to produce images af the right background size and colours. It does not however manage to generate clearly identifiable characters. This is becasue the discriminator is able to distinguis based on othe features such as the smoothness of edges. Ti will never have tp identify individual characters and match them agaisnt the text. Therfore we alse feed the discriminator with real images associated to random text. This forces the discriminator to actually read prperly and provide good gradients to the generator. This method is generalizable to other cGANs and to our knowledge not present in the literature.

To avercome convergence issues, we implement one of the most advanced apprcoahs, [Wasserstain GAN](https://arxiv.org/abs/1701.07875).
We observe that a WGAN does not achieve much faster than usign crossentropy loss. This is because as is the rpomise of this approach, the gradients never saturate, which they initially do otherwise. 

Unfortuanltey, convergence at one moment stagnates completely before the generator is able to recognizeable characters. IT turns out that the cross entropy is able to produce better gradients eventually which seems to be the most challenging part of trainig our GAN. This conclusion is in line with [Lucic et al. 2018](https://arxiv.org/abs/1711.10337), who conclude that many versions of "improved" GANs do effectively not beat the implementation of the original paper.

* **Gradient Penalty**: We fund that imposing the Lipschitz constraint vie weight clipping does prevent converegence. [Gulrajani et al. 2017](https://arxiv.org/abs/1704.00028) impose a soft unit norm constraint on the gradient with respect to the input. Impelementing this was hard because of an [issue in Keras](https://github.com/tensorflow/tensorflow/issues/36436). We therfore had to optimize the constraint in a seperate step [as opposed to a sigle step which optimizes aslo the loss for real and fake images]. This is however anyways required to use with MBD, which is essential for good results.
* **Shared optimizer**: It turned out that optimizing the constrain separately from optimizing the distance measure made traingin g veri sentive to the value of the balancing factor λ. To mitigate this, we use the same optimizer for both steps. The stabilization come from the the adaptive gradients included in Adam, coordiating the gradients between both steps.
* **No momentum**: We use Adam without momentum. This is because the fast convergence from WGAN-GP causes the discriminator to change a lot between steps, and hence varying the direction of gradients making momentum countra productive.



We followed many aprroaches to imrpoving perfmance many of which did not have an impact. We list some of the se below for completeness:
* **Latent variable z**: We attempted feeting a generator with a latent variable as is common in cGANs. The idea is that the randomness coming from z migh facilutae the genrating realisitng images independently from the text initially. Only later the generator might make its output conditional on the text
SaturatingLoss* **Non saturating loss**: At the bigging of traing, GANs suffer form a saturated disrcimintor, since the distribution of the real and fake distributions are very different. Therefore the [original paper](https://arxiv.org/abs/1711.10337) suggested a modified version of the cross entropy to avoid this. We find however that this not required. At least in our case, the difficulty is more to reach full convergence, rather that starting convergence initially.
* **Learning rate decay**:
* **Noise**: As suggested by [https://arxiv.org/abs/1701.04862](Arjovsky et al. 2017) adding noise to real and fake images before feeding them to the discriminator stabilizes training. Keras allows doing this as part of the model. 



