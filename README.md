# tf-text2image
 
| **Work In Progress** |
| --- |

This project uses a neural network to generate images from text using TensorFlow 2.

The text is composed of two parts:

* *chars*: The characters to be displayed in the image
* *spec*: A specification for the format of the text. For instance the color (so far), font, policy etc.

The data is generated synthetically by sampling the chars and specs randomly and plotting the image using matplotlib.

The advantage of synthetically generated data is that we can gradually increase the difficulty of the task as the model gets better at it. The ultimate goal is to use a Generative Adversarial Network (GAN).


### Preliminary results


<p float="center">
<img src="media/evaluation_1.png" alt="evaluation_1" width="300">
<img src="media/evaluation_2.png" alt="evaluation_2" width="300">
<img src="media/evaluation_3.png" alt="evaluation_3" width="300">
</p>