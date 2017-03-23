# Deep-Learning-Resources
Organized list of deep learning resources

* Reading List
* Projects
* Datasets

### Deep Learning Reading List
A reading list for getting into deep learning, as well as papers I find interesting.

The sections are as follows:

* Introduction Papers
* Convolutional Neural Network (CNN) Applications
* Generative Adversarial Networks (GANs)
* Reinforcement Learning
* Recurrent Neural Networks and Long Short Term Memory Networks (RNNs and LSTMs)
* Miscellaneous
* Notes

___

### Introduction Papers
These are some good intro papers for getting into and understanding deep learning.

[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

A bit long, but gives a good intro for CNNs in general. It contains the architecture for Le-Net which is
used for classifying MNIST.
___

[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

This is the paper that kicked off the deep learning hype. The architecture used, "AlexNet", is used for
classifying images on ImageNet.
___

[A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/abs/1603.07285)

This paper gives good illustrations for what is going on during convolutions, transpose convolutions 
(sometimes known as deconvolutions), max pooling, and other techniques used.

___

### CNN Applications
Applications of CNNs that go beyond classification (classification included)


[Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

This paper introduces VGG Net, which is another popular architecture used for classification.

___

[Action-Conditional Video Prediction using Deep Networks in Atari Games](https://arxiv.org/abs/1507.08750)

This paper shows how to predict the next frames in Atari games. They also provide a very good way of combining
multiple streams of information.

___

[Colorful Image Colorization](http://richzhang.github.io/colorization/)

Colorizing black and white images.

___

[Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)

Recovering a high resolution from a low resolution input.

___

[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network]([Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network)

Super resolution done in realtime.

___

[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

The Inception architecture.

___

[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

Furthur improvements on Inception and Inception-ResNet

___

[Multi-Scale Context Aggregation By Dilated Convolutions](https://arxiv.org/abs/1511.07122)

Semantic segmentation using Dilated Convolutions.

___

[Network In Network](https://arxiv.org/abs/1312.4400)

A novel deep network that builds micro neural networks within the receptive fields of convolutions.

___

[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Introduction to ease the training of deeper networks.

___

[Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)

Exploring the activations within convolutions to show CNNs have localization ability.

___

[Improving Image Classification with Location Context](https://arxiv.org/abs/1505.03873)

Improving image classification by using location context.

___

[The Game Imitation: A Portable Deep Learning Model for Modern Gaming AI](http://cs231n.stanford.edu/reports2016/113_Report.pdf)

A convolutional model used to train a gaming AI.
___

### Generative Adversarial Networks
Generative Adversarial Networks (GANs) are really cool. They are a newish (2014) technique for generating
data.

[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

This is the first GANs paper which lays out the foundation for all of the following papers on the subject.

___


[Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)

GANs are notorious for being difficult to train. This gives some insights on making them easier to
train.

___

[Conditional Adversarial Networks](https://arxiv.org/abs/1411.1784)

Conditioning GANs on some extra information, such that the generation process can be directed.

___

[Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

Bridging the gap between deep learning with CNNs and GANs.

___

[Invertible Conditional GANs for image editing](https://arxiv.org/abs/1611.06355)

Showing how the use of encoders can inverse the mapping of a conditional GAN, allowing the editing of images.

___

[Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300)

An autoencoder network that leverages learned representations.

___

[NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)

This is a summary of the tutorial presented at NIPS by Ian Goodfellow (who wrote the first GANs paper).
Because these are generally difficult to train, this is very useful.

___

### Reinforcement Learning

[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

A great paper showing how Deep Networks can be used with Q-Learning for Reinforcement Learning.

___

[Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)

This paper shows a fault in the paper above as well as a solution using multiple networks.

___

[Dynamic Frame skip Deep Q Network](https://arxiv.org/abs/1605.05365)

This paper treats the number of frames to skip when training as a learnable parameter.

___

[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

### Recurrent Neural Networks and Long Short Term Memory Networks

[DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/abs/1502.04623)

Generating images using a RNN in a human-style fashion.

___

### Miscellaneous

[Tensorflow: A system for large-scale machine learning](https://arxiv.org/abs/1605.08695)

Introduction to the Tensorflow Deep Learning Framework.

___

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

Autoencoder using a variational loss function.

___

[Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)

A tutorial for variational autoencoders.

___

### Other Notes

Noteable architectures:
* Le-Net
* Alexnet
* VGG
* Inception V1/V2/V3/V4
* Inception-ResNet-V2
* ResNet

More can be found [here](https://github.com/tensorflow/models/tree/master/slim)

___

[GA3C: GPU-based A3C for Deep Reinforcement Learning](https://arxiv.org/pdf/1611.06256.pdf)

___

[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)

### Projects

[TensorKart - Playing Mario Kart 64 with Tensorflow](https://github.com/kevinhughes27/TensorKart)

___

[Simple Reinforcement Learning with Tensorflow](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2#.i814bqyv1)
___

[Top Deep Learning Projects](https://github.com/hunkim/DeepLearningStars)

___

[Wasserstein GAN in Tensorflow](https://github.com/martinarjovsky/WassersteinGAN)

___

[DCGANs in Tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)

___

[EBGANs in Tensorflow](https://github.com/shekkizh/EBGAN.tensorflow)

___

[Energy Based GANs NIPS 2016 Workshop Slides](https://drive.google.com/file/d/0BxKBnD5y2M8NbzBUbXRwUDBZOVU/view)

___

[How to train a GAN?](https://github.com/soumith/ganhacks)

___

[GA3C: Reinforcement Learning through Asynchronous Advantage Actor-Critic on a GPU](https://github.com/NVlabs/GA3C)

___

### Data

[Street Map Plugin for Unreal Engine 4](https://github.com/ue4plugins/StreetMap)

___

[Open Source tool for RF reverse engineering](https://github.com/paulgclark/waveconverter)

___


[Python API Wrappers](https://github.com/realpython/list-of-python-api-wrappers/blob/master/readme.md)

