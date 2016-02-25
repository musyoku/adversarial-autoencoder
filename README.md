# Adversarial Autoencoder

This is the [Chainer](http://chainer.org/) implementation of [Adversarial Autoencoder [arXiv:1511.05644]](http://arxiv.org/pdf/1511.05644v1.pdf)

[この記事](http://musyoku.github.io/2016/02/22/adversarial-autoencoder/)で実装したコードです。

## MNIST Unsupervised Learning

### Uniform (-2.0 ~ 2.0)

#### 1,000 train data

![Uniform](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/uniform_train_labeled_z.png?raw=true)

#### 9,000 test data

![Uniform](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/uniform_test_labeled_z.png?raw=true)

## MNIST Supervised Learning

### 10 2D-Gaussian Distribution

#### 1,000 train data 

![10 2D-Gaussian](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/10_2d-gaussian_train_labeled_z.png?raw=true)

#### 9,000 test data

![10 2D-Gaussian](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/10_2d-gaussian_test_labeled_z.png?raw=true)

### Swiss Roll Distribution

#### 1,000 train data

![Swiss Roll](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/swiss_roll_train_labeled_z.png?raw=true)

#### 9,000 test data

![Swiss Roll](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/swiss_roll_test_labeled_z.png?raw=true)
