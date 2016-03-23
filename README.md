# Adversarial Autoencoder

This is the [Chainer](http://chainer.org/) implementation of [Adversarial Autoencoder [arXiv:1511.05644]](http://arxiv.org/pdf/1511.05644v1.pdf)

[この記事](http://musyoku.github.io/2016/02/22/adversarial-autoencoder/)で実装したコードです。

## Requirements

- Chainer 1.6+

## Running

Create "images" directory in the root or use "image_dir" option to specify the directory that contains training images.

Options:
- --image_dir
	- Specify the directory that contains the training images.
- --load_epoch 
	- Specify the model you want to load.

### Class label

Please add the label index (must start 0) to the image filename.

format:	`[0-9]+_.+\.(bmp|png|jpg)`

e.g. MNIST

![example](http://musyoku.github.io/images/post/2016-02-22/class_label_example.png)

### Training

e.g. swiss roll distribution

`cd swiss_roll`

`python train.py`

### Visualizing:

e.g. swiss roll distribution

`cd swiss_roll`

`python visualize.py --load_epoch 10`


## MNIST Unsupervised Learning

### Uniform (-2.0 ~ 2.0)

#### 1,000 train data

![Uniform](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/uniform_train_z.png?raw=true)

#### 9,000 test data

![Uniform](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/uniform_test_z.png?raw=true)

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
