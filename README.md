## Adversarial AutoEncoder

- Code for the [paper](https://arxiv.org/abs/1511.05644)

### Requirements

- Chainer 2+
- Python 2 or 3

## Incorporating Label Information in the Adversarial Regularization

run `semi-supervised/regularize_z/train.py`

We trained with a prior (a mixture of 10 2-D Gaussians or Swissroll distribution) on 10K labeled MNIST examples and 40K unlabeled MNIST examples.

![gaussian](http://musyoku.github.io/images/post/2016-02-22/gaussian.png)

![swissroll](http://musyoku.github.io/images/post/2016-02-22/swissroll.png)

## Supervised Adversarial Autoencoders

run `supervised/learn_style/train.py`

![analogy](https://github.com/musyoku/musyoku.github.io/blob/master/images/post/2016-02-22/analogy_supervised.png?raw=true)

## Semi-Supervised Adversarial Autoencoders

run `semi-supervised/classification/train.py`

| data | # |
|:--:|:--:|
| labeled | 100 |
| unlabeled | 49900 |
| validation | 10000 |

#### Validation accuracy at each epoch

![classification](https://github.com/musyoku/musyoku.github.io/blob/master/images/post/2016-02-22/classification.png?raw=true)

#### Analogies

![analogy_semi](https://github.com/musyoku/musyoku.github.io/blob/master/images/post/2016-02-22/analogy_semi.png?raw=true)

## Unsupervised clustering

run `unsupervised/clustering/train.py`

#### 16 clusters

![clusters_16](https://github.com/musyoku/musyoku.github.io/blob/master/images/post/2016-02-22/clusters_16.png?raw=true)

#### 32 clusters

![clusters_32](https://github.com/musyoku/musyoku.github.io/blob/master/images/post/2016-02-22/clusters_32.png?raw=true)

## Dimensionality reduction

run `unsupervised/dim_reduction/train.py`

![reduction_unsupervised](https://github.com/musyoku/musyoku.github.io/blob/master/images/post/2016-02-22/reduction_unsupervised.png?raw=true)

run `semi-supervised/dim_reduction/train.py`

![reduction_100](https://github.com/musyoku/musyoku.github.io/blob/master/images/post/2016-02-22/reduction_100.png?raw=true)
