## Adversarial AutoEncoder

This is the Chainer implementation of [Adversarial Autoencoder [arXiv:1511.05644]](http://arxiv.org/abs/1511.05644)

[この記事](http://musyoku.github.io/2016/02/22/adversarial-autoencoder/)と[この記事](http://musyoku.github.io/2016/08/09/Adversarial-AutoeEncoder%E3%81%A7%E5%8D%8A%E6%95%99%E5%B8%AB%E3%81%82%E3%82%8A%E5%AD%A6%E7%BF%92/)で実装したコードです。

論文で報告されている結果があまり出ていないためコードに不具合があるかもしれません。

See also:[Variational AutoEncoder](https://github.com/musyoku/variational-autoencoder)
See also:[Auxiliary Deep Generative Model](https://github.com/musyoku/adgm)

### Requirements

- Chainer 1.8+
- Pillow
- Pylab
- matplotlib.patches
- pandas

#### Download MNIST

run `mnist-tools.py` to download and extract MNIST.

#### How to label your own dataset 

You can provide label information by filename.

format:

`{label_id}_{unique_filename}.{extension}`

regex:

`([0-9]+)_.+\.(bmp|png|jpg)`

e.g. MNIST

![labeling](http://musyoku.github.io/images/post/2016-07-02/labeling.png)

## Adversarial Regularization

run `upervised/regularize_z/train.py`

![result](http://musyoku.github.io/images/post/2016-08-09/supervised/regularize_z/labeled_z_10_gaussian.png)

![result](http://musyoku.github.io/images/post/2016-08-09/supervised/regularize_z/labeled_z_swiss_roll.png)

## Supervised Adversarial Autoencoders

run `supervised/learn_style/train.py`

![result](http://musyoku.github.io/images/post/2016-08-09/supervised/learn_style/analogy.png)

## Semi-Supervised Adversarial Autoencoders

run `semi-supervised/classification/train.py`

### 100 labeled data and 49,900 unlabeled data

![result](http://musyoku.github.io/images/post/2016-08-09/semi_supervised.png)

## Unsupervised Clustering with Adversarial Autoencoders

run `unsupervised/clustering/train.py`

### 16 clusters

![result](http://musyoku.github.io/images/post/2016-08-09/unsupervised/clustering/clusters_16.png)

### 32 clusters

![result](http://musyoku.github.io/images/post/2016-08-09/unsupervised/clustering/clusters_32.png)

I think I need more training time.

## Dimensionality Reduction with Adversarial Autoencoders

run `semi-supervised/dim_reduction/train.py`

or

run `unsupervised/dim_reduction/train.py`

### 100 labeled

![result](http://musyoku.github.io/images/post/2016-08-09/semi-supervised/dim_reduction/labeled_z_100.png)

### 1000 labeled

![result](http://musyoku.github.io/images/post/2016-08-09/semi-supervised/dim_reduction/labeled_z_1000.png)

### unsupervised 20 clusters

![result](http://musyoku.github.io/images/post/2016-08-09/unsupervised/dim_reduction/labeled_z.png)