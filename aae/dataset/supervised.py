import numpy as np
from chainer import cuda
from ..utils import onehot

class Dataset():
	def __init__(self, train, test):
		self.images_train, self.labels_train = train
		self.images_test, self.labels_test = test
		self.dataset_indices = np.arange(0, len(self.images_train))
		self.shuffle()

	def sample_minibatch(self, batchsize, gpu=True):
		batch_indices = self.dataset_indices[:batchsize]
		x_batch = self.images_train[batch_indices]
		y_batch = self.labels_train[batch_indices]
		self.dataset_indices = np.roll(self.dataset_indices, batchsize)
		y_onehot_batch = onehot(y_batch)
		
		if gpu:
			x_batch = cuda.to_gpu(x_batch)
			y_batch = cuda.to_gpu(y_batch)
			y_onehot_batch = cuda.to_gpu(y_onehot_batch)

		return x_batch, y_batch, y_onehot_batch

	def shuffle(self):
		np.random.shuffle(self.dataset_indices)