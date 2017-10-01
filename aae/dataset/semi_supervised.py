from __future__ import division
import math
import numpy as np
from chainer import cuda
from ..utils import onehot

class Iterator():
	def __init__(self, images, labels, indices, batchsize, using_gpu):
		self.images = images
		self.labels = labels
		self.indices = indices
		self.batchsize = batchsize
		self.using_gpu = using_gpu
		self.pos = 0

	def get_total_iterations(self):
		return math.ceil(len(self.indices) / self.batchsize)

	def __iter__(self):
		return self

	def __next__(self):
		indices = self.indices[self.pos:self.pos + self.batchsize]

		x_batch = self.images[indices]
		y_batch = self.labels[indices]
		
		if self.using_gpu:
			x_batch = cuda.to_gpu(x_batch)
			y_batch = cuda.to_gpu(y_batch)

		batchsize = min(self.batchsize, len(self.indices) -1 - self.pos)
		if batchsize <= 0:
			raise StopIteration()

		self.pos += batchsize

		return x_batch, y_batch

	next = __next__

class Dataset():
	def __init__(self, train, test, num_labeled_data=100, num_classes=10, num_extra_classes=0):
		self.images_train, self.labels_train = train
		self.images_test, self.labels_test = test
		self.num_classes = num_classes
		self.num_extra_classes = num_extra_classes
		indices = np.arange(0, len(self.images_train))
		np.random.shuffle(indices)

		indices_u = []
		indices_l = []
		counts = [0] * num_classes
		num_per_class = num_labeled_data // num_classes

		for index in indices:
			label = self.labels_train[index]
			if counts[label] < num_per_class:
				counts[label] += 1
				indices_l.append(index)
				continue
			indices_u.append(index)

		self.indices_l = np.asarray(indices_l)
		self.indices_u = np.asarray(indices_u)
		self.shuffle()

	def get_labeled_data(self):
		return self.images_train[self.indices_l], self.labels_train[self.indices_l]

	def get_num_labeled_data(self):
		return len(self.indices_l)

	def get_num_unlabeled_data(self):
		return len(self.indices_u)

	def get_iterator(self, batchsize, train=False, test=False, labeled=False, unlabeled=False, gpu=True):
		if train:
			if labeled:
				return self.get_iterator_train_labeled(batchsize, gpu)
			if unlabeled:
				return self.get_iterator_train_unlabeled(batchsize, gpu)
			raise NotImplementedError()
			
		if test:
			return self.get_iterator_test(batchsize, gpu)

		raise NotImplementedError()

	def get_iterator_train_labeled(self, batchsize, gpu=True):
		return Iterator(self.images_train, self.labels_train, self.indices_l, batchsize, gpu)

	def get_iterator_train_unlabeled(self, batchsize, gpu=True):
		return Iterator(self.images_train, self.labels_train, self.indices_u, batchsize, gpu)

	def sample_labeled_minibatch(self, batchsize, gpu=True):
		x_batch, y_batch, y_onehot_batch = self._sample_minibatch(self.indices_l[:batchsize], batchsize, gpu)
		self.indices_l = np.roll(self.indices_l, batchsize)
		return x_batch, y_batch, y_onehot_batch

	def sample_unlabeled_minibatch(self, batchsize, gpu=True):
		x_batch, y_batch, y_onehot_batch = self._sample_minibatch(self.indices_u[:batchsize], batchsize, gpu)
		self.indices_u = np.roll(self.indices_u, batchsize)
		return x_batch

	def _sample_minibatch(self, batch_indices, batchsize, gpu):
		x_batch = self.images_train[batch_indices]
		y_batch = self.labels_train[batch_indices]
		y_onehot_batch = onehot(y_batch, self.num_classes + self.num_extra_classes)

		if gpu:
			x_batch = cuda.to_gpu(x_batch)
			y_batch = cuda.to_gpu(y_batch)
			y_onehot_batch = cuda.to_gpu(y_onehot_batch)

		return x_batch, y_batch, y_onehot_batch

	def shuffle(self):
		np.random.shuffle(self.indices_l)
		np.random.shuffle(self.indices_u)