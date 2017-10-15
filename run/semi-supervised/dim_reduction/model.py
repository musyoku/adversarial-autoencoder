import sys, os, chainer, operator
import numpy as np
from functools import reduce
from chainer import functions, cuda
sys.path.append(os.path.join("..", "..", ".."))
import aae.nn as nn

class Model(nn.Module):
	def __init__(self, ndim_x=28*28, ndim_y=10, ndim_z=2, ndim_h=1000, cluster_head_distance_threshold=1):
		super(Model, self).__init__()
		self.ndim_x = ndim_x
		self.ndim_y = ndim_y
		self.ndim_z = ndim_z
		self.ndim_h = ndim_h
		self.cluster_head_distance_threshold = cluster_head_distance_threshold

		self.decoder = nn.Module(
			# nn.GaussianNoise(std=0.3),
			nn.Linear(ndim_z, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_x),
			nn.Tanh(),
		)

		self.encoder = nn.Module(
			# nn.GaussianNoise(std=0.3),
			nn.Linear(ndim_x, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_h),
			nn.ReLU(),
		)
		self.encoder.head_y = nn.Linear(ndim_h, ndim_y)
		self.encoder.head_z = nn.Linear(ndim_h, ndim_z)

		self.discriminator_z = nn.Module(
			nn.GaussianNoise(std=0.3),
			nn.Linear(ndim_z, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, 2),
		)

		self.discriminator_y = nn.Module(
			nn.GaussianNoise(std=0.3),
			# nn.GaussianNoise(0, 0.3),
			nn.Linear(ndim_y, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, 2),
		)

		self.cluster_head = nn.Linear(ndim_y, ndim_z, nobias=True)

		for param in self.params():
			if param.name == "W":
				param.data[...] = np.random.normal(0, 0.01, param.data.shape)

		for param in self.cluster_head.params():
			if param.name == "W":
				param.data[...] = np.random.normal(0, 1, param.data.shape)

	def encode_x_yz(self, x, apply_softmax_y=True):
		internal = self.encoder(x)
		y = self.encoder.head_y(internal)
		z = self.encoder.head_z(internal)
		if apply_softmax_y:
			y = functions.softmax(y)
		return y, z

	def encode_yz_representation(self, y, z):
		cluster_head = self.cluster_head(y)
		return cluster_head + z

	def decode_representation_x(self, representation):
		return self.decoder(representation)

	def discriminate_z(self, z, apply_softmax=False):
		logit = self.discriminator_z(z)
		if apply_softmax:
			return functions.softmax(logit)
		return logit

	def discriminate_y(self, y, apply_softmax=False):
		logit = self.discriminator_y(y)
		if apply_softmax:
			return functions.softmax(logit)
		return logit

	# compute combination nCr
	def nCr(self, n, r):
		r = min(r, n - r)
		if r == 0: return 1
		numer = reduce(operator.mul, range(n, n - r, -1))
		denom = reduce(operator.mul, range(1, r + 1))
		return numer // denom

	def compute_distance_of_cluster_heads(self):
		# list all possible combinations of two cluster heads
		num_combination = self.nCr(self.ndim_y, 2)

		# a_labels
		# [0, 1, 0, 0]
		# [0, 0, 1, 0]
		# [0, 0, 1, 0]
		# [0, 0, 0, 1]
		# [0, 0, 0, 1]
		# [0, 0, 0, 1]
		a_labels = np.zeros((num_combination, self.ndim_y), dtype=np.float32)
		for i in range(1, self.ndim_y):
			for n in range(i):
				j = int(0.5 * i * (i - 1) + n)
				a_labels[j, i] = 1

		# b_labels
		# [1, 0, 0, 0]
		# [1, 0, 0, 0]
		# [0, 1, 0, 0]
		# [1, 0, 0, 0]
		# [0, 1, 0, 0]
		# [0, 0, 1, 0]
		b_labels = np.zeros((num_combination, self.ndim_y), dtype=np.float32)
		for i in range(1, self.ndim_y):
			for n in range(i):
				j = int(0.5 * i * (i - 1) + n)
				b_labels[j, n] = 1


		xp = self.xp
		if xp is not np:
			a_labels = cuda.to_gpu(a_labels)
			b_labels = cuda.to_gpu(b_labels)

		a_vector = self.cluster_head(a_labels)
		b_vector = self.cluster_head(b_labels)
		distance = functions.sqrt(functions.sum((a_vector - b_vector) ** 2, axis=1))

		# clip
		distance = functions.clip(distance, 0.0, float(self.cluster_head_distance_threshold))

		return distance