import sys, os, chainer
import numpy as np
from chainer import functions
sys.path.append(os.path.join("..", "..", ".."))
import aae.nn as nn

class Model(nn.Module):
	def __init__(self, ndim_x=28*28, ndim_y=16, ndim_z=10, ndim_h=1000):
		super(Model, self).__init__()
		self.ndim_x = ndim_x
		self.ndim_y = ndim_y
		self.ndim_z = ndim_z
		self.ndim_h = ndim_h

		self.decoder = nn.Module(
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_x),
			nn.Tanh(),
		)
		self.decoder.merge_y = nn.Linear(ndim_y, ndim_h, nobias=True)
		self.decoder.merge_z = nn.Linear(ndim_z, ndim_h, nobias=True)
		self.decoder.merge_bias = nn.Bias(shape=(ndim_h,))

		self.encoder = nn.Module(
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

		for param in self.params():
			if param.name == "W":
				param.data[...] = np.random.normal(0, 0.01, param.data.shape)

	def encode_x_yz(self, x, apply_softmax_y=True):
		internal = self.encoder(x)
		y = self.encoder.head_y(internal)
		z = self.encoder.head_z(internal)
		if apply_softmax_y:
			y = functions.softmax(y)
		return y, z

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

	def decode_yz_x(self, y, z):
		merge = self.decoder.merge_bias(self.decoder.merge_y(y) + self.decoder.merge_z(z))
		merge = functions.relu(merge)
		return self.decoder(merge)