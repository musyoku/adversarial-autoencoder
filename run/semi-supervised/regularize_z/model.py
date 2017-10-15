import sys, os, chainer
import numpy as np
from chainer import functions
sys.path.append(os.path.join("..", "..", ".."))
import aae.nn as nn

class Model(nn.Module):
	def __init__(self, ndim_x=28*28, ndim_y=11, ndim_z=2, ndim_h=1000):
		super(Model, self).__init__()
		self.ndim_x = ndim_x
		self.ndim_y = ndim_y
		self.ndim_z = ndim_z
		self.ndim_h = ndim_h
		
		self.decoder = nn.Module(
			# nn.BatchNormalization(ndim_h),
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
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_x, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_z),
		)

		self.discriminator = nn.Module(
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, ndim_h),
			nn.ReLU(),
			# nn.BatchNormalization(ndim_h),
			nn.Linear(ndim_h, 2),
		)
		self.discriminator.merge_y = nn.Linear(ndim_y, ndim_h, nobias=True)
		self.discriminator.merge_z = nn.Module(
			nn.GaussianNoise(std=0.3),
			nn.Linear(ndim_z, ndim_h, nobias=True),
		)
		self.discriminator.merge_bias = nn.Bias(shape=(ndim_h,))

		for param in self.params():
			if param.name == "W":
				param.data[...] = np.random.normal(0, 0.01, param.data.shape)

	def encode_x_z(self, x):
		return self.encoder(x)

	def discriminate(self, y, z, apply_softmax=False):
		merge = self.discriminator.merge_bias(self.discriminator.merge_y(y) + self.discriminator.merge_z(z))
		logit = self.discriminator(merge)
		if apply_softmax:
			return functions.softmax(logit)
		return logit

	def decode_z_x(self, z):
		return self.decoder(z)