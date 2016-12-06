# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, math, random, copy
from chainer import cuda, Variable, serializers
import params
import sequential

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

class AAE(object):
	def __init__(self, params):
		super(AAE, self).__init__()
		self.params = copy.deepcopy(params)
		self.config = to_object(params["config"])
		self.chain_discriminator = sequential.chain.Chain()
		self.chain_decoder = sequential.chain.Chain()
		self.chain_generator = sequential.chain.Chain()
		self._gpu = False

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		self.chain_decoder.load(dir + "/decoder.hdf5")
		self.chain_discriminator.load(dir + "/discriminator.hdf5")
		self.chain_generator.load(dir + "/generator.hdf5")

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		self.chain_decoder.save(dir + "/decoder.hdf5")
		self.chain_discriminator.save(dir + "/discriminator.hdf5")
		self.chain_generator.save(dir + "/generator.hdf5")

	def backprop_decoder(self, loss):
		self.chain_decoder.backprop(loss)

	def backprop_discriminator(self, loss):
		self.chain_discriminator.backprop(loss)

	def backprop_generator(self, loss):
		self.chain_generator.backprop(loss)

	def update_momentum_for_decoder(self, momentum):
		self.chain_decoder.update_momentum(momentum)

	def update_momentum_for_discriminator(self, momentum):
		self.chain_discriminator.update_momentum(momentum)

	def update_momentum_for_generator(self, momentum):
		self.chain_generator.update_momentum(momentum)

	def update_learning_rate_for_decoder(self, lr):
		self.chain_decoder.update_learning_rate(lr)

	def update_learning_rate_for_discriminator(self, lr):
		self.chain_discriminator.update_learning_rate(lr)

	def update_learning_rate_for_generator(self, lr):
		self.chain_generator.update_learning_rate(lr)

	def to_gpu(self):
		self.chain_decoder.to_gpu()
		self.chain_discriminator.to_gpu()
		self.chain_generator.to_gpu()
		self._gpu = True

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self._gpu

	@property
	def xp(self):
		if self.gpu_enabled:
			return cuda.cupy
		return np

	def to_variable(self, x):
		if isinstance(x, Variable) == False:
			x = Variable(x)
			if self.gpu_enabled:
				x.to_gpu()
		return x

	def to_numpy(self, x):
		if isinstance(x, Variable) == True:
			x.to_cpu()
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		return x.shape[0]