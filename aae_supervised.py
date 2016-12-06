# -*- coding: utf-8 -*-
import numpy as np
import aae
import params
import sequential
from chainer import functions as F

class Config(params.Params):
	def __init__(self):
		self.ndim_x = 28 * 28
		self.ndim_y = 10
		self.ndim_z = 10
		self.distribution_z = "deterministic"	# deterministic or gaussian
		self.weight_init_std = 0.01
		self.weight_initializer = "Normal"	# Normal, GlorotNormal or HeNormal
		self.nonlinearity = "relu"
		self.optimizer = "Adam"
		self.learning_rate = 0.0003
		self.momentum = 0.9
		self.gradient_clipping = 10
		self.weight_decay = 0

class AAE(aae.AAE):
	def __init__(self, params):
		super(AAE, self).__init__(params)
		self.init(params)

	def init(self, params):
		config = self.config

		self.decoder = sequential.from_dict(params["decoder"])
		self.chain_decoder.add_sequence(self.decoder)
		self.chain_decoder.setup_optimizers(config.optimizer, config.learning_rate, config.momentum, config.weight_decay, config.gradient_clipping)
		
		self.discriminator = sequential.from_dict(params["discriminator"])
		self.chain_discriminator.add_sequence(self.discriminator)
		self.chain_discriminator.setup_optimizers(config.optimizer, config.learning_rate, config.momentum, config.weight_decay, config.gradient_clipping)
		
		self.generator = sequential.from_dict(params["generator"])
		self.chain_generator.add_sequence(self.generator)
		self.chain_generator.setup_optimizers(config.optimizer, config.learning_rate, config.momentum, config.weight_decay, config.gradient_clipping)

	def encode_x_z(self, x, test=False):
		config = self.config
		x = self.to_variable(x)
		if config.distribution_z == "deterministic":
			z = self.generator(x, test=test)
		elif config.distribution_z == "gaussian":
			z_mean, z_ln_var = self.generator(x, test=test)
			z = F.gaussian(z_mean, z_ln_var)
		return z

	def decode_yz_x(self, y, z, test=False):
		y = self.to_variable(y)
		z = self.to_variable(z)
		x = self.decoder(y, z, test=test)
		return x

	def discriminate_z(self, z_batch, test=False, apply_softmax=True):
		z_batch = self.to_variable(z_batch)
		prob = self.discriminator(z_batch, test=test)
		if apply_softmax:
			prob = F.softmax(prob)
		return prob
