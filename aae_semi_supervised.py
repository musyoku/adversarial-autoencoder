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
		self.chain_decoder.add_sequence_with_name(self.decoder, "decoder")
		self.chain_decoder.setup_optimizers(config.optimizer, config.learning_rate, config.momentum, config.weight_decay, config.gradient_clipping)
		
		self.discriminator_y = sequential.from_dict(params["discriminator_y"])
		self.discriminator_z = sequential.from_dict(params["discriminator_z"])
		self.chain_discriminator.add_sequence_with_name(self.discriminator_y, "y")
		self.chain_discriminator.add_sequence_with_name(self.discriminator_z, "z")
		self.chain_discriminator.setup_optimizers(config.optimizer, config.learning_rate, config.momentum, config.weight_decay, config.gradient_clipping)
		
		self.generator_shared = sequential.from_dict(params["generator_shared"])
		self.generator_y = sequential.from_dict(params["generator_y"])
		self.generator_z = sequential.from_dict(params["generator_z"])
		self.chain_generator.add_sequence_with_name(self.generator_shared, "shared")
		self.chain_generator.add_sequence_with_name(self.generator_y, "y")
		self.chain_generator.add_sequence_with_name(self.generator_z, "z")
		self.chain_generator.setup_optimizers(config.optimizer, config.learning_rate, config.momentum, config.weight_decay, config.gradient_clipping)

	def encode_x_yz(self, x, apply_softmax=True, test=False):
		config = self.config
		x = self.to_variable(x)
		shared_output = self.generator_shared(x, test=test)
		if config.distribution_z == "deterministic":
			z = self.generator_z(shared_output, test=test)
		elif config.distribution_z == "gaussian":
			z_mean, z_ln_var = self.generator_z(shared_output, test=test)
			z = F.gaussian(z_mean, z_ln_var)
		y_distribution = self.generator_y(shared_output, test=test)
		if apply_softmax:
			y_distribution = F.softmax(y_distribution)
		return y_distribution, z

	def argmax_x_label(self, x, test=False):
		y_distribution, z = self.encode_x_yz(x, apply_softmax=True, test=test)
		return self.argmax_label_from_distribution(y_distribution)

	def decode_yz_x(self, y, z, test=False):
		y = self.to_variable(y)
		z = self.to_variable(z)
		x = self.decoder(y, z, test=test)
		return x

	def argmax_label_from_distribution(self, y_distribution):
		y_distribution = self.to_numpy(y_distribution)
		return np.argmax(y_distribution, axis=1)

	# use gumbel-softmax
	def sample_onehot_from_unnormalized_distribution(self, unnormalized_y_distribution, temperature=0.01, test=False):
		eps = 1e-16
		u = np.random.uniform(0, 1, unnormalized_y_distribution.shape).astype(unnormalized_y_distribution.dtype)
		g = self.to_variable(-np.log(-np.log(u + eps) + eps))
		one_hot = F.softmax((unnormalized_y_distribution + g) / temperature)
		return one_hot

	# use gumbel-softmax
	def argmax_onehot_from_unnormalized_distribution(self, unnormalized_y_distribution, temperature=0.01, test=False):
		eps = 1e-16
		one_hot = F.softmax(unnormalized_y_distribution / temperature)
		return one_hot

	def discriminate_z(self, z_batch, test=False, apply_softmax=True):
		z_batch = self.to_variable(z_batch)
		prob = self.discriminator_z(z_batch, test=test)
		if apply_softmax:
			prob = F.softmax(prob)
		return prob

	def discriminate_y(self, y_batch, test=False, apply_softmax=True):
		y_batch = self.to_variable(y_batch)
		prob = self.discriminator_y(y_batch, test=test)
		if apply_softmax:
			prob = F.softmax(prob)
		return prob
