# -*- coding: utf-8 -*-
import numpy as np
import math, operator
import params
import sequential
from chainer import functions as F
import aae_semi_supervised as aae

class Config(params.Params):
	def __init__(self):
		self.ndim_x = 28 * 28
		self.ndim_y = 10
		self.ndim_reduction = 2
		self.ndim_z = self.ndim_reduction
		self.cluster_head_distance_threshold = 2
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

		self.chain_cluster_head = sequential.chain.Chain()
		self.cluster_head = sequential.from_dict(params["cluster_head"])
		self.chain_cluster_head.add_sequence_with_name(self.cluster_head, "cluster_head")
		self.chain_cluster_head.setup_optimizers(config.optimizer, config.learning_rate, config.momentum, config.weight_decay, config.gradient_clipping)

	def load(self, dir=None):
		super(AAE, self).load(dir)
		self.chain_cluster_head.load(dir + "/cluster_head.hdf5")

	def save(self, dir=None):
		super(AAE, self).save(dir)
		self.chain_cluster_head.save(dir + "/cluster_head.hdf5")
		
	def to_gpu(self):
		super(AAE, self).to_gpu()
		self.chain_cluster_head.to_gpu()

	def decode_yz_x(self, y, z, test=False):
		raise Exception()

	def encode_yz_representation(self, y, z, test=False):
		y = self.to_variable(y)
		z = self.to_variable(z)
		cluster_head = self.cluster_head(y, test=test)
		return cluster_head + z

	def decode_representation_x(self, representation, test=False):
		representation = self.to_variable(representation)
		return self.decoder(representation, test=test)

	def backprop_cluster_head(self, loss):
		self.chain_cluster_head.backprop(loss)

	# compute combination nCr
	def nCr(self, n, r):
		r = min(r, n - r)
		if r == 0: return 1
		numer = reduce(operator.mul, xrange(n, n - r, -1))
		denom = reduce(operator.mul, xrange(1, r + 1))
		return numer // denom

	def compute_distance_of_cluster_heads(self):
		config = self.config

		# list all possible combinations of two cluster heads
		num_combination = self.nCr(config.ndim_y, 2)

		# starting_labels
		# [0, 1, 0, 0]
		# [0, 0, 1, 0]
		# [0, 0, 1, 0]
		# [0, 0, 0, 1]
		# [0, 0, 0, 1]
		# [0, 0, 0, 1]
		starting_labels = np.zeros((num_combination, config.ndim_y), dtype=np.float32)
		for i in xrange(1, config.ndim_y):
			for n in xrange(i):
				j = int(0.5 * i * (i - 1) + n)
				starting_labels[j, i] = 1

		# ending_labels
		# [1, 0, 0, 0]
		# [1, 0, 0, 0]
		# [0, 1, 0, 0]
		# [1, 0, 0, 0]
		# [0, 1, 0, 0]
		# [0, 0, 1, 0]
		ending_labels = np.zeros((num_combination, config.ndim_y), dtype=np.float32)
		for i in xrange(1, config.ndim_y):
			for n in xrange(i):
				j = int(0.5 * i * (i - 1) + n)
				ending_labels[j, n] = 1

		starting_labels = self.to_variable(starting_labels)
		ending_labels = self.to_variable(ending_labels)
		starting_vector = self.cluster_head(starting_labels)
		ending_vector = self.cluster_head(ending_labels)
		distance = F.sqrt(F.sum((starting_vector - ending_vector) ** 2, axis=1))

		# clip
		distance = F.minimum(distance, self.to_variable(np.full(distance.shape, config.cluster_head_distance_threshold, dtype=np.float32)))
		return distance