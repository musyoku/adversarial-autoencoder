# -*- coding: utf-8 -*-
import operator as op
import math
import numpy as np
import chainer, os, collections, six
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
import aae
import aae_semi_supervised
from aae import activations, GradientClipping

class Conf(aae.Conf):
	def __init__(self):
		super(Conf, self).__init__()
		# number of category
		self.ndim_y = 10
		self.distance_threshold = 2
		self.learning_rate_for_reconstruction_cost = 0.001
		self.learning_rate_for_cluster_head = 0.01
		self.learning_rate_for_adversarial_cost = 0.001
		self.learning_rate_for_semi_supervised_cost = 0.01

		self.autoencoder_sample_y = False
		self.generator_shared_hidden_units = [500]

		self.discriminator_y_hidden_units = [500]
		self.discriminator_y_activation_function = "softplus"
		self.discriminator_y_apply_dropout = False
		self.discriminator_y_apply_batchnorm = False
		self.discriminator_y_apply_batchnorm_to_input = False

class AAE(aae_semi_supervised.AAE):

	def __init__(self, conf, name="aae"):
		conf.check()
		self.conf = conf
		self.generator_x_yz, self.decoder_r_x, self.discriminator_z, self.discriminator_y = self.build()
		self.name = name

		self.optimizer_generator_x_yz = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_generator_x_yz.setup(self.generator_x_yz)
		# self.optimizer_generator_x_yz.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_generator_x_yz.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_decoder_r_x = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder_r_x.setup(self.decoder_r_x)
		# self.optimizer_decoder_r_x.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_decoder_r_x.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_discriminator_z = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_discriminator_z.setup(self.discriminator_z)
		# self.optimizer_discriminator_z.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_discriminator_z.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_discriminator_y = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_discriminator_y.setup(self.discriminator_y)
		# self.optimizer_discriminator_y.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_discriminator_y.add_hook(GradientClipping(conf.gradient_clipping))

		self.cluster_head = self.build_cluster_head()
		self.optimizer_cluster_head = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_cluster_head.setup(self.cluster_head)
		# self.optimizer_cluster_head.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_cluster_head.add_hook(GradientClipping(conf.gradient_clipping))

	def update_learning_rate(self, lr):
		self.optimizer_generator_x_yz.alpha = lr
		self.optimizer_discriminator_z.alpha = lr
		self.optimizer_discriminator_y.alpha = lr
		self.optimizer_decoder_r_x.alpha = lr
		self.optimizer_cluster_head.alpha = lr

	def build(self):
		wscale = 1

		generator_x_yz = self.build_generator_x_yz()
		decoder_r_x = self.build_decoder_r_x()
		discriminator_z = self.build_discriminator_z()
		discriminator_y = self.build_discriminator_y()

		return generator_x_yz, decoder_r_x, discriminator_z, discriminator_y

	def build_decoder_r_x(self):
		conf = self.conf

		decoder_attributes = {}
		decoder_units = [(conf.ndim_z, conf.decoder_hidden_units[0])]
		decoder_units += zip(conf.decoder_hidden_units[:-1], conf.decoder_hidden_units[1:])
		decoder_units += [(conf.decoder_hidden_units[-1], conf.ndim_x)]
		for i, (n_in, n_out) in enumerate(decoder_units):
			decoder_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=conf.wscale)
			if conf.batchnorm_before_activation:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		decoder_r_x = aae.BernoulliDecoder(**decoder_attributes)
		decoder_r_x.n_layers = len(decoder_units)
		decoder_r_x.activation_function = conf.decoder_activation_function
		decoder_r_x.apply_dropout = conf.decoder_apply_dropout
		decoder_r_x.apply_batchnorm = conf.decoder_apply_batchnorm
		decoder_r_x.apply_batchnorm_to_input = conf.decoder_apply_batchnorm_to_input
		decoder_r_x.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			decoder_r_x.to_gpu()

		return decoder_r_x

	def build_cluster_head(self):
		conf = self.conf

		cluster_head_attributes = {}
		cluster_head_attributes["wc"] = L.Linear(conf.ndim_y, conf.ndim_z, wscale=conf.wscale, nobias=True)
		cluster_head = ClusterHead(**cluster_head_attributes)

		if conf.gpu_enabled:
			cluster_head.to_gpu()

		return cluster_head

	def zero_grads(self):
		self.optimizer_generator_x_yz.zero_grads()
		self.optimizer_decoder_r_x.zero_grads()
		self.optimizer_discriminator_z.zero_grads()
		self.optimizer_discriminator_y.zero_grads()
		self.optimizer_cluster_head.zero_grads()

	def update_autoencoder(self):
		self.optimizer_generator_x_yz.update()
		self.optimizer_decoder_r_x.update()

	def encode_x_r(self, x, noise=None, test=True):
		if isinstance(self.generator_x_yz, aae.UniversalApproximatorGenerator):
			y_distribution, z = self.generator_x_yz(x, test=False, apply_f=True, noise=noise)
		else:
			y_distribution, z = self.generator_x_yz(x, test=False, apply_f=True)

		if self.conf.autoencoder_sample_y:
			y = self.sample_y(y_distribution, argmax=False, test=False)
		else:
			y = y_distribution

		head = self.cluster_head(y)
		representation = head + z
		return representation

	def decode_r_x(self, representation, test=True, apply_f=True):
		return self.decoder_r_x(representation, test=test, apply_f=apply_f)

	def loss_autoencoder_unsupervised(self, x, noise=None):
		representation = self.encode_x_r(x, noise=noise, test=False)
		_x = self.decoder_r_x(representation, test=False, apply_f=True)
		return F.mean_squared_error(x, _x)

	def train_autoencoder_unsupervised(self, x, noise=None):
		loss = self.loss_autoencoder_unsupervised(x, noise=noise)

		self.zero_grads()
		loss.backward()
		self.update_autoencoder()

		if self.gpu_enabled:
			loss.to_cpu()

		return float(loss.data)

	def loss_cluster_head(self):
		def ncr(n, r):
			r = min(r, n-r)
			if r == 0: return 1
			numer = reduce(op.mul, xrange(n, n-r, -1))
			denom = reduce(op.mul, xrange(1, r+1))
			return numer // denom

		xp = self.xp
		comb = ncr(self.conf.ndim_y, 2)
		base = xp.zeros((comb, self.conf.ndim_y), dtype=xp.float32)
		for i in xrange(1, self.conf.ndim_y):
			for n in xrange(i):
				j = int(0.5 * i * (i - 1) + n)
				base[j, i] = 1
		target = xp.zeros((comb, self.conf.ndim_y), dtype=xp.float32)
		for i in xrange(1, self.conf.ndim_y):
			for n in xrange(i):
				j = int(0.5 * i * (i - 1) + n)
				target[j, n] = 1

		base = Variable(base)
		base = self.cluster_head(base)
		target = Variable(target)
		target = self.cluster_head(target)
		distance = F.sum((base - target) ** 2, axis=1)

		# if the distance is larger than eta, the cost function is zero, 
		# and if it is smaller than eta, the cost function linearly penalizes the distance
		eta = self.conf.distance_threshold
		threshold = Variable(xp.full((distance.data.shape[0],), eta ** 2, dtype=xp.float32))
		distance = F.minimum(distance - threshold, Variable(xp.zeros((distance.data.shape[0],), dtype=xp.float32)))
		# eps = Variable(xp.full((distance.data.shape[0],), 1e-6, dtype=xp.float32))
		loss = -distance
		return F.sum(loss) / distance.data.shape[0]

	def train_cluster_head(self):
		loss = self.loss_cluster_head()

		self.zero_grads()
		loss.backward()
		self.optimizer_cluster_head.update()

		if self.gpu_enabled:
			loss.to_cpu()

		return float(loss.data)


class ClusterHead(chainer.Chain):

	def __call__(self, x):
		return self.wc(x)