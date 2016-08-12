# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
import aae
from aae import activations, GradientClipping

class Conf(aae.Conf):
	def __init__(self):
		super(Conf, self).__init__()
		# number of category
		self.ndim_y = 10

class AAE(aae.AAE):

	def __init__(self, conf, name="aae"):
		conf.check()
		self.conf = conf
		self.generator_x_z, self.decoder_yz_x, self.discriminator_z = self.build()
		self.name = name

		self.optimizer_generator_x_z = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_generator_x_z.setup(self.generator_x_z)
		# self.optimizer_generator_x_z.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_generator_x_z.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_decoder_yz_x = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder_yz_x.setup(self.decoder_yz_x)
		# self.optimizer_decoder_yz_x.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_decoder_yz_x.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_discriminator_z = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_discriminator_z.setup(self.discriminator_z)
		# self.optimizer_discriminator_z.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_discriminator_z.add_hook(GradientClipping(conf.gradient_clipping))

	def build(self):
		wscale = 1

		generator_x_z = self.build_generator_x_z()
		decoder_yz_x = self.build_decoder_yz_x()
		discriminator_z = self.build_discriminator_z()

		return generator_x_z, decoder_yz_x, discriminator_z

	def build_decoder_yz_x(self):
		conf = self.conf

		decoder_attributes = {}
		decoder_units = zip(conf.decoder_hidden_units[:-1], conf.decoder_hidden_units[1:])
		decoder_units += [(conf.decoder_hidden_units[-1], conf.ndim_x)]
		for i, (n_in, n_out) in enumerate(decoder_units):
			decoder_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=conf.wscale)
			if conf.batchnorm_before_activation:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				decoder_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
		decoder_attributes["layer_merge_z"] = L.Linear(conf.ndim_z, conf.decoder_hidden_units[0], wscale=conf.wscale)
		decoder_attributes["layer_merge_y"] = L.Linear(conf.ndim_y, conf.decoder_hidden_units[0], wscale=conf.wscale)
		if conf.batchnorm_before_activation:
			decoder_attributes["batchnorm_merge"] = L.BatchNormalization(conf.decoder_hidden_units[0])
		else:
			decoder_attributes["batchnorm_merge"] = L.BatchNormalization(conf.ndim_z)

		decoder_yz_x = BernoulliDecoder(**decoder_attributes)
		decoder_yz_x.n_layers = len(decoder_units)
		decoder_yz_x.activation_function = conf.decoder_activation_function
		decoder_yz_x.apply_dropout = conf.decoder_apply_dropout
		decoder_yz_x.apply_batchnorm = conf.decoder_apply_batchnorm
		decoder_yz_x.apply_batchnorm_to_input = conf.decoder_apply_batchnorm_to_input
		decoder_yz_x.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			decoder_yz_x.to_gpu()

		return decoder_yz_x

	def zero_grads(self):
		self.optimizer_generator_x_z.zero_grads()
		self.optimizer_decoder_yz_x.zero_grads()
		self.optimizer_discriminator_z.zero_grads()

	def update_autoencoder(self):
		self.optimizer_generator_x_z.update()
		self.optimizer_decoder_yz_x.update()

	def update_generator(self):
		self.optimizer_generator_x_z.update()

	def loss_autoencoder(self, x, y, noise=None):
		if isinstance(self.generator_x_z, aae.UniversalApproximatorGenerator):
			z = self.generator_x_z(x, test=False, apply_f=True, noise=noise)
		else:
			z = self.generator_x_z(x, test=False, apply_f=True)
		_x = self.decoder_yz_x(y, z, test=False, apply_f=True)
		return F.mean_squared_error(x, _x)

	def train_autoencoder(self, x, y, noise=None):
		loss = self.loss_autoencoder(x, y, noise=noise)

		self.zero_grads()
		loss.backward()
		self.update_autoencoder()

		if self.gpu_enabled:
			loss.to_cpu()

		return float(loss.data)

class BernoulliDecoder(aae.BernoulliDecoder):

	def forward_one_step(self, y, z, test=False):
		f = activations[self.activation_function]

		if self.apply_batchnorm_to_input:
			if self.batchnorm_before_activation:
				merged_input = f(self.batchnorm_merge(self.layer_merge_z(z) + self.layer_merge_y(y), test=test))
			else:
				merged_input = f(self.layer_merge_z(self.batchnorm_merge(z, test=test)) + self.layer_merge_y(y))
		else:
			merged_input = f(self.layer_merge_z(z) + self.layer_merge_y(y))

		chain = [merged_input]
		
		# Hidden
		for i in range(self.n_layers):
			u = chain[-1]

			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)

			if i == self.n_layers - 1:
				# Do not apply batchnorm to network output when batchnorm_before_activation = True
				if self.apply_batchnorm and self.batchnorm_before_activation == False:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)

			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)

			if i == self.n_layers - 1:
				output = u
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not test)

			chain.append(output)

		return chain[-1]

	def __call__(self, y, z, test=False, apply_f=True):
		output = self.forward_one_step(y, z, test=test)
		if apply_f:
			return F.sigmoid(output)
		return output