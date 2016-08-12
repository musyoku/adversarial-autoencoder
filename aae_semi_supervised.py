# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
import aae
import aae_style
from aae import activations, GradientClipping

class Conf(aae.Conf):
	def __init__(self):
		super(Conf, self).__init__()
		# number of category
		self.ndim_y = 10
		self.learning_rate_for_reconstruction_cost = 0.001
		self.learning_rate_for_adversarial_cost = 0.001
		self.learning_rate_for_semi_supervised_cost = 0.01
		self.momentum_for_reconstruction_cost = 0.9
		self.momentum_for_adversarial_cost = 0.1
		self.momentum_for_semi_supervised_cost = 0.9

		self.autoencoder_x_y_sample_y = False
		self.generator_shared_hidden_units = [500]

		self.discriminator_y_hidden_units = [500]
		self.discriminator_y_activation_function = "softplus"
		self.discriminator_y_apply_dropout = False
		self.discriminator_y_apply_batchnorm = False
		self.discriminator_y_apply_batchnorm_to_input = False

class AAE(aae_style.AAE):

	def __init__(self, conf, name="aae"):
		conf.check()
		self.conf = conf
		self.generator_x_yz, self.decoder_yz_x, self.discriminator_z, self.discriminator_y = self.build()
		self.name = name

		self.optimizer_generator_x_yz = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_generator_x_yz.setup(self.generator_x_yz)
		# self.optimizer_generator_x_yz.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_generator_x_yz.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_decoder_yz_x = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder_yz_x.setup(self.decoder_yz_x)
		# self.optimizer_decoder_yz_x.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_decoder_yz_x.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_discriminator_z = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_discriminator_z.setup(self.discriminator_z)
		# self.optimizer_discriminator_z.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_discriminator_z.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_discriminator_y = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_discriminator_y.setup(self.discriminator_y)
		# self.optimizer_discriminator_y.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_discriminator_y.add_hook(GradientClipping(conf.gradient_clipping))

	def update_learning_rate(self, lr):
		self.optimizer_generator_x_yz.alpha = lr
		self.optimizer_discriminator_z.alpha = lr
		self.optimizer_discriminator_y.alpha = lr
		self.optimizer_decoder_yz_x.alpha = lr

	def update_momentum(self, momentum):
		self.optimizer_generator_x_yz.beta1 = momentum
		self.optimizer_discriminator_z.beta1 = momentum
		self.optimizer_discriminator_y.beta1 = momentum
		self.optimizer_decoder_yz_x.beta1 = momentum

	def build(self):
		wscale = 1

		generator_x_yz = self.build_generator_x_yz()
		decoder_yz_x = self.build_decoder_yz_x()
		discriminator_z = self.build_discriminator_z()
		discriminator_y = self.build_discriminator_y()

		return generator_x_yz, decoder_yz_x, discriminator_z, discriminator_y

	def build_generator_x_yz(self):
		conf = self.conf

		generator_attributes = {}
		generator_shared_units = [(conf.ndim_x, conf.generator_shared_hidden_units[0])]
		generator_shared_units += zip(conf.generator_shared_hidden_units[:-1], conf.generator_shared_hidden_units[1:])

		for i, (n_in, n_out) in enumerate(generator_shared_units):
			generator_attributes["shared_layer_%i" % i] = L.Linear(n_in, n_out, wscale=conf.wscale)
			if conf.batchnorm_before_activation:
				generator_attributes["shared_batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				generator_attributes["shared_batchnorm_%i" % i] = L.BatchNormalization(n_in)

		if len(conf.generator_hidden_units)  == 0:
			raise Exception()
			
		generator_units = [(conf.generator_shared_hidden_units[-1], conf.generator_hidden_units[0])]
		generator_units += zip(conf.generator_hidden_units[:-1], conf.generator_hidden_units[1:])

		for i, (n_in, n_out) in enumerate(generator_units):
			generator_attributes["layer_classifier_%i" % i] = L.Linear(n_in, n_out, wscale=conf.wscale)
			generator_attributes["layer_style_%i" % i] = L.Linear(n_in, n_out, wscale=conf.wscale)
			if conf.batchnorm_before_activation:
				generator_attributes["batchnorm_classifier_%i" % i] = L.BatchNormalization(n_out)
				generator_attributes["batchnorm_style_%i" % i] = L.BatchNormalization(n_out)
			else:
				generator_attributes["batchnorm_classifier_%i" % i] = L.BatchNormalization(n_in)
				generator_attributes["batchnorm_style_%i" % i] = L.BatchNormalization(n_in)

		if conf.q_z_x_type == aae.Q_Z_X_TYPE_DETERMINISTIC:
			generator_attributes["layer_output_classifier"] = L.Linear(conf.generator_hidden_units[-1], conf.ndim_y, wscale=conf.wscale)
			generator_attributes["layer_output_style"] = L.Linear(conf.generator_hidden_units[-1], conf.ndim_z, wscale=conf.wscale)
			generator_x_yz = SemiSupervisedMLPGenerator(**generator_attributes)

		elif conf.q_z_x_type == aae.Q_Z_X_TYPE_GAUSSIAN:
			generator_attributes["layer_output_classifier"] = L.Linear(conf.generator_hidden_units[-1], conf.ndim_y, wscale=conf.wscale)
			generator_attributes["layer_output_style_mean"] = L.Linear(conf.generator_hidden_units[-1], conf.ndim_z, wscale=conf.wscale)
			generator_attributes["layer_output_style_var"] = L.Linear(conf.generator_hidden_units[-1], conf.ndim_z, wscale=conf.wscale)
			generator_x_yz = SemiSupervisedGaussianGenerator(**generator_attributes)

		elif conf.q_z_x_type == aae.Q_Z_X_TYPE_UNIVERSAL_APPROXIMATOR_POSTERIOR:
			# generator_attributes["layer_output_classifier"] = L.Linear(conf.generator_hidden_units[-1], conf.ndim_y, wscale=conf.wscale)
			# generator_attributes["layer_output_style"] = L.Linear(conf.generator_hidden_units[-1], conf.ndim_z, wscale=conf.wscale)
			# generator_x_yz = UniversalApproximatorGenerator(**generator_attributes)
			raise Exception()

		else:
			raise Exception()

		generator_x_yz.n_layers = len(generator_units)
		generator_x_yz.n_shared_layers = len(generator_shared_units)
		generator_x_yz.activation_function = conf.generator_activation_function
		generator_x_yz.apply_dropout = conf.generator_apply_dropout
		generator_x_yz.apply_batchnorm = conf.generator_apply_batchnorm
		generator_x_yz.apply_batchnorm_to_input = conf.generator_apply_batchnorm_to_input
		generator_x_yz.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			generator_x_yz.to_gpu()

		return generator_x_yz
		
	def build_discriminator_y(self):
		conf = self.conf

		discriminator_y_attributes = {}
		discriminator_y_units = [(conf.ndim_y, conf.discriminator_y_hidden_units[0])]
		discriminator_y_units += zip(conf.discriminator_y_hidden_units[:-1], conf.discriminator_y_hidden_units[1:])
		discriminator_y_units += [(conf.discriminator_y_hidden_units[-1], 2)]
		for i, (n_in, n_out) in enumerate(discriminator_y_units):
			discriminator_y_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=conf.wscale)
			if conf.batchnorm_before_activation:
				discriminator_y_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				discriminator_y_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		discriminator_y = aae.SoftmaxClassifier(**discriminator_y_attributes)
		discriminator_y.n_layers = len(discriminator_y_units)
		discriminator_y.activation_function = conf.discriminator_y_activation_function
		discriminator_y.apply_dropout = conf.discriminator_y_apply_dropout
		discriminator_y.apply_batchnorm = conf.discriminator_y_apply_batchnorm
		discriminator_y.apply_batchnorm_to_input = conf.discriminator_y_apply_batchnorm_to_input
		discriminator_y.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			discriminator_y.to_gpu()

		return discriminator_y

	def encode_x_yz(self, x, noise=None, test=True, apply_f=True):
		if isinstance(self.generator_x_yz, aae.UniversalApproximatorGenerator):
			y_distribution, z = self.generator_x_yz(x, test=test, apply_f=apply_f, noise=noise)
		elif isinstance(self.generator_x_yz, SemiSupervisedGaussianGenerator):
			if apply_f:
				y_distribution, z = self.generator_x_yz(x, test=test, apply_f=apply_f)
			else:
				y_distribution, z_mean, z_ln_var = self.generator_x_yz(x, test=test, apply_f=apply_f)
				return y_distribution, z_mean, z_ln_var
		else:
			y_distribution, z = self.generator_x_yz(x, test=test, apply_f=apply_f)

		return y_distribution, z

	def sample_y(self, y_distribution, argmax=False, test=False):
		if isinstance(y_distribution, Variable):
			y_distribution = y_distribution.data

		batchsize = y_distribution.shape[0]
		n_labels = y_distribution.shape[1]
		if self.gpu_enabled:
			y_distribution = cuda.to_cpu(y_distribution)
		sampled_y = np.zeros((batchsize, n_labels), dtype=np.float32)
		if argmax:
			args = np.argmax(y_distribution, axis=1)
			for b in xrange(batchsize):
				sampled_y[b, args[b]] = 1
		else:
			for b in xrange(batchsize):
				label_id = np.random.choice(np.arange(n_labels), p=y_distribution[b])
				sampled_y[b, label_id] = 1
		sampled_y = Variable(sampled_y)
		if self.gpu_enabled:
			sampled_y.to_gpu()
		return sampled_y

	def sample_x_label(self, x, argmax=True, test=False):
		batchsize = x.data.shape[0]
		y_distribution, _ = self.generator_x_yz(x, test=test, apply_f=True)
		y_distribution = y_distribution.data
		n_labels = y_distribution.shape[1]
		if self.gpu_enabled:
			y_distribution = cuda.to_cpu(y_distribution)
		if argmax:
			sampled_label = np.argmax(y_distribution, axis=1)
		else:
			sampled_label = np.zeros((batchsize,), dtype=np.int32)
			labels = np.arange(n_labels)
			for b in xrange(batchsize):
				label_id = np.random.choice(labels, p=y_distribution[b])
				sampled_label[b] = label_id
		return sampled_label

	def zero_grads(self):
		self.optimizer_generator_x_yz.zero_grads()
		self.optimizer_decoder_yz_x.zero_grads()
		self.optimizer_discriminator_z.zero_grads()
		self.optimizer_discriminator_y.zero_grads()
		
	def update_autoencoder(self):
		self.optimizer_generator_x_yz.update()
		self.optimizer_decoder_yz_x.update()

	def update_discriminator(self):
		self.optimizer_discriminator_z.update()
		self.optimizer_discriminator_y.update()

	def update_generator(self):
		self.optimizer_generator_x_yz.update()

	def update_classifier(self):
		self.optimizer_generator_x_yz.update()

	def loss_autoencoder_unsupervised(self, x, noise=None):
		if isinstance(self.generator_x_yz, aae.UniversalApproximatorGenerator):
			y_distribution, z = self.generator_x_yz(x, test=False, apply_f=True, noise=noise)
		else:
			y_distribution, z = self.generator_x_yz(x, test=False, apply_f=True)

		if self.conf.autoencoder_x_y_sample_y:
			y = self.sample_y(y_distribution, argmax=False, test=False)
			_x = self.decoder_yz_x(y, z, test=False, apply_f=True)
		else:
			_x = self.decoder_yz_x(y_distribution, z, test=False, apply_f=True)

		return F.mean_squared_error(x, _x)

	def train_autoencoder_unsupervised(self, x, noise=None):
		loss = self.loss_autoencoder_unsupervised(x, noise=noise)

		self.zero_grads()
		loss.backward()
		self.update_autoencoder()

		if self.gpu_enabled:
			loss.to_cpu()

		return float(loss.data)

	def loss_classifier(self, x, label, noise=None):
		if isinstance(self.generator_x_yz, aae.UniversalApproximatorGenerator):
			y_raw_output, z = self.generator_x_yz(x, test=False, apply_f=False, noise=noise)
		elif isinstance(self.generator_x_yz, SemiSupervisedGaussianGenerator):
			y_raw_output, z_mean, z_ln_var = self.generator_x_yz(x, test=False, apply_f=False)
		else:
			y_raw_output, z = self.generator_x_yz(x, test=False, apply_f=False)

		return F.softmax_cross_entropy(y_raw_output, label)

	def train_classifier(self, x, label, noise=None):
		loss = self.loss_classifier(x, label, noise=noise)

		self.zero_grads()
		loss.backward()
		self.update_classifier()

		if self.gpu_enabled:
			loss.to_cpu()

		return float(loss.data)

	def loss_generator_x_yz(self, x, noise=None):
		xp = self.xp

		# We fool discriminator into thinking that z_fake comes from the true prior distribution. 
		if isinstance(self.generator_x_yz, aae.UniversalApproximatorGenerator):
			y_distribution_fake, z_fake = self.generator_x_yz(x, test=False, apply_f=True, noise=noise)
		else:
			y_distribution_fake, z_fake = self.generator_x_yz(x, test=False, apply_f=True)

		p_fake_z = self.discriminator_z(z_fake, softmax=False, test=False)

		# y_fake = self.sample_y(y_distribution_fake, argmax=False, test=False)
		p_fake_y = self.discriminator_y(y_distribution_fake, softmax=False, test=False)

		# 0: Samples from true distribution
		# 1: Samples from generator
		loss_z = F.softmax_cross_entropy(p_fake_z, Variable(xp.zeros(p_fake_z.data.shape[0], dtype=np.int32)))
		loss_y = F.softmax_cross_entropy(p_fake_y, Variable(xp.zeros(p_fake_y.data.shape[0], dtype=np.int32)))

		return loss_z + loss_y

	def train_generator_x_yz(self, x, noise=None):
		loss = self.loss_generator_x_yz(x, noise=noise)

		self.zero_grads()
		loss.backward()
		self.update_generator()

		if self.gpu_enabled:
			loss.to_cpu()

		return float(loss.data)

	def loss_discriminator_yz(self, x, y_true, z_true, noise=None):
		xp = self.xp

		# y_true came from true prior distribution
		p_true_y = self.discriminator_y(y_true, softmax=False, test=False)

		# z_true came from true prior distribution
		p_true_z = self.discriminator_z(z_true, softmax=False, test=False)

		# 0: Samples from true distribution
		# 1: Samples from generator
		loss_true_y = F.softmax_cross_entropy(p_true_y, Variable(xp.zeros(p_true_y.data.shape[0], dtype=np.int32)))
		loss_true_z = F.softmax_cross_entropy(p_true_z, Variable(xp.zeros(p_true_z.data.shape[0], dtype=np.int32)))

		# z_fake was generated by generator
		if isinstance(self.generator_x_yz, aae.UniversalApproximatorGenerator):
			y_distribution_fake, z_fake = self.generator_x_yz(x, test=False, apply_f=True, noise=noise)
		else:
			y_distribution_fake, z_fake = self.generator_x_yz(x, test=False, apply_f=True)

		# y_fake = self.sample_y(y_distribution_fake, argmax=False, test=False)
		p_fake_y = self.discriminator_y(y_distribution_fake, softmax=False, test=False)
		loss_fake_y = F.softmax_cross_entropy(p_fake_y, Variable(xp.ones(p_fake_y.data.shape[0], dtype=np.int32)))

		p_fake_z = self.discriminator_z(z_fake, softmax=False, test=False)
		loss_fake_z = F.softmax_cross_entropy(p_fake_z, Variable(xp.ones(p_fake_z.data.shape[0], dtype=np.int32)))

		return loss_true_y + loss_fake_y + loss_true_z + loss_fake_z

	def train_discriminator_yz(self, x, y_true, z_true, noise=None):
		loss = self.loss_discriminator_yz(x, y_true, z_true, noise=noise)

		self.zero_grads()
		loss.backward()
		self.update_discriminator()

		if self.gpu_enabled:
			loss.to_cpu()

		return float(loss.data) / 2.0

class SemiSupervisedMLPGenerator(aae.MultiLayerPerceptron):

	def compute_hidden_output(self, x, test=False):
		f = activations[self.activation_function]
		chain = [x]

		# Shared units
		for i in range(self.n_shared_layers):
			u = chain[-1]
			if self.batchnorm_before_activation:
				u = getattr(self, "shared_layer_%i" % i)(u)

			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "shared_batchnorm_%d" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "shared_batchnorm_%d" % i)(u, test=test)

			if self.batchnorm_before_activation == False:
				u = getattr(self, "shared_layer_%i" % i)(u)

			output = f(u)

			if self.apply_dropout:
				output = F.dropout(output, train=not test)

			chain.append(output)

		chain_classifier = [chain[-1]]
		chain_style = [chain[-1]]

		for i in range(self.n_layers):
			u_classifier = chain_classifier[-1]
			u_style = chain_style[-1]

			if self.batchnorm_before_activation:
				u_classifier = getattr(self, "layer_classifier_%i" % i)(u_classifier)
				u_style = getattr(self, "layer_style_%i" % i)(u_style)

			if self.apply_batchnorm:
				u_classifier = getattr(self, "batchnorm_classifier_%d" % i)(u_classifier, test=test)
				u_style = getattr(self, "batchnorm_style_%d" % i)(u_style, test=test)

			if self.batchnorm_before_activation == False:
				u_classifier = getattr(self, "layer_classifier_%i" % i)(u_classifier)
				u_style = getattr(self, "layer_style_%i" % i)(u_style)

			output_classifier = f(u_classifier)
			output_style = f(u_style)

			if self.apply_dropout:
				output_classifier = F.dropout(output_classifier, train=not test)
				output_style = F.dropout(output_style, train=not test)

			chain_classifier.append(output_classifier)
			chain_style.append(output_style)

		return chain_classifier[-1], chain_style[-1]

	def forward_one_step(self, x, test=False):
		output_classifier, output_style = self.compute_hidden_output(x, test=test)
		output_classifier = self.layer_output_classifier(output_classifier)
		output_style = self.layer_output_style(output_style)
		return output_classifier, output_style

	def __call__(self, x, test=False, apply_f=True):
		output_classifier, output_style = self.forward_one_step(x, test=test)
		if apply_f:
			output_classifier = F.softmax(output_classifier)
		return output_classifier, output_style

class SemiSupervisedGaussianGenerator(SemiSupervisedMLPGenerator):

	def forward_one_step(self, x, test=False, clip=True):
		output_classifier, output_style = self.compute_hidden_output(x, test=test)
		output_classifier = self.layer_output_classifier(output_classifier)
		style_mean = self.layer_output_style_mean(output_style)
		style_ln_var = self.layer_output_style_var(output_style)

		# avoid nan
		if clip:
			clip_min = math.log(0.001)
			clip_max = math.log(10)
			style_ln_var = F.clip(style_ln_var, clip_min, clip_max)

		return output_classifier, style_mean, style_ln_var

	def __call__(self, x, test=False, apply_f=True):
		output_classifier, style_mean, style_ln_var = self.forward_one_step(x, test=test, clip=True)
		if apply_f:
			output_style = F.gaussian(style_mean, style_ln_var)
			output_classifier = F.softmax(output_classifier)
			return output_classifier, output_style
		return output_classifier, style_mean, style_ln_var
