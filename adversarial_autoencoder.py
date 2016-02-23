# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import chainer
from chainer import cuda, Variable, function, FunctionSet, optimizers
from chainer import functions as F
from chainer import links as L
from activations import activations

class Generator(chainer.Chain):
	def __init__(self, **layers):
		super(Generator, self).__init__(**layers)

	def forward_one_step_deteministic(self, x, test):
		activate = activations[self.activation_type]
		chain = [x]

		# Hidden
		for i in range(self.n_layers - 1):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.enable_batchnorm:
				u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			output = activate(u)
			if self.enable_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		# Output
		u = getattr(self, "layer_%i" % (self.n_layers - 1))(chain[-1])
		if self.enable_batchnorm and self.enable_batchnorm_to_output:
			u = getattr(self, "batchnorm_%i" % (self.n_layers - 1))(u, test=test)
		if self.output_activation_type is None:
			chain.append(u)
		else:
			chain.append(activations[self.output_activation_type](u))

		return chain[-1]

	def forward_one_step_gaussian(self, x, test):
		activate = activations[self.activation_type]

		chain_mean = [x]
		chain_variance = [x]

		# Hidden
		for i in range(self.n_layers - 1):
			u = getattr(self, "layer_mean_%i" % i)(chain_mean[-1])
			if self.enable_batchnorm:
				u = getattr(self, "batchnorm_mean_%i" % i)(u, test=test)
			output = activate(u)
			if self.enable_dropout:
				output = F.dropout(output, train=not test)
			chain_mean.append(output)

			u = getattr(self, "layer_variance_%i" % i)(chain_variance[-1])
			if self.enable_batchnorm:
				u = getattr(self, "batchnorm_variance_%i" % i)(u, test=test)
			output = activate(u)
			if self.enable_dropout:
				output = F.dropout(output, train=not test)
			chain_variance.append(output)


		# Output
		u = getattr(self, "layer_mean_%i" % (self.n_layers - 1))(chain_mean[-1])
		if self.enable_batchnorm and self.enable_batchnorm_to_output:
			u = getattr(self, "batchnorm_mean_%i" % (self.n_layers - 1))(u, test=test)
		if self.output_activation_type is None:
			chain_mean.append(u)
		else:
			chain_mean.append(activations[self.output_activation_type](u))

		u = getattr(self, "layer_variance_%i" % (self.n_layers - 1))(chain_variance[-1])
		if self.enable_batchnorm and self.enable_batchnorm_to_output:
			u = getattr(self, "batchnorm_variance_%i" % (self.n_layers - 1))(u, test=test)
		if self.output_activation_type is None:
			chain_variance.append(u)
		else:
			chain_variance.append(activations[self.output_activation_type](u))

		mean = chain_mean[-1]

		## log(sigma^2)
		ln_var = chain_variance[-1]

		return F.gaussian(mean, ln_var)

	def forward_one_step(self, x, test=False):
		encoder = {
			"deterministic": self.forward_one_step_deteministic, 
			"gaussian": self.forward_one_step_gaussian
		}
		if self.encoder_type not in encoder:
			raise NotImplementedError()

		return encoder[self.encoder_type](x, test=test)

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

class Discriminator(chainer.Chain):
	def __init__(self, **layers):
		super(Discriminator, self).__init__(**layers)

	# 出力は2次元のベクトル
	## 0番目の要素はデータが本物である度合い
	## 1番目の要素はデータが偽物である度合い
	# 確率に変換するにはexp(z[0])/{exp(z[0]) + exp(z[1])}
	# 後の誤差逆伝播を考えここでは確率への変換は行わない
	def forward_one_step(self, z, test=False):
		activate = activations[self.activation_type]
		chain = [z]

		# Hidden
		for i in range(self.n_layers - 1):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.enable_batchnorm:
				if i == 0 and self.enable_batchnorm_to_input == False:
					pass
				else:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			output = activate(u)
			if self.enable_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		# Output
		u = getattr(self, "layer_%i" % (self.n_layers - 1))(chain[-1])
		if self.enable_batchnorm:
			u = getattr(self, "batchnorm_%i" % (self.n_layers - 1))(u, test=test)
		if self.softmax_activation_type is None:
			chain.append(u)
		else:
			chain.append(activations[self.softmax_activation_type](u))

		return chain[-1]

	def __call__(self, z, test=False):
		return self.forward_one_step(z, test=test)

class Decoder(chainer.Chain):
	def __init__(self, **layers):
		super(Decoder, self).__init__(**layers)

	def forward_one_step(self, z, test):
		activate = activations[self.activation_type]
		chain = [z]

		# Hidden
		for i in range(self.n_layers - 1):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.enable_batchnorm:
				u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			chain.append(activate(u))

		# Output
		u = getattr(self, "layer_%i" % (self.n_layers - 1))(chain[-1])
		if self.enable_batchnorm and self.enable_batchnorm_to_output:
			u = getattr(self, "batchnorm_%i" % (self.n_layers - 1))(u, test=test)
		if self.output_activation_type is None:
			chain.append(u)
		else:
			chain.append(activations[self.output_activation_type](u))

		return chain[-1]

	def __call__(self, z, test=False):
		return self.forward_one_step(z, test=test)