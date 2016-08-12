# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L

activations = {
	"sigmoid": F.sigmoid, 
	"tanh": F.tanh, 
	"softplus": F.softplus, 
	"relu": F.relu, 
	"leaky_relu": F.leaky_relu, 
	"elu": F.elu
}

Q_Z_X_TYPE_DETERMINISTIC = 1
Q_Z_X_TYPE_GAUSSIAN = 2
Q_Z_X_TYPE_UNIVERSAL_APPROXIMATOR_POSTERIOR = 3	# It is currently not supported

class Conf(object):
	def __init__(self):
		self.image_width = 28
		self.image_height = 28
		self.ndim_x = 28 * 28
		self.ndim_z = 50
		self.wscale = 1

		# Deterministic / Gaussian / Universal approximator posterior
		self.q_z_x_type = Q_Z_X_TYPE_DETERMINISTIC

		# True : y = f(BN(Wx + b))
		# False: y = BN(W*f(x) + b)
		self.batchnorm_before_activation = True

		self.generator_hidden_units = [500]
		self.generator_activation_function = "softplus"
		self.generator_apply_dropout = False
		self.generator_apply_batchnorm = False
		self.generator_apply_batchnorm_to_input = False

		self.decoder_hidden_units = [500]
		self.decoder_activation_function = "softplus"
		self.decoder_apply_dropout = False
		self.decoder_apply_batchnorm = False
		self.decoder_apply_batchnorm_to_input = False

		self.discriminator_z_hidden_units = [500]
		self.discriminator_z_activation_function = "softplus"
		self.discriminator_z_apply_dropout = False
		self.discriminator_z_apply_batchnorm = False
		self.discriminator_z_apply_batchnorm_to_input = False

		self.gpu_enabled = True
		self.learning_rate = 0.0003
		self.gradient_momentum = 0.9
		self.gradient_clipping = 5.0

	def check(self):
		if self.q_z_x_type == Q_Z_X_TYPE_UNIVERSAL_APPROXIMATOR_POSTERIOR:
			raise Exception("Universal approximator posterior is currently not supported.")
		pass

def sum_sqnorm(arr):
	sq_sum = collections.defaultdict(float)
	for x in arr:
		with cuda.get_device(x) as dev:
			x = x.ravel()
			s = x.dot(x)
			sq_sum[int(dev)] += s
	return sum([float(i) for i in six.itervalues(sq_sum)])
	
class GradientClipping(object):
	name = "GradientClipping"

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = np.sqrt(sum_sqnorm([p.grad for p in opt.target.params()]))
		if norm == 0:
			return
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad *= rate

class AAE(object):
	# name is used for the filename when you save the model
	def __init__(self, conf, name="aae"):
		conf.check()
		self.conf = conf
		self.generator_x_z, self.decoder_z_x, self.discriminator_z = self.build()
		self.name = name

		self.optimizer_generator_x_z = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_generator_x_z.setup(self.generator_x_z)
		# self.optimizer_generator_x_z.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_generator_x_z.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_decoder_z_x = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_decoder_z_x.setup(self.decoder_z_x)
		# self.optimizer_decoder_z_x.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_decoder_z_x.add_hook(GradientClipping(conf.gradient_clipping))

		self.optimizer_discriminator_z = optimizers.Adam(alpha=conf.learning_rate, beta1=conf.gradient_momentum)
		self.optimizer_discriminator_z.setup(self.discriminator_z)
		# self.optimizer_discriminator_z.add_hook(optimizer.WeightDecay(0.00001))
		self.optimizer_discriminator_z.add_hook(GradientClipping(conf.gradient_clipping))

	def build(self):
		generator_x_z = self.build_generator_x_z()
		decoder_z_x = self.build_decoder_z_x()
		discriminator_z = self.build_discriminator_z()

		return generator_x_z, decoder_z_x, discriminator_z

	def build_generator_x_z(self):
		conf = self.conf

		generator_attributes = {}
		generator_units = [(conf.ndim_x, conf.generator_hidden_units[0])]
		generator_units += zip(conf.generator_hidden_units[:-1], conf.generator_hidden_units[1:])

		if conf.q_z_x_type != Q_Z_X_TYPE_GAUSSIAN:
			generator_units += [(conf.generator_hidden_units[-1], conf.ndim_z)]

		for i, (n_in, n_out) in enumerate(generator_units):
			generator_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=conf.wscale)
			if conf.batchnorm_before_activation:
				generator_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				generator_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		if conf.q_z_x_type == Q_Z_X_TYPE_DETERMINISTIC:
			generator_x_z = MultiLayerPerceptron(**generator_attributes)
		elif conf.q_z_x_type == Q_Z_X_TYPE_GAUSSIAN:
			generator_attributes["layer_output_mean"] = L.Linear(conf.generator_hidden_units[-1], conf.ndim_z, wscale=conf.wscale)
			generator_attributes["layer_output_var"] = L.Linear(conf.generator_hidden_units[-1], conf.ndim_z, wscale=conf.wscale)
			generator_x_z = GaussianGenerator(**generator_attributes)
		elif conf.q_z_x_type == Q_Z_X_TYPE_UNIVERSAL_APPROXIMATOR_POSTERIOR:
			generator_x_z = UniversalApproximatorGenerator(**generator_attributes)
		else:
			raise Exception()

		generator_x_z.n_layers = len(generator_units)
		generator_x_z.activation_function = conf.generator_activation_function
		generator_x_z.apply_dropout = conf.generator_apply_dropout
		generator_x_z.apply_batchnorm = conf.generator_apply_batchnorm
		generator_x_z.apply_batchnorm_to_input = conf.generator_apply_batchnorm_to_input
		generator_x_z.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			generator_x_z.to_gpu()

		return generator_x_z

	def build_decoder_z_x(self, wscale=1):
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

		decoder_z_x = BernoulliDecoder(**decoder_attributes)
		decoder_z_x.n_layers = len(decoder_units)
		decoder_z_x.activation_function = conf.decoder_activation_function
		decoder_z_x.apply_dropout = conf.decoder_apply_dropout
		decoder_z_x.apply_batchnorm = conf.decoder_apply_batchnorm
		decoder_z_x.apply_batchnorm_to_input = conf.decoder_apply_batchnorm_to_input
		decoder_z_x.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			decoder_z_x.to_gpu()

		return decoder_z_x

	def build_discriminator_z(self, wscale=1):
		conf = self.conf

		discriminator_z_attributes = {}
		discriminator_z_units = [(conf.ndim_z, conf.discriminator_z_hidden_units[0])]
		discriminator_z_units += zip(conf.discriminator_z_hidden_units[:-1], conf.discriminator_z_hidden_units[1:])
		discriminator_z_units += [(conf.discriminator_z_hidden_units[-1], 2)]
		for i, (n_in, n_out) in enumerate(discriminator_z_units):
			discriminator_z_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=conf.wscale)
			if conf.batchnorm_before_activation:
				discriminator_z_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				discriminator_z_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		discriminator_z = SoftmaxClassifier(**discriminator_z_attributes)
		discriminator_z.n_layers = len(discriminator_z_units)
		discriminator_z.activation_function = conf.discriminator_z_activation_function
		discriminator_z.apply_dropout = conf.discriminator_z_apply_dropout
		discriminator_z.apply_batchnorm = conf.discriminator_z_apply_batchnorm
		discriminator_z.apply_batchnorm_to_input = conf.discriminator_z_apply_batchnorm_to_input
		discriminator_z.batchnorm_before_activation = conf.batchnorm_before_activation

		if conf.gpu_enabled:
			discriminator_z.to_gpu()

		return discriminator_z

	@property
	def xp(self):
		if hasattr(self, "generator_x_z"):
			return self.generator_x_z.xp
		if hasattr(self, "generator_x_yz"):
			return self.generator_x_yz.xp
		return np

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return True if self.xp is cuda.cupy else False

	def zero_grads(self):
		self.optimizer_generator_x_z.zero_grads()
		self.optimizer_decoder_z_x.zero_grads()
		self.optimizer_discriminator_z.zero_grads()

	def update(self):
		self.optimizer_generator_x_z.update()
		self.optimizer_decoder_z_x.update()
		self.optimizer_discriminator_z.update()

	def update_autoencoder(self):
		self.optimizer_generator_x_z.update()
		self.optimizer_decoder_z_x.update()

	def update_discriminator(self):
		self.optimizer_discriminator_z.update()

	def update_generator(self):
		self.optimizer_generator_x_z.update()

	def loss_autoencoder(self, x, noise=None):
		if isinstance(self.generator_x_z, UniversalApproximatorGenerator):
			z = self.generator_x_z(x, test=False, apply_f=True, noise=noise)
		else:
			z = self.generator_x_z(x, test=False, apply_f=True)
		_x = self.decoder_z_x(z, test=False, apply_f=True)
		return F.mean_squared_error(x, _x)

	def train_autoencoder(self, x, noise=None):
		loss = self.loss_autoencoder(x, noise=noise)

		self.zero_grads()
		loss.backward()
		self.update_autoencoder()

		if self.gpu_enabled:
			loss.to_cpu()

		return float(loss.data)

	def loss_generator_x_z(self, x, noise=None):
		xp = self.xp

		# We fool discriminator into thinking that z_fake comes from the true prior distribution. 
		if isinstance(self.generator_x_z, UniversalApproximatorGenerator):
			z_fake = self.generator_x_z(x, test=False, apply_f=True, noise=noise)
		else:
			z_fake = self.generator_x_z(x, test=False, apply_f=True)
		p_fake = self.discriminator_z(z_fake, softmax=False)

		# 0: Samples from true distribution
		# 1: Samples from generator
		loss = F.softmax_cross_entropy(p_fake, Variable(xp.zeros(p_fake.data.shape[0], dtype=np.int32)))

		return loss

	def train_generator_x_z(self, x, noise=None):
		loss = self.loss_generator_x_z(x, noise=noise)

		self.zero_grads()
		loss.backward()
		self.update_generator()

		if self.gpu_enabled:
			loss.to_cpu()

		return float(loss.data)

	def loss_discriminator_z(self, x, z_true, noise=None):
		xp = self.xp

		# z_true came from true prior distribution
		p_true = self.discriminator_z(z_true, softmax=False)

		# 0: Samples from true distribution
		# 1: Samples from generator
		loss_true = F.softmax_cross_entropy(p_true, Variable(xp.zeros(p_true.data.shape[0], dtype=np.int32)))

		# z_fake was generated by generator
		if isinstance(self.generator_x_z, UniversalApproximatorGenerator):
			z_fake = self.generator_x_z(x, test=False, apply_f=True, noise=noise)
		else:
			z_fake = self.generator_x_z(x, test=False, apply_f=True)
		p_fake = self.discriminator_z(z_fake, softmax=False)
		loss_fake = F.softmax_cross_entropy(p_fake, Variable(xp.ones(p_fake.data.shape[0], dtype=np.int32)))

		return loss_true + loss_fake

	def train_discriminator_z(self, x, z_true, noise=None):
		loss = self.loss_discriminator_z(x, z_true, noise=noise)

		self.zero_grads()
		loss.backward()
		self.update_discriminator()

		if self.gpu_enabled:
			loss.to_cpu()

		return float(loss.data) / 2.0

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				filename = dir + "/{}_{}.hdf5".format(self.name, attr)
				if os.path.isfile(filename):
					print "loading",  filename
					serializers.load_hdf5(filename, prop)
				else:
					print filename, "missing."
		print "model loaded."

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				serializers.save_hdf5(dir + "/{}_{}.hdf5".format(self.name, attr), prop)
		print "model saved."


class MultiLayerPerceptron(chainer.Chain):
	def __init__(self, **layers):
		super(MultiLayerPerceptron, self).__init__(**layers)
		self.activation_function = "softplus"
		self.apply_batchnorm_to_input = False
		self.apply_batchnorm = False
		self.apply_dropout = False
		self.batchnorm_before_activation = True

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		for i in range(self.n_layers):
			u = chain[-1]
			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			elif i == self.n_layers - 1:
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

	def __call__(self, x, test=False, apply_f=True):
		return self.forward_one_step(x, test=test)

class SoftmaxClassifier(MultiLayerPerceptron):

	def __call__(self, x, test=False, softmax=True):
		output = self.forward_one_step(x, test=test)
		if softmax:
			return F.softmax(output)
		return output

class GaussianGenerator(MultiLayerPerceptron):

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def forward_one_step(self, x, test=False):
		f = activations[self.activation_function]
		chain = [x]

		for i in range(self.n_layers):
			u = chain[-1]
			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)
			if i == 0:
				if self.apply_batchnorm_to_input:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			elif i == self.n_layers - 1:
				if self.apply_batchnorm and self.batchnorm_before_activation == False:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_%d" % i)(u, test=test)
			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)
			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		u = chain[-1]
		mean = self.layer_output_mean(u)

		# log(sd^2)
		u = chain[-1]
		ln_var = self.layer_output_var(u)

		return mean, ln_var

	def __call__(self, x, test=False, apply_f=True):
		mean, ln_var = self.forward_one_step(x, test=test)
		if apply_f:
			return F.gaussian(mean, ln_var)
		return mean, ln_var

class UniversalApproximatorGenerator(MultiLayerPerceptron):

	def __call__(self, x, test=False, apply_f=True, noise=None):
		# z = self.forward_one_step(x, test=test)
		# if apply_f:
		# 	if isinstance(noise, Variable):
		# 		return z + noise
		# 	else:
		# 		raise Exception("Invalid input noise")
		# return z
		raise Exception("not supported")

class BernoulliDecoder(MultiLayerPerceptron):

	def __call__(self, z, test=False, apply_f=False):
		output = self.forward_one_step(z, test=test)
		if apply_f:
			return F.sigmoid(output)
		return output