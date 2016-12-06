import numpy as np
import os, time, math, collections, six
import chainer
from chainer import optimizers, serializers, Variable
from chainer import cuda
from chainer import optimizer
import sequential
import links

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
		if norm < self.threshold:
			return
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad *= rate

class Eve(optimizer.GradientMethod):
	def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, beta3=0.999, eps=1e-8, lower_threshold=0.1, upper_threshold=10):
		self.alpha = alpha
		self.beta1 = beta1
		self.beta2 = beta2
		self.beta3 = beta3
		self.eps = eps
		self.lower_threshold = lower_threshold
		self.upper_threshold = upper_threshold

	def init_state(self, param, state):
		xp = cuda.get_array_module(param.data)
		with cuda.get_device(param.data):
			state['m'] = xp.zeros_like(param.data)
			state['v'] = xp.zeros_like(param.data)
			state['d'] = xp.ones(1, dtype=param.data.dtype)
			state['f'] = xp.zeros(1, dtype=param.data.dtype)

	def _update_d_and_f(self, state):
		d, f = state['d'], state['f']
		if self.t > 1:
			old_f = float(cuda.to_cpu(state['f']))
			if self.loss > old_f:
				delta = self.lower_threshold + 1.
				Delta = self.upper_threshold + 1.
			else:
				delta = 1. / (self.upper_threshold + 1.)
				Delta = 1. / (self.lower_threshold + 1.)
			c = min(max(delta, self.loss / (old_f + 1e-12)), Delta)
			new_f = c * old_f
			r = abs(new_f - old_f) / (min(new_f, old_f) + 1e-12)
			d += (1 - self.beta3) * (r - d)
			f[:] = new_f
		else:
			f[:] = self.loss

	def update_one_cpu(self, param, state):
		m, v, d = state['m'], state['v'], state['d']
		grad = param.grad

		self._update_d_and_f(state)
		m += (1. - self.beta1) * (grad - m)
		v += (1. - self.beta2) * (grad * grad - v)
		param.data -= self.lr * m / (d * np.sqrt(v) + self.eps)

	def update_one_gpu(self, param, state):
		self._update_d_and_f(state)
		cuda.elementwise(
			'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps, T d',
			'T param, T m, T v',
			'''m += one_minus_beta1 * (grad - m);
			   v += one_minus_beta2 * (grad * grad - v);
			   param -= lr * m / (d * sqrt(v) + eps);''',
			'eve')(param.grad, self.lr, 1 - self.beta1, 1 - self.beta2,
				   self.eps, float(state['d']), param.data, state['m'],
				   state['v'])

	@property
	def lr(self):
		fix1 = 1. - self.beta1 ** self.t
		fix2 = 1. - self.beta2 ** self.t
		return self.alpha * math.sqrt(fix2) / fix1

	def update(self, lossfun=None, *args, **kwds):
		# Overwrites GradientMethod.update in order to get loss values
		if lossfun is None:
			raise RuntimeError('Eve.update requires lossfun to be specified')
		loss_var = lossfun(*args, **kwds)
		self.loss = float(loss_var.data)
		super(Eve, self).update(lossfun=lambda: loss_var)

def get_optimizer(name, lr, momentum=0.9):
	if name.lower() == "adam":
		return optimizers.Adam(alpha=lr, beta1=momentum)
	if name.lower() == "eve":
		return Eve(alpha=lr, beta1=momentum)
	if name.lower() == "adagrad":
		return optimizers.AdaGrad(lr=lr)
	if name.lower() == "adadelta":
		return optimizers.AdaDelta(rho=momentum)
	if name.lower() == "nesterov" or name.lower() == "nesterovag":
		return optimizers.NesterovAG(lr=lr, momentum=momentum)
	if name.lower() == "rmsprop":
		return optimizers.RMSprop(lr=lr, alpha=momentum)
	if name.lower() == "momentumsgd":
		return optimizers.MomentumSGD(lr=lr, mommentum=mommentum)
	if name.lower() == "sgd":
		return optimizers.SGD(lr=lr)

class Chain(chainer.Chain):

	def add_sequence(self, sequence):
		self.add_sequence_with_name(sequence)
		self.sequence = sequence

	def add_sequence_with_name(self, sequence, name="link"):
		if isinstance(sequence, sequential.Sequential) == False:
			raise Exception()
		for i, link in enumerate(sequence.links):
			if isinstance(link, chainer.link.Link):
				self.add_link("{}_{}".format(name, i), link)
			elif isinstance(link, links.Gaussian):
				self.add_link("{}_{}_ln_var".format(name, i), link.layer_ln_var)
				self.add_link("{}_{}_mean".format(name, i), link.layer_mean)
			elif isinstance(link, links.MinibatchDiscrimination):
				self.add_link("{}_{}".format(name, i), link.T)
			elif isinstance(link, links.Merge):
				for l, layer in enumerate(link.merge_layers):
					self.add_link("{}_{}_{}".format(name, i, l), layer)

	def load(self, filename):
		if os.path.isfile(filename):
			print "loading {} ...".format(filename)
			serializers.load_hdf5(filename, self)
		else:
			pass
			# print filename, "not found."

	def save(self, filename):
		if os.path.isfile(filename):
			os.remove(filename)
		serializers.save_hdf5(filename, self)

	def setup_optimizers(self, optimizer_name, lr, momentum=0.9, weight_decay=0, gradient_clipping=0):
		opt = get_optimizer(optimizer_name, lr, momentum)
		opt.use_cleargrads()
		opt.setup(self)
		if weight_decay > 0:
			opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))
		if gradient_clipping > 0:
			opt.add_hook(GradientClipping(gradient_clipping))
		self.optimizer = opt

	def update_learning_rate(self, lr):
		if isinstance(self.optimizer, optimizers.Adam):
			self.optimizer.alpha = lr
			return
		if isinstance(self.optimizer, Eve):
			self.optimizer.alpha = lr
			return
		if isinstance(self.optimizer, optimizers.AdaDelta):
			# AdaDelta has no learning rate
			return
		self.optimizer.lr = lr

	def update_momentum(self, momentum):
		if isinstance(self.optimizer, optimizers.Adam):
			self.optimizer.beta1 = momentum
			return
		if isinstance(self.optimizer, Eve):
			self.optimizer.beta1 = momentum
			return
		if isinstance(self.optimizer, optimizers.AdaDelta):
			self.optimizer.rho = momentum
			return
		if isinstance(self.optimizer, optimizers.NesterovAG):
			self.optimizer.momentum = momentum
			return
		if isinstance(self.optimizer, optimizers.RMSprop):
			self.optimizer.alpha = momentum
			return
		if isinstance(self.optimizer, optimizers.MomentumSGD):
			self.optimizer.mommentum = momentum
			return

	def backprop(self, loss):
		# self.optimizer.zero_grads()
		# loss.backward()
		# if isinstance(self.optimizer, Eve):
		# 	self.optimizer.update(loss)
		# else:
		# 	self.optimizer.update()
		if isinstance(loss, Variable):
			self.optimizer.update(lossfun=lambda: loss)
		else:
			self.optimizer.update(lossfun=loss)

	def __call__(self, *args, **kwargs):
		return self.sequence(*args, **kwargs)
