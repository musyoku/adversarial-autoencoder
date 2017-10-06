import collections, six
import numpy as np
from chainer import optimizers, cuda

class Optimizer():
	def __init__(self, name, lr, momentum=0.9):
		self.optimizer = get_optimizer(name, lr, momentum)

	def setup(self, model):
		self.optimizer.setup(model)

	def get_learning_rate(self):
		return get_current_learning_rate(self.optimizer)

	def set_learning_rate(self, new_lr):
		set_learning_rate(self.optimizer, new_lr)

	def decrease_learning_rate(self, factor, final_value):
		decrease_learning_rate(self.optimizer, factor, final_value)

	def add_hook(self, hook):		
		self.optimizer.add_hook(hook)

	def update(self):
		self.optimizer.update()

def _sum_sqnorm(arr):
	sq_sum = collections.defaultdict(float)
	for x in arr:
		with cuda.get_device_from_array(x) as dev:
			x = x.ravel()
			s = x.dot(x)
			sq_sum[int(dev)] += s
	return sum([float(i) for i in six.itervalues(sq_sum)])

class GradientClipping(object):
	name = 'GradientClipping'

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = np.sqrt(_sum_sqnorm([p.grad for p in opt.target.params(False)]))
		if norm == 0:
			return
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params(False):
				grad = param.grad
				with cuda.get_device_from_array(grad):
					grad *= rate


def get_current_learning_rate(opt):
	if isinstance(opt, optimizers.NesterovAG):
		return opt.lr
	if isinstance(opt, optimizers.MomentumSGD):
		return opt.lr
	if isinstance(opt, optimizers.SGD):
		return opt.lr
	if isinstance(opt, optimizers.Adam):
		return opt.alpha
	raise NotImplementedError()

def set_learning_rate(opt, lr):
	if isinstance(opt, optimizers.NesterovAG):
		opt.lr = lr
		return
	if isinstance(opt, optimizers.MomentumSGD):
		opt.lr = lr
		return
	if isinstance(opt, optimizers.SGD):
		opt.lr = lr
		return
	if isinstance(opt, optimizers.Adam):
		opt.alpha = lr
		return
	raise NotImplementedError()

def get_optimizer(name, lr, momentum):
	name = name.lower()
	if name == "sgd":
		return optimizers.SGD(lr=lr)
	if name == "msgd":
		return optimizers.MomentumSGD(lr=lr, momentum=momentum)
	if name == "nesterov":
		return optimizers.NesterovAG(lr=lr, momentum=momentum)
	if name == "adam":
		return optimizers.Adam(alpha=lr, beta1=momentum)
	raise NotImplementedError()

def decrease_learning_rate(opt, factor, final_value):
	if isinstance(opt, optimizers.NesterovAG):
		if opt.lr <= final_value:
			return final_value
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.SGD):
		if opt.lr <= final_value:
			return final_value
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.MomentumSGD):
		if opt.lr <= final_value:
			return final_value
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.Adam):
		if opt.alpha <= final_value:
			return final_value
		opt.alpha *= factor
		return
	raise NotImplementedError()