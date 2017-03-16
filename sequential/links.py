import chainer
from chainer import functions as F
from chainer import initializers

class Link(object):
	pass

class Gaussian(Link):
	def __init__(self, layer_mean, layer_ln_var):
		self.layer_mean = layer_mean
		self.layer_ln_var = layer_ln_var

	def __call__(self, x):
		return self.layer_mean(x), self.layer_ln_var(x)

class MinibatchDiscrimination(Link):
	def __init__(self, T, num_kernels=50, ndim_kernel=5, train_weights=True):
		self.T = T
		self.num_kernels = num_kernels
		self.ndim_kernel = ndim_kernel
		self.train_weights = train_weights
		self.initial_T = None

	def __call__(self, x):
		xp = chainer.cuda.get_array_module(x.data)
		batchsize = x.shape[0]
		if self.train_weights == False and self.initial_T is not None:
			self.T.W.data = self.initial_T

		M = F.reshape(self.T(x), (-1, self.num_kernels, self.ndim_kernel))
		M = F.expand_dims(M, 3)
		M_T = F.transpose(M, (3, 1, 2, 0))
		M, M_T = F.broadcast(M, M_T)

		norm = F.sum(abs(M - M_T), axis=2)
		eraser = F.broadcast_to(xp.eye(batchsize, dtype=x.dtype).reshape((batchsize, 1, batchsize)), norm.shape)
		c_b = F.exp(-(norm + 1e6 * eraser))
		o_b = F.sum(c_b, axis=2)

		if self.train_weights == False:
			self.initial_T = self.T.W.data

		return F.concat((x, o_b), axis=1)

class Merge(object):
	def __init__(self):
		self.merge_layers = []

	def append_layer(self, layer):
		self.merge_layers.append(layer)

	def __call__(self, *args):
		output = 0
		if len(args) != len(self.merge_layers):
			raise Exception()
		for i, data in enumerate(args):
			output += self.merge_layers[i](data)
		return output
		
class PixelShuffler2D(object):
	def __init__(self):
		self.conv = None
		self.r = 2

	def __call__(self, x):
		r = self.r
		out = self.conv(x)
		batchsize = out.shape[0]
		in_channels = out.shape[1]
		out_channels = in_channels / (r ** 2)
		in_height = out.shape[2]
		in_width = out.shape[3]
		out_height = in_height * r
		out_width = in_width * r
		out = F.reshape(out, (batchsize, 1, r * r, out_channels * in_height * in_width, 1))
		out = F.transpose(out, (0, 1, 3, 2, 4))
		out = F.reshape(out, (batchsize, out_channels, in_height, in_width, r, r))
		out = F.transpose(out, (0, 1, 2, 4, 3, 5))
		out = F.reshape(out, (batchsize, out_channels, out_height, out_width))
		return out
		
# class BatchRenormalization(link.Link):
# 	def __init__(self, size, decay=0.9, eps=2e-5, rmax=1, dmax=0, dtype=numpy.float32, use_gamma=True, use_beta=True, initial_gamma=None, initial_beta=None, use_cudnn=True):
# 		super(BatchNormalization, self).__init__(size, decay=decay, eps=eps, dtype=dtype, use_gamma=use_gamma, use_beta=use_beta, initial_gamma=initial_gamma, initial_beta=initial_beta, use_cudnn=use_cudnn)
# 		self.add_persistent("r", numpy.zeros(size, dtype=dtype))
# 		self.add_persistent("d", numpy.zeros(size, dtype=dtype))
# 		self.rmax = rmax
# 		self.dmax = dmax

# 	def __call__(self, x, test=False, finetune=False):
# 		if hasattr(self, "gamma"):
# 			gamma = self.gamma
# 		else:
# 			with cuda.get_device(self._device_id):
# 				gamma = variable.Variable(self.xp.ones(self.avg_mean.shape, dtype=x.dtype), volatile="auto")
# 		if hasattr(self, "beta"):
# 			beta = self.beta
# 		else:
# 			with cuda.get_device(self._device_id):
# 				beta = variable.Variable(self.xp.zeros(self.avg_mean.shape, dtype=x.dtype), volatile="auto")

# 		if not test:
# 			if finetune:
# 				self.N += 1
# 				decay = 1. - 1. / self.N
# 			else:
# 				decay = self.decay

# 			func = batch_normalization.BatchNormalizationFunction(
# 				self.eps, self.avg_mean, self.avg_var, True, decay,
# 				self.use_cudnn)
# 			ret = func(x, gamma, beta)

# 			self.avg_mean[:] = func.running_mean
# 			self.avg_var[:] = func.running_var
# 		else:
# 			# Use running average statistics or fine-tuned statistics.
# 			mean = variable.Variable(self.avg_mean, volatile="auto")
# 			var = variable.Variable(self.avg_var, volatile="auto")
# 			ret = batch_normalization.fixed_batch_normalization(
# 				x, gamma, beta, mean, var, self.eps, self.use_cudnn)
# 		return ret
