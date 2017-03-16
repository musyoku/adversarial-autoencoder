import math
from chainer import cuda, Variable, function
from chainer import functions as F

class Function(object):

	def __call__(self, x):
		raise NotImplementedError()

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			dict[attr] = value
		return dict

class ActivationFunction(Function):
	pass

class Activation(object):
	def __init__(self, nonlinearity="relu"):
		self.nonlinearity = nonlinearity

	def to_function(self):
		if self.nonlinearity.lower() == "clipped_relu":
			return clipped_relu()
		if self.nonlinearity.lower() == "crelu":
			return crelu()
		if self.nonlinearity.lower() == "elu":
			return elu()
		if self.nonlinearity.lower() == "hard_sigmoid":
			return hard_sigmoid()
		if self.nonlinearity.lower() == "leaky_relu":
			return leaky_relu()
		if self.nonlinearity.lower() == "relu":
			return relu()
		if self.nonlinearity.lower() == "sigmoid":
			return sigmoid()
		if self.nonlinearity.lower() == "softmax":
			return softmax()
		if self.nonlinearity.lower() == "softplus":
			return softplus()
		if self.nonlinearity.lower() == "tanh":
			return tanh()
		raise NotImplementedError()

class clipped_relu(ActivationFunction):
	def __init__(self, z=20.0):
		self._function = "clipped_relu"
		self.z = z

	def __call__(self, x):
		return F.clipped_relu(x, self.z)

class crelu(ActivationFunction):
	def __init__(self, axis=1):
		self._function = "crelu"
		self.axis = axis

	def __call__(self, x):
		return F.crelu(x, self.axis)

class elu(ActivationFunction):
	def __init__(self, alpha=1.0):
		self._function = "elu"
		self.alpha = alpha

	def __call__(self, x):
		return F.elu(x, self.alpha)

class hard_sigmoid(ActivationFunction):
	def __init__(self):
		self._function = "hard_sigmoid"
		pass

	def __call__(self, x):
		return F.hard_sigmoid(x)

class leaky_relu(ActivationFunction):
	def __init__(self, slope=0.2):
		self._function = "leaky_relu"
		self.slope = slope

	def __call__(self, x):
		return F.leaky_relu(x, self.slope)

class log_softmax(Function):
	def __init__(self, use_cudnn=True):
		self._function = "log_softmax"
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.log_softmax(x, self.use_cudnn)

class maxout(Function):
	def __init__(self, pool_size, axis=1):
		self._function = "maxout"
		self.pool_size = pool_size
		self.axis = axis

	def __call__(self, x):
		return F.maxout(x, self.pool_size, self.axis)

class relu(ActivationFunction):
	def __init__(self, use_cudnn=True):
		self._function = "relu"
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.relu(x, self.use_cudnn)

class sigmoid(ActivationFunction):
	def __init__(self, use_cudnn=True):
		self._function = "sigmoid"
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.sigmoid(x, self.use_cudnn)

class softmax(Function):
	def __init__(self, use_cudnn=True):
		self._function = "softmax"
		self.use_cudnn = use_cudnn
		pass
	def __call__(self, x):
		return F.softmax(x, self.use_cudnn)

class softplus(ActivationFunction):
	def __init__(self, use_cudnn=True):
		self._function = "softplus"
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.softplus(x, self.use_cudnn)

class tanh(ActivationFunction):
	def __init__(self, use_cudnn=True):
		self._function = "tanh"
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.tanh(x, self.use_cudnn)

class dropout(Function):
	def __init__(self, ratio=0.5):
		self._function = "dropout"
		self.ratio = ratio

	def __call__(self, x, train=True):
		return F.dropout(x, self.ratio, train)

class gaussian_noise(Function):
	def __init__(self, std=0.3):
		self._function = "gaussian_noise"
		self.std = std

	def __call__(self, x, test=False):
		if test == True:
			return x
		xp = cuda.get_array_module(x.data)
		ln_var = math.log(self.std ** 2)
		noise = F.gaussian(Variable(xp.zeros_like(x.data)), Variable(xp.full_like(x.data, ln_var)))
		return x + noise

class average_pooling_2d(Function):
	def __init__(self, ksize, stride=None, pad=0, use_cudnn=True):
		self._function = "average_pooling_2d"
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.average_pooling_2d(x, self.ksize, self.stride, self.pad, self.use_cudnn)

class max_pooling_2d(Function):
	def __init__(self, ksize, stride=None, pad=0, cover_all=True, use_cudnn=True):
		self._function = "max_pooling_2d"
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.cover_all = cover_all
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.max_pooling_2d(x, self.ksize, self.stride, self.pad, self.cover_all, self.use_cudnn)

class spatial_pyramid_pooling_2d(Function):
	def __init__(self, pyramid_height, pooling_class, use_cudnn=True):
		self._function = "spatial_pyramid_pooling_2d"
		self.pyramid_height = pyramid_height
		self.pooling_class = pooling_class
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.spatial_pyramid_pooling_2d(x, self.pyramid_height, self.pooling_class, self.use_cudnn)

class unpooling_2d(Function):
	def __init__(self, ksize, stride=None, pad=0, outsize=None, cover_all=True):
		self._function = "unpooling_2d"
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.outsize = outsize
		self.cover_all = cover_all

	def __call__(self, x):
		return F.unpooling_2d(x, self.ksize, self.stride, self.pad, self.outsize, self.cover_all)

class unpooling_2d(Function):
	def __init__(self, ksize, stride=None, pad=0, outsize=None, cover_all=True):
		self._function = "unpooling_2d"
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.outsize = outsize
		self.cover_all = cover_all

	def __call__(self, x):
		return F.unpooling_2d(x, self.ksize, self.stride, self.pad, self.outsize, self.cover_all)

class reshape(Function):
	def __init__(self, shape):
		self._function = "reshape"
		self.shape = shape

	def __call__(self, x):
		return F.reshape(x, self.shape)

class reshape_1d(Function):
	def __init__(self):
		self._function = "reshape_1d"

	def __call__(self, x):
		batchsize = x.data.shape[0]
		return F.reshape(x, (batchsize, -1))
		
# class BatchRenormalizationFunction(function.Function):
# 	def forward(self, inputs):
# 		xp = cuda.get_array_module(*inputs)
# 		x, gamma, beta = inputs[:3]
# 		if self.train:
# 			if self.running_mean is None:
# 				self.running_mean = xp.zeros_like(gamma)
# 				self.running_var = xp.zeros_like(gamma)
# 			else:
# 				self.running_mean = xp.array(self.running_mean)
# 				self.running_var = xp.array(self.running_var)
# 		elif len(inputs) == 5:
# 			self.fixed_mean = inputs[3]
# 			self.fixed_var = inputs[4]

# 		# TODO(bkvogel): Check for float16 support again in next cuDNN version.
# 		if x[0].dtype == numpy.float16:
# 			# cuDNN v5 batch normalization does not seem to support float16.
# 			self.use_cudnn = False

# 		head_ndim = gamma.ndim + 1
# 		expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
# 		gamma = gamma[expander]
# 		beta = beta[expander]

# 		# cuDNN only supports these tensor dimensions because they are
# 		# the most commonly used. If there is a need to support other
# 		# dimensions with cuDNN, we could consider reshaping the input
# 		# into a 2-dim array with channels as second dim and m=<product
# 		# of all dimensions except the 2nd dimension> as the first
# 		# dimension.
# 		self.cudnn_dim_ok = x.ndim == 2 or x.ndim == 4

# 		cudnn_updated_running_stats = False
# 		if xp is not numpy and cuda.cudnn_enabled and self.use_cudnn and \
# 				self.cudnn_dim_ok and _cudnn_version >= 5000:
# 			if x.ndim == 4:
# 				# for convolutional layer
# 				self.mode = libcudnn.CUDNN_BATCHNORM_SPATIAL
# 			else:
# 				# for linear layer
# 				self.mode = libcudnn.CUDNN_BATCHNORM_PER_ACTIVATION

# 			x = cuda.cupy.ascontiguousarray(x)
# 			gamma = cuda.cupy.ascontiguousarray(gamma)
# 			beta = cuda.cupy.ascontiguousarray(beta)
# 			dtype = x.dtype
# 			handle = cudnn.get_handle()
# 			x_desc = cudnn.create_tensor_descriptor(_as4darray(x))
# 			derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
# 			libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
# 											  x_desc.value, self.mode)
# 			one = numpy.array(1, dtype=dtype).ctypes
# 			zero = numpy.array(0, dtype=dtype).ctypes
# 			y = cuda.cupy.empty_like(x)
# 			# Factor used in the moving average
# 			factor = 1 - self.decay

# 			if self.train:
# 				if self.mean_cache is None:
# 					# Output cache to speed up backward pass.
# 					self.mean_cache = xp.empty_like(gamma)
# 					# Output cache to speed up backward pass.
# 					self.var_cache = xp.empty_like(gamma)
# 				# Note: cuDNN computes the mini-batch mean and variance
# 				# internally. We can simply (optionally) pass
# 				# it the running-average mean and variance arrays.
# 				libcudnn.batchNormalizationForwardTraining(
# 					handle, self.mode, one.data, zero.data,
# 					x_desc.value, x.data.ptr, x_desc.value,
# 					y.data.ptr, derivedBnDesc.value, gamma.data.ptr,
# 					beta.data.ptr, factor, self.running_mean.data.ptr,
# 					self.running_var.data.ptr, self.eps,
# 					self.mean_cache.data.ptr, self.var_cache.data.ptr)

# 				print self.mean_cache
# 				print self.var_cache
# 				cudnn_updated_running_stats = True
# 			else:
# 				libcudnn.batchNormalizationForwardInference(
# 					handle, self.mode, one.data, zero.data,
# 					x_desc.value, x.data.ptr, x_desc.value, y.data.ptr,
# 					derivedBnDesc.value, gamma.data.ptr, beta.data.ptr,
# 					self.fixed_mean.data.ptr, self.fixed_var.data.ptr,
# 					self.eps)
# 		else:
# 			if self.train:
# 				axis = (0,) + tuple(range(head_ndim, x.ndim))
# 				mean = x.mean(axis=axis)
# 				var = x.var(axis=axis)
# 				print mean
# 				print var
# 				var += self.eps
# 			else:
# 				mean = self.fixed_mean
# 				var = self.fixed_var + self.eps
# 			self.std = xp.sqrt(var, dtype=var.dtype)
# 			if xp is numpy:
# 				self.x_hat = _xhat(x, mean, self.std, expander)
# 				y = gamma * self.x_hat
# 				y += beta
# 			else:
# 				self.x_hat, y = cuda.elementwise(
# 					'T x, T mean, T std, T gamma, T beta', 'T x_hat, T y',
# 					'''
# 					x_hat = (x - mean) / std;
# 					y = gamma * x_hat + beta;
# 					''',
# 					'bn_fwd')(x, mean[expander], self.std[expander], gamma,
# 							  beta)

# 		if self.train and (not cudnn_updated_running_stats):
# 			# Note: If in training mode, the cuDNN forward training function
# 			# will do this for us, so
# 			# only run following code if cuDNN was not used.
# 			# Update running statistics:
# 			m = x.size // gamma.size
# 			adjust = m / max(m - 1., 1.)  # unbiased estimation
# 			self.running_mean *= self.decay
# 			temp_ar = xp.array(mean)
# 			temp_ar *= (1 - self.decay)
# 			self.running_mean += temp_ar
# 			del temp_ar
# 			self.running_var *= self.decay
# 			temp_ar = xp.array(var)
# 			temp_ar *= (1 - self.decay) * adjust
# 			self.running_var += temp_ar
# 			del temp_ar
# 		return y,