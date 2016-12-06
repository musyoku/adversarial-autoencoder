import math
from chainer import cuda, Variable
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