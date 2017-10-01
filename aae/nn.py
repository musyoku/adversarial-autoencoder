import chainer, math, os, uuid
from chainer import functions, cuda, serializers, links
from chainer.links import *

# Standar functions

class ClippedReLU():
	def __init__(self, z=20):
		self.z = z

	def __call__(self, x):
		return functions.clipped_relu(x, self.z)

class CReLU():
	def __init__(self, axis=1):
		self.axis = axis

	def __call__(self, x):
		return functions.crelu(x, self.axis)

class ELU():
	def __init__(self, alpha=1):
		self.alpha = alpha

	def __call__(self, x):
		return functions.elu(x, self.alpha)
	
def HardSigmoid():
	return functions.hard_sigmoid

class LeakyReLU():
	def __init__(self, slope=1):
		self.slope = slope

	def __call__(self, x):
		return functions.leaky_relu(x, self.slope)
	
def LogSoftmax():
	return functions.log_softmax

class Maxout():
	def __init__(self, pool_size=0.5):
		self.pool_size = pool_size

	def __call__(self, x):
		return functions.maxout(x, self.pool_size)
	
def ReLU():
	return functions.relu

def Sigmoid():
	return functions.sigmoid

class Softmax():
	def __init__(self, axis=1):
		self.axis = axis

	def __call__(self, x):
		return functions.softmax(x, self.axis)

class Softplus():
	def __init__(self, beta=1):
		self.beta = beta

	def __call__(self, x):
		return functions.softplus(x, self.beta)

def Tanh():
	return functions.tanh

# Pooling

class AveragePooling2D():
	def __init__(self, ksize, stride=None, pad=0):
		self.ksize = ksize
		self.stride = stride
		self.pad = pad

	def __call__(self, x):
		return functions.average_pooling_2d(x, self.ksize, self.stride, self.pad)

class AveragePoolingND():
	def __init__(self, ksize, stride=None, pad=0):
		self.ksize = ksize
		self.stride = stride
		self.pad = pad

	def __call__(self, x):
		return functions.average_pooling_nd(x, self.ksize, self.stride, self.pad)

class MaxPooling2D():
	def __init__(self, ksize, stride=None, pad=0, cover_all=True):
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.cover_all = cover_all

	def __call__(self, x):
		return functions.max_pooling_2d(x, self.ksize, self.stride, self.pad)

class MaxPoolingND():
	def __init__(self, ksize, stride=None, pad=0, cover_all=True):
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.cover_all = cover_all

	def __call__(self, x):
		return functions.max_pooling_nd(x, self.ksize, self.stride, self.pad)

class SpatialPyramidPooling2D():
	def __init__(self, pyramid_height, pooling_class):
		self.pyramid_height = pyramid_height
		self.pooling_class = pooling_class

	def __call__(self, x):
		return functions.spatial_pyramid_pooling_2d(x, self.pyramid_height, self.pooling_class)

class Unpooling2D():
	def __init__(self, ksize, stride=None, pad=0, outsize=None, cover_all=True):
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.outsize = outsize
		self.cover_all = cover_all

	def __call__(self, x):
		return functions.unpooling_2d(x, self.ksize, self.stride, self.pad, self.outsize, self.cover_all)

class UpSampling2D():
	def __init__(self, indexes, ksize, stride=None, pad=0, outsize=None, cover_all=True):
		self.indexes = indexes
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.outsize = outsize
		self.cover_all = cover_all

	def __call__(self, x):
		return functions.upsampling_2d(x, self.indexes, self.ksize, self.stride, self.pad, self.outsize, self.cover_all)

# Array manipulations

class BroadcastTo():
	def __init__(self, shape):
		self.shape = shape

	def __call__(self, x):
		return functions.broadcast_to(x, self.shape)

class ExpandDims():
	def __init__(self, axis):
		self.axis = axis

	def __call__(self, x):
		return functions.expand_dims(x, self.axis)

def Flatten():
	return functions.flatten

class Reshape():
	def __init__(self, shape):
		self.shape = shape

	def __call__(self, x):
		return functions.reshape(x, self.shape)

class RollAxis():
	def __init__(self, axis, start=0):
		self.axis = axis
		self.start = start

	def __call__(self, x):
		return functions.rollaxis(x, self.axis, self.start)

class Squeeze():
	def __init__(self, axis):
		self.axis = axis

	def __call__(self, x):
		return functions.squeeze(x, self.axis)

class SwapAxes():
	def __init__(self, axis1, axis2):
		self.axis1 = axis1
		self.axis2 = axis2

	def __call__(self, x):
		return functions.swapaxes(x, self.axis1, self.axis2)

class Tile():
	def __init__(self, reps):
		self.reps = reps

	def __call__(self, x):
		return functions.tile(x, self.reps)

class Transpose():
	def __init__(self, axes):
		self.axes = axes

	def __call__(self, x):
		return functions.transpose(x, self.axes)

# Noise injections

class Dropout():
	def __init__(self, ratio=0.5):
		self.ratio = ratio

	def __call__(self, x):
		return functions.dropout(x, self.ratio)

class GaussianNoise():
	def __init__(self, mean=0, std=1):
		self.mean = mean
		self.std = std

	def __call__(self, x):
		if chainer.config.train == False:
			return x
		data = x.data if isinstance(x, chainer.Variable) else x
		xp = cuda.get_array_module(data)
		ln_var = math.log(self.std ** 2)
		noise = functions.gaussian(xp.full_like(data, self.mean), xp.full_like(data, ln_var))
		return x + noise

# Connections

class Residual(object):
	def __init__(self, *layers):
		self.layers = layers

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

# Chain

class Module(chainer.Chain):
	def __init__(self, *layers):
		super(Module, self).__init__()
		self.layers = []
		self.links = []
		self.modules = []
		if len(layers) > 0:
			self.add(*layers)

	def add(self, *layers):
		with self.init_scope():
			for i, layer in enumerate(layers):
				index = i + len(self.layers)

				if isinstance(layer, chainer.Link):
					setattr(self, "_sequential_%d" % index, layer)

				if isinstance(layer, Residual):
					for _index, _layer in enumerate(layer.layers):
						if isinstance(_layer, chainer.Link):
							setattr(self, "_sequential_{}_{}".format(index, _index), _layer)
		self.layers += layers

	def __setattr__(self, name, value):
		if isinstance(value, Module):
			self.modules.append((name, value))
			self._set_module(name, value)
			return super(chainer.Link, self).__setattr__(name, value)	# prevent module from being added to self._children

		if isinstance(value, chainer.Link):
			with self.init_scope():
				if name.startswith("_sequential_"):
					return super(Module, self).__setattr__(name, value)
				self.links.append((name, value))
				return super(Module, self).__setattr__(name, value)

		super(Module, self).__setattr__(name, value)

	def _set_module(self, namespace, module):
		assert isinstance(module, Module)

		for index, layer in enumerate(module.layers):
			if isinstance(layer, chainer.Link):
				super(Module, self).__setattr__("_module_{}_sequential_{}".format(namespace, index), layer)

			if isinstance(layer, Residual):
				for _index, _layer in enumerate(layer.layers):
					if isinstance(_layer, chainer.Link):
						super(Module, self).__setattr__("_module_{}_sequential_{}_{}".format(namespace, index, _index), _layer)
		
		for index, (link_name, link) in enumerate(module.links):
			assert isinstance(link, chainer.Link)
			super(Module, self).__setattr__("_module_{}_link_{}".format(namespace, link_name), link)

		for index, (module_name, module) in enumerate(module.modules):
			assert isinstance(module, Module)
			self._set_module("{}_{}".format(namespace, module_name), module)

	def save(self, filename):
		tmp_filename = filename + "." + str(uuid.uuid4())
		serializers.save_hdf5(tmp_filename, self)
		if os.path.isfile(filename):
			os.remove(filename)
		os.rename(tmp_filename, filename)

	def load(self, filename):
		if os.path.isfile(filename):
			print("Loading {} ...".format(filename))
			serializers.load_hdf5(filename, self)
			return True
		return False

	def __call__(self, x):
		for layer in self.layers:
			y = layer(x)
			if isinstance(layer, Residual):
				y += x
			x = y
		return x