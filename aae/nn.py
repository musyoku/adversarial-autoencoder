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
		self.__layers__ = layers

	def __call__(self, x):
		for layer in self.__layers__:
			x = layer(x)
		return x

# Chain

class Module(chainer.Chain):
	def __init__(self, *layers):
		super(Module, self).__init__()
		self.__module_name__ = None
		self.__layers__ = []
		self.__links__ = []
		self.__modules__ = []
		self.__parent_module__ = None
		if len(layers) > 0:
			self.add(*layers)

	def add(self, *layers):
		with self.init_scope():
			for i, layer in enumerate(layers):
				index = i + len(self.__layers__)

				if isinstance(layer, chainer.Link):
					setattr(self, "_nn_layer_%d" % index, layer)

				if isinstance(layer, Residual):
					for _index, _layer in enumerate(layer.__layers__):
						if isinstance(_layer, chainer.Link):
							setattr(self, "_nn_layer_{}_res_{}".format(index, _index), _layer)
		self.__layers__ += layers

	def __setattr__(self, name, value):
		assert isinstance(value, Residual) is False

		if isinstance(value, Module):
			self.__module_name__ = name

			self.__modules__.append((name, value))
			value.set_parent_module(self)

			self.update_params()
			return super(chainer.Link, self).__setattr__(name, value)	# prevent module from being added to self._children

		if isinstance(value, chainer.Link):
			if name.startswith("_nn_layer_"):
				return self.super__setattr__(name, value)

			self.__links__.append((name, value))

			self.update_params()
			
			with self.init_scope():
				return self.super__setattr__(name, value)

		super(Module, self).__setattr__(name, value)

	def update_params(self):
		for index, (module_name, module) in enumerate(self.__modules__):
			self.set_submodule(module_name, module)

		if self.__parent_module__ is not None:
			self.__parent_module__.update_params()

	def set_submodule(self, namespace, module):
		assert isinstance(module, Module)

		self.set_submodule_layers(namespace, module)
		self.set_submodule_links(namespace, module)

		for index, (module_name, module) in enumerate(module.__modules__):
			assert isinstance(module, Module)
			self.set_submodule("{}_{}".format(namespace, module_name), module)

	def set_submodule_layers(self, namespace, module):
		with self.init_scope():
			for index, layer in enumerate(module.__layers__):
				if isinstance(layer, chainer.Link):
					self.super__setattr__("_nn_{}_layer_{}".format(namespace, index), layer)

				if isinstance(layer, Residual):
					for resnet_index, _layer in enumerate(layer.__layers__):
						if isinstance(_layer, chainer.Link):
							self.super__setattr__("_nn_{}_layer_{}_res_{}".format(namespace, index, resnet_index), _layer)

	def set_submodule_links(self, namespace, module):
		with self.init_scope():
			for index, (link_name, link) in enumerate(module.__links__):
				assert isinstance(link, chainer.Link)
				self.super__setattr__("_nn_{}_link_{}".format(namespace, link_name), link)

	def super__setattr__(self, name, value):
		if name in dir(self):
			return
		super(Module, self).__setattr__(name, value)

	def set_parent_module(self, module):
		super(Module, self).__setattr__("__parent_module__", module)

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
		for layer in self.__layers__:
			y = layer(x)
			if isinstance(layer, Residual) and x.shape == y.shape:
				y += x
			x = y
		return x