import copy, json, types
import chainer
import layers
import functions
from util import get_weight_initializer

class Sequential(object):
	def __init__(self, weight_initializer=None, weight_std=None):
		self.layers = []
		self.links = []
		self.built = False

		self.weight_initializer = weight_initializer	# Normal / GlorotNormal / HeNormal
		self.weight_std = weight_std

	def add(self, layer):
		if isinstance(layer, layers.Layer) or isinstance(layer, functions.Function):
			self.layers.append(layer)
		elif isinstance(layer, functions.Activation):
			self.layers.append(layer.to_function())
		elif isinstance(layer, Residual):
			self.layers.append(layer)
		else:
			raise Exception()

	def layer_from_dict(self, dictionary):
		if "_layer" in dictionary:
			if hasattr(layers, dictionary["_layer"]):
				args = self.dict_to_layer_init_args(dictionary)
				return getattr(layers, dictionary["_layer"])(**args)
		if "_function" in dictionary:
			if hasattr(functions, dictionary["_function"]):
				args = self.dict_to_layer_init_args(dictionary)
				return getattr(functions, dictionary["_function"])(**args)
		if "_residual" in dictionary:
			weight_initializer = self.weight_initializer
			weight_std = self.weight_std
			if dictionary["weight_initializer"] is not None:
				weight_initializer = dictionary["weight_initializer"]
			if dictionary["weight_std"] is not None:
				weight_std = dictionary["weight_std"]
			seq = Residual(weight_initializer=weight_initializer, weight_std=weight_std)
			seq.from_dict(dictionary)
			return seq
		raise Exception()

	def dict_to_layer_init_args(self, dictionary):
		args = copy.deepcopy(dictionary)
		remove_keys = []
		for key, value in args.iteritems():
			if key[0] == "_":
				remove_keys.append(key)
		for key in remove_keys:
			del args[key]
		return args

	def layer_to_chainer_link(self, layer):
		if hasattr(layer, "_layer"):
			if isinstance(layer, layers.GRU):
				layer._init = get_weight_initializer(self.weight_initializer, self.weight_std)
				layer._inner_init = get_weight_initializer(self.weight_initializer, self.weight_std)
			elif isinstance(layer, layers.LSTM):
				layer._lateral_init  = get_weight_initializer(self.weight_initializer, self.weight_std)
				layer._upward_init  = get_weight_initializer(self.weight_initializer, self.weight_std)
				layer._bias_init = get_weight_initializer(self.weight_initializer, self.weight_std)
				layer._forget_bias_init = get_weight_initializer(self.weight_initializer, self.weight_std)
			elif isinstance(layer, layers.StatelessLSTM):
				layer._lateral_init  = get_weight_initializer(self.weight_initializer, self.weight_std)
				layer._upward_init  = get_weight_initializer(self.weight_initializer, self.weight_std)
			elif isinstance(layer, layers.StatefulGRU):
				layer._init = get_weight_initializer(self.weight_initializer, self.weight_std)
				layer._inner_init = get_weight_initializer(self.weight_initializer, self.weight_std)
			elif isinstance(layer, layers.Gaussian):
				layer._initialW_mean = get_weight_initializer(self.weight_initializer, self.weight_std)
				layer._initialW_ln_var = get_weight_initializer(self.weight_initializer, self.weight_std)
			elif isinstance(layer, layers.Merge):
				for i in xrange(layer.num_inputs):
					setattr(layer, "_initialW_%d" % i, get_weight_initializer(self.weight_initializer, self.weight_std))
			else:
				layer._initialW = get_weight_initializer(self.weight_initializer, self.weight_std)
			return layer.to_link()
		if hasattr(layer, "_function"):
			return layer
		raise Exception()

	def build(self, new_weight_initializer=None, new_weight_std=None):
		self.links = []
		# overwrite initializer if needed
		if new_weight_initializer is not None:
			self.weight_initializer = new_weight_initializer
		if new_weight_std is not None:
			self.weight_std = new_weight_std
		# convert layers to Chainer Link objects
		for i, layer in enumerate(self.layers):
			if isinstance(layer, Residual):
				weight_initializer = self.weight_initializer
				weight_std = self.weight_std
				if layer.weight_initializer is not None:
					weight_initializer = layer.weight_initializer
				if layer.weight_std is not None:
					weight_std = layer.weight_std
				layer.build(new_weight_initializer=weight_initializer, new_weight_std=weight_std)
				self.links.append(layer)
			else:
				link = self.layer_to_chainer_link(layer)
				self.links.append(link)
		self.built = True

	def to_dict(self):
		layers = []
		for layer in self.layers:
			config = layer.to_dict()
			dictionary = {}
			if isinstance(layer, Residual):
				dictionary["_residual"] = True
			for key, value in config.iteritems():
				if isinstance(value, (int, float, str, bool, type(None), tuple, list, dict)):
					dictionary[key] = value
			layers.append(dictionary)
		return {
			"layers": layers,
			"weight_initializer": self.weight_initializer,
			"weight_std": self.weight_std
		}

	def to_json(self):
		result = self.to_dict()
		return json.dumps(result, sort_keys=True, indent=4, separators=(',', ': '))

	def from_json(self, str):
		self.links = []
		self.layers = []
		attributes = {}
		dict_array = json.loads(str)
		self.from_dict(dict_array)

	def from_dict(self, dictionary):
		weight_initializer = dictionary["weight_initializer"]
		weight_std = dictionary["weight_std"]
		for i, layer_dict in enumerate(dictionary["layers"]):
			layer = self.layer_from_dict(layer_dict)
			self.layers.append(layer)

	def __call__(self, *args, **kwargs):
		assert self.built == True
		x = None
		activations = []
		if "test" not in kwargs:
			kwargs["test"] = False
		for link in self.links:
			if isinstance(link, functions.dropout):
				x = link(args[0] if x is None else x, train=not kwargs["test"])
			elif isinstance(link, chainer.links.BatchNormalization):
				x = link(args[0] if x is None else x, test=kwargs["test"])
			elif isinstance(link, functions.gaussian_noise):
				x = link(args[0] if x is None else x, test=kwargs["test"])
			elif isinstance(link, Residual):
				y = link(args[0] if x is None else x, test=kwargs["test"])
				x = args[0] if x is None else x
				x = y + x
			else:
				if x is None:
					x = link(*args)
				else:
					x = link(x)
					if isinstance(link, functions.ActivationFunction):
						activations.append(x)
		if "return_activations" in kwargs and kwargs["return_activations"] == True:
			return x, activations
		return x

class Residual(Sequential):
	def __init__(self, weight_initializer=None, weight_std=None):
		super(Residual, self).__init__(weight_initializer=weight_initializer, weight_std=weight_std)
		self._residual = True
