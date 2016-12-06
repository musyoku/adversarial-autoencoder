import math
import numpy as np
from chainer import cuda, Variable
from chainer import initializers
from chainer import link
from chainer import function
from chainer.utils import array
from chainer.utils import type_check
from chainer.functions.connection import linear

def _as_mat(x):
	if x.ndim == 2:
		return x
	return x.reshape(len(x), -1)

def get_norm(W):
	xp = cuda.get_array_module(W)
	norm = xp.sqrt(xp.sum(W ** 2, axis=1)) + 1e-9
	norm = norm.reshape((-1, 1))
	return norm

class LinearFunction(linear.LinearFunction):

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(3 <= n_in, n_in <= 4)
		x_type, w_type, g_type = in_types[:3]

		type_check.expect(
			x_type.dtype.kind == "f",
			w_type.dtype.kind == "f",
			g_type.dtype.kind == "f",
			x_type.ndim >= 2,
			w_type.ndim == 2,
			g_type.ndim == 2,
			type_check.prod(x_type.shape[1:]) == w_type.shape[1],
		)

		if n_in.eval() == 4:
			b_type = in_types[3]
			type_check.expect(
				b_type.dtype == x_type.dtype,
				b_type.ndim == 1,
				b_type.shape[0] == w_type.shape[0],
			)

	def forward(self, inputs):
		x = _as_mat(inputs[0])
		V = inputs[1]
		g = inputs[2]
		xp = cuda.get_array_module(V)

		self.normV = get_norm(V)
		self.normalizedV = V / self.normV
		self.W = g * self.normalizedV

		y = x.dot(self.W.T).astype(x.dtype, copy=False)
		if len(inputs) == 4:
			b = inputs[3]
			y += b
		return y,

	def backward(self, inputs, grad_outputs):
		x = _as_mat(inputs[0])
		V = inputs[1]
		g = inputs[2]
		W = self.W
		xp = cuda.get_array_module(x)

		gy = grad_outputs[0]
		gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
		gW = gy.T.dot(x).astype(W.dtype, copy=False)

		gg = xp.sum(gW * self.normalizedV, axis=1, keepdims=True).astype(g.dtype, copy=False)
		gV = g * (gW - gg * self.normalizedV) / self.normV
		gV = gV.astype(V.dtype, copy=False)

		if len(inputs) == 4:
			gb = gy.sum(0)
			return gx, gV, gg, gb
		else:
			return gx, gV, gg

def linear(x, V, g, b=None):
	if b is None:
		return LinearFunction()(x, V, g)
	else:
		return LinearFunction()(x, V, g, b)

class Linear(link.Link):

	def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False, initialV=None, dtype=np.float32):
		super(Linear, self).__init__()

		self.initialV = initialV
		self.wscale = wscale
		self.nobias = nobias
		self.dtype = dtype
		self.out_size = out_size

		if in_size is None:
			self.add_uninitialized_param("V")
		else:
			self._initialize_weight(in_size)

		if nobias:
			self.b = None
		else:
			self.add_uninitialized_param("b")

		self.add_uninitialized_param("g")

	def _initialize_weight(self, in_size):
		self.add_param("V", (self.out_size, in_size), initializer=initializers._get_initializer(self.initialV, math.sqrt(self.wscale)))

	def _initialize_params(self, t):
		xp = cuda.get_array_module(t)
		self.mean_t = xp.mean(t, axis=0)
		self.std_t = xp.sqrt(xp.var(t, axis=0))
		g = 1 / self.std_t
		b = -self.mean_t / self.std_t

		# print "g <- {}, b <- {}".format(g, b)

		if self.nobias == False:
			self.add_param("b", self.out_size, initializer=initializers.Constant(b, self.dtype))
		self.add_param("g", (self.out_size, 1), initializer=initializers.Constant(g.reshape(-1, 1), self.dtype))
		
	def _get_W_data(self):
		V = self.V.data
		xp = cuda.get_array_module(V)
		norm = get_norm(V)
		V = V / norm
		return self.g.data * V

	def __call__(self, x):
		if hasattr(self, "V") == False:
			with cuda.get_device(self._device_id):
				self._initialize_weight(x.size // len(x.data))

		if hasattr(self, "b") == False or hasattr(self, "g") == False:
			xp = cuda.get_array_module(x.data)
			t = linear(x, self.V, Variable(xp.full((self.out_size, 1), 1.0).astype(x.dtype)))	# compute output with g = 1 and without bias
			self._initialize_params(t.data)
			return (t - self.mean_t) / self.std_t

		return linear(x, self.V, self.g, self.b)