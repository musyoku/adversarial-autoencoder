# -*- coding: utf-8 -*-
import math
import numpy as np
from six import moves
from chainer import cuda, Variable, initializers, link, function
from chainer.utils import conv, type_check
from chainer.functions.connection import deconvolution_2d, convolution_2d

if cuda.cudnn_enabled:
	cudnn = cuda.cudnn
	libcudnn = cuda.cudnn.cudnn
	_cudnn_version = libcudnn.getVersion()
	_fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
	if _cudnn_version >= 4000:
		_bwd_filter_pref = \
			libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
		_bwd_data_pref = \
			libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT

_check_cudnn_acceptable_type = convolution_2d._check_cudnn_acceptable_type

def get_norm(W):
	xp = cuda.get_array_module(W)
	norm = xp.sqrt(xp.sum(W ** 2, axis=(0, 2, 3))) + 1e-9
	norm = norm.reshape((1, -1, 1, 1))
	return norm

def _pair(x):
	if hasattr(x, "__getitem__"):
		return x
	return x, x

class Deconvolution2DFunction(deconvolution_2d.Deconvolution2DFunction):

	def __init__(self, stride=1, pad=0, outsize=None, use_cudnn=True):
		self.sy, self.sx = _pair(stride)
		self.ph, self.pw = _pair(pad)
		self.use_cudnn = use_cudnn
		self.outh, self.outw = (None, None) if outsize is None else outsize

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(3 <= n_in, n_in <= 4)
		x_type = in_types[0]
		v_type = in_types[1]
		g_type = in_types[2]

		type_check.expect(
			x_type.dtype.kind == "f",
			v_type.dtype.kind == "f",
			g_type.dtype.kind == "f",
			x_type.ndim == 4,
			v_type.ndim == 4,
			g_type.ndim == 4,
			x_type.shape[1] == v_type.shape[0]
		)

		if self.outh is not None:
			type_check.expect(
				x_type.shape[2] ==
				conv.get_conv_outsize(self.outh, v_type.shape[2],self.sy, self.ph),
			)
		if self.outw is not None:
			type_check.expect(
				x_type.shape[3] ==
				conv.get_conv_outsize(self.outw, v_type.shape[3], self.sx, self.pw),
			)

		if n_in.eval() == 4:
			b_type = in_types[3]
			type_check.expect(
				b_type.dtype == x_type.dtype,
				b_type.ndim == 1,
				b_type.shape[0] == v_type.shape[1]
			)

	def forward_cpu(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		
		self.normV = get_norm(V)
		self.normalizedV = V / self.normV
		self.W = g * self.normalizedV

		if b is None:
			return super(Deconvolution2DFunction, self).forward_cpu((x, self.W))
		return super(Deconvolution2DFunction, self).forward_cpu((x, self.W, b))

	def forward_gpu(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None

		self.normV = get_norm(V)
		self.normalizedV = V / self.normV
		self.W = g * self.normalizedV
		
		if b is None:
			return super(Deconvolution2DFunction, self).forward_gpu((x, self.W))
		return super(Deconvolution2DFunction, self).forward_gpu((x, self.W, b))

	def backward_cpu(self, inputs, grad_outputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		if b is None:
			gb = None
			gx, gW = super(Deconvolution2DFunction, self).backward_cpu((x, self.W), grad_outputs)
		else:
			gx, gW, gb = super(Deconvolution2DFunction, self).backward_cpu((x, self.W, b), grad_outputs)

		xp = cuda.get_array_module(x)
		gg = xp.sum(gW * self.normalizedV, axis=(0, 2, 3), keepdims=True).astype(g.dtype, copy=False)
		gV = g * (gW - gg * self.normalizedV) / self.normV
		gV = gV.astype(V.dtype, copy=False)

		if b is None:
			return gx, gV, gg
		else:
			return gx, gV, gg, gb

	def backward_gpu(self, inputs, grad_outputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		if b is None:
			gb = None
			gx, gW = super(Deconvolution2DFunction, self).backward_gpu((x, self.W), grad_outputs)
		else:
			gx, gW, gb = super(Deconvolution2DFunction, self).backward_gpu((x, self.W, b), grad_outputs)

		xp = cuda.get_array_module(x)
		gg = xp.sum(gW * self.normalizedV, axis=(0, 2, 3), keepdims=True).astype(g.dtype, copy=False)
		gV = g * (gW - gg * self.normalizedV) / self.normV
		gV = gV.astype(V.dtype, copy=False)

		if b is None:
			return gx, gV, gg
		else:
			return gx, gV, gg, gb


def deconvolution_2d(x, V, g, b=None, stride=1, pad=0, outsize=None, use_cudnn=True):
	func = Deconvolution2DFunction(stride, pad, outsize, use_cudnn)
	if b is None:
		return func(x, V, g)
	else:
		return func(x, V, g, b)

class Deconvolution2D(link.Link):

	def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
				 wscale=1, bias=0, nobias=False, outsize=None, use_cudnn=True,
				 initialV=None, dtype=np.float32):
		kh, kw = _pair(ksize)
		self.stride = _pair(stride)
		self.pad = _pair(pad)
		self.outsize = (None, None) if outsize is None else outsize
		self.use_cudnn = use_cudnn
		self.dtype = dtype
		self.nobias = nobias
		self.out_channels = out_channels
		self.in_channels = in_channels

		V_shape = (in_channels, out_channels, kh, kw)
		super(Deconvolution2D, self).__init__(V=V_shape)

		if isinstance(initialV, (np.ndarray, cuda.ndarray)):
			assert initialV.shape == (in_channels, out_channels, kh, kw)
		initializers.init_weight(self.V.data, initialV, scale=math.sqrt(wscale))

		if nobias:
			self.b = None
		else:
			self.add_uninitialized_param("b")
		self.add_uninitialized_param("g")

	def _initialize_params(self, t):
		xp = cuda.get_array_module(t)
		# 出力チャネルごとにミニバッチ平均をとる
		self.mean_t = xp.mean(t, axis=(0, 2, 3)).reshape((1, -1, 1, 1))
		self.std_t = xp.sqrt(xp.var(t, axis=(0, 2, 3))).reshape((1, -1, 1, 1))
		g = 1 / self.std_t
		b = -self.mean_t / self.std_t

		# print "g <- {}, b <- {}".format(g.reshape((-1,)), b.reshape((-1,)))

		if self.nobias == False:
			self.add_param("b", self.out_channels, initializer=initializers.Constant(b.reshape((-1, )), self.dtype))
		self.add_param("g", (1, self.out_channels, 1, 1), initializer=initializers.Constant(g, self.dtype))

	def _get_W_data(self):
		V = self.V.data
		xp = cuda.get_array_module(V)
		norm = xp.linalg.norm(V)
		V = V / norm
		return self.g.data * V

	def __call__(self, x):

		if hasattr(self, "b") == False or hasattr(self, "g") == False:
			xp = cuda.get_array_module(x.data)
			t = deconvolution_2d(x, self.V, Variable(xp.full((1, self.out_channels, 1, 1), 1.0).astype(x.dtype)), None, self.stride, self.pad, self.outsize, self.use_cudnn)	# compute output with g = 1 and without bias
			self._initialize_params(t.data)
			return (t - self.mean_t) / self.std_t

		return deconvolution_2d(
			x, self.V, self.g, self.b, self.stride, self.pad,
			self.outsize, self.use_cudnn)
