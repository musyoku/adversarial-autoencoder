import chainer

def get_weight_initializer(weight_initializer, weight_std):
	assert weight_initializer is not None
	if weight_initializer.lower() == "normal":
		return chainer.initializers.Normal(weight_std)
	if weight_initializer.lower() == "glorotnormal":
		return chainer.initializers.GlorotNormal(weight_std)
	if weight_initializer.lower() == "henormal":
		return chainer.initializers.HeNormal(weight_std)
	raise Exception()

def get_optimizer(name, lr, momentum=0.9):
	if name.lower() == "adam":
		return chainer.optimizers.Adam(alpha=lr, beta1=momentum)
	if name.lower() == "eve":
		return Eve(alpha=lr, beta1=momentum)
	if name.lower() == "adagrad":
		return chainer.optimizers.AdaGrad(lr=lr)
	if name.lower() == "adadelta":
		return chainer.optimizers.AdaDelta(rho=momentum)
	if name.lower() == "nesterov" or name.lower() == "nesterovag":
		return chainer.optimizers.NesterovAG(lr=lr, momentum=momentum)
	if name.lower() == "rmsprop":
		return chainer.optimizers.RMSprop(lr=lr, alpha=momentum)
	if name.lower() == "momentumsgd":
		return chainer.optimizers.MomentumSGD(lr=lr, mommentum=mommentum)
	if name.lower() == "sgd":
		return chainer.optimizers.SGD(lr=lr)
	raise Exception()


def get_conv_outsize(in_size, ksize, stride, padding, cover_all=False, d=1):
	dk = ksize + (ksize - 1) * (d - 1)
	if cover_all:
		return (in_size + padding * 2 - dk + stride - 1) // stride + 1
	else:
		return (in_size + padding * 2 - dk) // stride + 1

def get_conv_padding(in_size, ksize, stride):
	pad2 = stride - (in_size - ksize) % stride
	if pad2 % stride == 0:
		return 0
	if pad2 % 2 == 1:
		return pad2
	return pad2 / 2

def get_deconv_padding(in_size, out_size, ksize, stride, cover_all=False):
	if cover_all:
		return (stride * (in_size - 1) + ksize - stride + 1 - out_size) // 2
	else:
		return (stride * (in_size - 1) + ksize - out_size) // 2

def get_deconv_outsize(in_size, ksize, stride, padding, cover_all=False):
	if cover_all:
		return stride * (in_size - 1) + ksize - stride + 1 - 2 * padding
	else:
		return stride * (in_size - 1) + ksize - 2 * padding

def get_deconv_insize(out_size, ksize, stride, padding, cover_all=False):
	if cover_all:
		return (out_size - ksize + stride - 1 + 2 * padding) // stride + 1
	else:
		return (out_size - ksize + 2 * padding) // stride + 1

def get_paddings_of_deconv_layers(out_size, num_layers, ksize, stride):
	# compute required deconv paddings
	paddings = []
	deconv_out_sizes = [out_size]
	for i in xrange(num_layers):
		deconv_out_sizes.append(get_conv_outsize(deconv_out_sizes[-1], ksize, stride, get_conv_padding(deconv_out_sizes[-1], ksize, stride)))

	# out_size of hidden layer must be a multiple of stride
	for i, size in enumerate(deconv_out_sizes[1:]):
		if size % stride != 0:
			deconv_out_sizes[i + 1] = size + size % stride

	deconv_out_sizes.reverse()
	paddings = []
	for i, (in_size, out_size) in enumerate(zip(deconv_out_sizes[:-1], deconv_out_sizes[1:])):
		paddings.append(get_deconv_padding(in_size, out_size, ksize, stride))

	return paddings

def get_in_size_of_deconv_layers(out_size, num_layers, ksize, stride):
	# compute required deconv paddings
	paddings = []
	deconv_out_sizes = [out_size]
	for i in xrange(num_layers):
		deconv_out_sizes.append(get_conv_outsize(deconv_out_sizes[-1], ksize, stride, get_conv_padding(deconv_out_sizes[-1], ksize, stride)))
	return deconv_out_sizes[-1]