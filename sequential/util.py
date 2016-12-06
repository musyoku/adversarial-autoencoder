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

def get_paddings_of_deconv_layers(target_size, num_layers, ksize, stride):
	# compute required deconv paddings
	paddings = []
	deconv_out_sizes = [target_size]
	deconv_out_sizes.append(get_conv_outsize(deconv_out_sizes[-1], ksize, stride, get_conv_padding(deconv_out_sizes[-1], ksize, stride)))
	deconv_out_sizes.append(get_conv_outsize(deconv_out_sizes[-1], ksize, stride, get_conv_padding(deconv_out_sizes[-1], ksize, stride)))
	deconv_out_sizes.append(get_conv_outsize(deconv_out_sizes[-1], ksize, stride, get_conv_padding(deconv_out_sizes[-1], ksize, stride)))
	# target_size of hidden layer must be an even number
	for i, size in enumerate(deconv_out_sizes[1:-1]):
		if size % 2 == 1:
			deconv_out_sizes[i+1] = size + 1

	deconv_out_sizes.reverse()
	paddings = []
	for i, (in_size, target_size) in enumerate(zip(deconv_out_sizes[:-1], deconv_out_sizes[1:])):
		paddings.append(get_deconv_padding(in_size, target_size, ksize, stride))

	return paddings

def get_in_size_of_deconv_layers(target_size, num_layers, ksize, stride):
	# compute required deconv paddings
	paddings = []
	deconv_out_sizes = [target_size]
	deconv_out_sizes.append(get_conv_outsize(deconv_out_sizes[-1], ksize, stride, get_conv_padding(deconv_out_sizes[-1], ksize, stride)))
	deconv_out_sizes.append(get_conv_outsize(deconv_out_sizes[-1], ksize, stride, get_conv_padding(deconv_out_sizes[-1], ksize, stride)))
	deconv_out_sizes.append(get_conv_outsize(deconv_out_sizes[-1], ksize, stride, get_conv_padding(deconv_out_sizes[-1], ksize, stride)))
	return deconv_out_sizes[-1]