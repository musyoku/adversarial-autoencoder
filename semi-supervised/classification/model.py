# -*- coding: utf-8 -*-
import math
import json, os, sys
from chainer import cuda
from args import args
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from aae_semi_supervised import AAE, Config
from sequential import Sequential
from sequential.layers import Linear, Merge, BatchNormalization, Gaussian
from sequential.functions import Activation, dropout, gaussian_noise, tanh, sigmoid

try:
	os.mkdir(args.model_dir)
except:
	pass

model_filename = args.model_dir + "/model.json"

if os.path.isfile(model_filename):
	print "loading", model_filename
	with open(model_filename, "r") as f:
		try:
			params = json.load(f)
		except Exception as e:
			raise Exception("could not load {}".format(model_filename))
else:
	config = Config()
	config.ndim_x = 28 * 28
	config.ndim_y = 10
	config.ndim_z = 10
	config.distribution_z = "deterministic"	# deterministic or gaussian
	config.weight_init_std = 0.001
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0001
	config.momentum = 0.1
	config.gradient_clipping = 5
	config.weight_decay = 0

	# x = decoder(y, z)
	decoder = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	decoder.add(Merge(num_inputs=2, out_size=1000, nobias=True))
	decoder.add(Activation(config.nonlinearity))
	decoder.add(Linear(None, 1000))
	decoder.add(Activation(config.nonlinearity))
	decoder.add(Linear(None, 1000))
	decoder.add(Activation(config.nonlinearity))
	decoder.add(Linear(None, config.ndim_x))
	decoder.add(sigmoid())

	discriminator_z = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	discriminator_z.add(gaussian_noise(std=0.3))
	discriminator_z.add(Linear(config.ndim_z, 1000))
	discriminator_z.add(Activation(config.nonlinearity))
	discriminator_z.add(Linear(None, 1000))
	discriminator_z.add(Activation(config.nonlinearity))
	discriminator_z.add(Linear(None, 2))

	discriminator_y = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	discriminator_y.add(gaussian_noise(std=0.3))
	discriminator_y.add(Linear(config.ndim_y, 1000))
	discriminator_y.add(Activation(config.nonlinearity))
	discriminator_y.add(Linear(None, 1000))
	discriminator_y.add(Activation(config.nonlinearity))
	discriminator_y.add(Linear(None, 2))

	# z, y_softmax = generator(x)
	generator_shared = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	generator_shared.add(Linear(config.ndim_x, 1000))
	generator_shared.add(Activation(config.nonlinearity))
	generator_shared.add(Linear(None, 1000))
	generator_shared.add(Activation(config.nonlinearity))

	generator_z = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	if config.distribution_z == "deterministic":
		generator_z.add(Linear(None, config.ndim_z))
	elif config.distribution_z == "gaussian":
		generator_z.add(Gaussian(None, config.ndim_z))	# outputs mean and ln(var)
	else:
		raise Exception()

	generator_y = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	generator_y.add(Linear(None, config.ndim_y))

	params = {
		"config": config.to_dict(),
		"decoder": decoder.to_dict(),
		"generator_shared": generator_shared.to_dict(),
		"generator_z": generator_z.to_dict(),
		"generator_y": generator_y.to_dict(),
		"discriminator_y": discriminator_y.to_dict(),
		"discriminator_z": discriminator_z.to_dict(),
	}

	with open(model_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

aae = AAE(params)
aae.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	aae.to_gpu()
