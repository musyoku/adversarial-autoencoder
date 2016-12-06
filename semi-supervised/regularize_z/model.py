# -*- coding: utf-8 -*-
import math
import json, os, sys
from chainer import cuda
from args import args
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from aae_regularized import AAE, Config
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
	config.ndim_z = 2
	config.distribution_z = "deterministic"	# deterministic or gaussian
	config.weight_init_std = 0.001
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0002
	config.momentum = 0.5
	config.gradient_clipping = 5
	config.weight_decay = 0

	decoder = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	decoder.add(Linear(None, 1000))
	decoder.add(Activation(config.nonlinearity))
	decoder.add(Linear(None, 1000))
	decoder.add(Activation(config.nonlinearity))
	decoder.add(Linear(None, config.ndim_x))
	decoder.add(sigmoid())

	discriminator = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	discriminator.add(Merge(num_inputs=2, out_size=1000, nobias=True))
	discriminator.add(gaussian_noise(std=0.3))
	discriminator.add(Activation(config.nonlinearity))
	discriminator.add(Linear(None, 1000))
	discriminator.add(Activation(config.nonlinearity))
	discriminator.add(Linear(None, 1000))
	discriminator.add(Activation(config.nonlinearity))
	discriminator.add(Linear(None, 2))

	generator = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	generator.add(Linear(config.ndim_x, 1000))
	generator.add(Activation(config.nonlinearity))
	generator.add(Linear(None, 1000))
	generator.add(Activation(config.nonlinearity))
	if config.distribution_z == "deterministic":
		generator.add(Linear(None, config.ndim_z))
	elif config.distribution_z == "gaussian":
		generator.add(Gaussian(None, config.ndim_z))	# outputs mean and ln(var)
	else:
		raise Exception()

	params = {
		"config": config.to_dict(),
		"decoder": decoder.to_dict(),
		"generator": generator.to_dict(),
		"discriminator": discriminator.to_dict(),
	}

	with open(model_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

aae = AAE(params)
aae.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	aae.to_gpu()
