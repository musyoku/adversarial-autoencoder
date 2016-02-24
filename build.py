# -*- coding: utf-8 -*-
import math
from chainer import links as L
from adversarial_autoencoder import Generator, Discriminator, Decoder

def build(config):
	config.check()

	# Generator
	gen_attributes = {}
	gen_layer_units = [(config.n_x, config.n_gen_hidden_units[0])]
	gen_layer_units += zip(config.n_gen_hidden_units[:-1], config.n_gen_hidden_units[1:])
	gen_layer_units += [(config.n_gen_hidden_units[-1], config.n_z)]

	if config.gen_encoder_type == "deterministic":
		for i, (n_in, n_out) in enumerate(gen_layer_units):
			gen_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=config.wscale_base * math.sqrt(n_in * n_out))
			gen_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

	elif config.gen_encoder_type == "gaussian":
		for i, (n_in, n_out) in enumerate(gen_layer_units):
			gen_attributes["layer_mean_%i" % i] = L.Linear(n_in, n_out, wscale=config.wscale_base * math.sqrt(n_in * n_out))
			gen_attributes["batchnorm_mean_%i" % i] = L.BatchNormalization(n_out)
			gen_attributes["layer_variance_%i" % i] = L.Linear(n_in, n_out, wscale=config.wscale_base * math.sqrt(n_in * n_out))
			gen_attributes["batchnorm_variance_%i" % i] = L.BatchNormalization(n_out)

	gen = Generator(**gen_attributes)
	gen.n_layers = len(gen_layer_units)
	gen.activation_type = config.gen_activation_type
	gen.output_activation_type = config.gen_output_activation_type
	gen.encoder_type = config.gen_encoder_type
	gen.enable_batchnorm = config.enable_batchnorm
	gen.enable_batchnorm_to_output = config.gen_enable_batchnorm_to_output
	gen.enable_dropout = config.gen_enable_dropout


	# Discriminator
	dis_attributes = {}
	dis_layer_units = [(config.n_dis_x, config.n_dis_hidden_units[0])]
	dis_layer_units += zip(config.n_dis_hidden_units[:-1], config.n_dis_hidden_units[1:])
	dis_layer_units += [(config.n_dis_hidden_units[-1], 2)]

	for i, (n_in, n_out) in enumerate(dis_layer_units):
		dis_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=config.wscale_base * math.sqrt(n_in * n_out))
		dis_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

	dis = Discriminator(**dis_attributes)
	dis.n_layers = len(dis_layer_units)
	dis.activation_type = config.dis_activation_type
	dis.softmax_activation_type = config.dis_softmax_activation_type
	dis.enable_batchnorm = config.enable_batchnorm
	dis.enable_batchnorm_to_input = config.dis_enable_batchnorm_to_input
	dis.enable_dropout = config.dis_enable_dropout

	# Decoder
	dec_attributes = {}
	dec_layer_units = [(config.n_z, config.n_dec_hidden_units[0])]
	dec_layer_units += zip(config.n_dec_hidden_units[:-1], config.n_dec_hidden_units[1:])
	dec_layer_units += [(config.n_dec_hidden_units[-1], config.n_x)]

	for i, (n_in, n_out) in enumerate(dec_layer_units):
		dec_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=config.wscale_base * math.sqrt(n_in * n_out))
		dec_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

	dec = Decoder(**dec_attributes)
	dec.n_layers = len(dec_layer_units)
	dec.activation_type = config.dec_activation_type
	dec.output_activation_type = config.dec_output_activation_type
	dec.enable_batchnorm = config.enable_batchnorm
	dec.enable_batchnorm_to_output = config.dec_enable_batchnorm_to_output
	dec.enable_dropout = config.dec_enable_dropout

	return gen, dis, dec