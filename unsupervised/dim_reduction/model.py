# -*- coding: utf-8 -*-
from args import args
import aae
from aae_dim_reduction import AAE, Conf

conf = Conf()
conf.gpu_enabled = True if args.gpu_enabled == 1 else False
conf.distance_threshold = 1
conf.learning_rate_for_reconstruction_cost = 0.0001
conf.learning_rate_for_adversarial_cost = 0.0001
conf.learning_rate_for_cluster_head = 0.01
conf.wscale = 0.1
conf.gradient_momentum = 0.1
conf.gradient_clipping = 5.0

# number of reduced dimention
conf.ndim_z = 2

conf.ndim_y = 20

conf.batchnorm_before_activation = True

conf.generator_shared_hidden_units = [3000]
conf.generator_hidden_units = [3000]
conf.generator_activation_function = "elu"
conf.generator_apply_dropout = False
conf.generator_apply_batchnorm = False
conf.generator_apply_batchnorm_to_input = False

conf.decoder_hidden_units = [3000, 3000]
conf.decoder_activation_function = "elu"
conf.decoder_apply_dropout = False
conf.decoder_apply_batchnorm = False
conf.decoder_apply_batchnorm_to_input = False

conf.discriminator_z_hidden_units = [3000, 3000]
conf.discriminator_z_activation_function = "elu"
conf.discriminator_z_apply_dropout = False
conf.discriminator_z_apply_batchnorm = False
conf.discriminator_z_apply_batchnorm_to_input = False

conf.discriminator_y_hidden_units = [3000, 3000]
conf.discriminator_y_activation_function = "elu"
conf.discriminator_y_apply_dropout = False
conf.discriminator_y_apply_batchnorm = False
conf.discriminator_y_apply_batchnorm_to_input = False

conf.q_z_x_type = aae.Q_Z_X_TYPE_GAUSSIAN

aae = AAE(conf, name="aae")
aae.load(args.model_dir)
