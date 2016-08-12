# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import util
from args import args
from model import conf, aae
import sampler

max_epoch = 1000
num_trains_per_epoch = 1000
batchsize = 100
n_steps_to_optimize_dis = 1

# Create labeled/unlabeled split in training set
n_types_of_label = conf.ndim_y
n_labeled_data = args.n_labeled_data
n_validation_data = 10000

# Export result to csv
csv_epoch = []

dataset, labels = util.load_labeled_images(args.train_image_dir)
labeled_dataset, labels, unlabeled_dataset, validation_dataset, validation_labels = util.create_semisupervised(dataset, labels, n_validation_data, n_labeled_data, n_types_of_label)

def sample_labeled_data():
	x, y_onehot, y_id = util.sample_x_and_label_variables(batchsize, conf.ndim_x, conf.ndim_y, labeled_dataset, labels, gpu_enabled=conf.gpu_enabled)
	noise = sampler.gaussian(batchsize, conf.ndim_x, mean=0, var=0.3, gpu_enabled=conf.gpu_enabled)
	# x.data += noise.data
	return x, y_onehot, y_id

def sample_unlabeled_data():
	x = util.sample_x_variable(batchsize, conf.ndim_x, unlabeled_dataset, gpu_enabled=conf.gpu_enabled)
	noise = sampler.gaussian(batchsize, conf.ndim_x, mean=0, var=0.3, gpu_enabled=conf.gpu_enabled)
	# x.data += noise.data
	return x

def sample_validation_data():
	return util.sample_x_and_label_variables(n_validation_data, conf.ndim_x, conf.ndim_y, validation_dataset, validation_labels, gpu_enabled=False)

total_time = 0
for epoch in xrange(1, max_epoch + 1):

	sum_loss_autoencoder = 0
	sum_loss_discriminator = 0
	sum_loss_generator = 0
	sum_loss_classifier = 0
	epoch_time = time.time()

	for t in xrange(1, num_trains_per_epoch + 1):
		# reconstruction phase
		x = sample_unlabeled_data()
		aae.update_learning_rate(conf.learning_rate_for_reconstruction_cost)
		aae.update_momentum(conf.momentum_for_reconstruction_cost)
		sum_loss_autoencoder += aae.train_autoencoder_unsupervised(x)

		# regularization phase
		## train discriminator
		aae.update_learning_rate(conf.learning_rate_for_adversarial_cost)
		aae.update_momentum(conf.momentum_for_adversarial_cost)
		loss_discriminator = 0
		for k in xrange(n_steps_to_optimize_dis):
			if k > 0:
				x = sample_unlabeled_data()
			z_true = sampler.gaussian(batchsize, conf.ndim_z)
			y_true = sampler.onehot_categorical(batchsize, conf.ndim_y)
			loss_discriminator += aae.train_discriminator_yz(x, y_true, z_true)
		loss_discriminator /= n_steps_to_optimize_dis
		sum_loss_discriminator += loss_discriminator

		## train generator
		sum_loss_generator += aae.train_generator_x_yz(x)

		# semi-supervised classification phase
		x_labeled, y_onehot, y_id = sample_labeled_data()
		aae.update_learning_rate(conf.learning_rate_for_semi_supervised_cost)
		aae.update_momentum(conf.momentum_for_semi_supervised_cost)
		sum_loss_classifier += aae.train_classifier(x_labeled, y_id)

		if t % 10 == 0:
			sys.stdout.write("\rTraining in progress...({} / {})".format(t, num_trains_per_epoch))
			sys.stdout.flush()

	sys.stdout.write("\n")
	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	print "epoch:", epoch
	print "  loss:"
	print "    autoencoder  : {:.4f}".format(sum_loss_autoencoder / num_trains_per_epoch)
	print "    discriminator: {:.4f}".format(sum_loss_discriminator / num_trains_per_epoch)
	print "    generator    : {:.4f}".format(sum_loss_generator / num_trains_per_epoch)
	print "    classifier   : {:.4f}".format(sum_loss_classifier / num_trains_per_epoch)
	print "  time: {} min".format(int(epoch_time / 60)), "total: {} min".format(int(total_time / 60))
	aae.save(args.model_dir)

	# validation
	x_labeled, y_onehot, y_id = sample_validation_data()
	if conf.gpu_enabled:
		x_labeled.to_gpu()
	prediction = aae.sample_x_label(x_labeled, test=True, argmax=True)
	correct = 0
	for i in xrange(n_validation_data):
		if prediction[i] == y_id.data[i]:
			correct += 1
	print "classification accuracy (validation): {}".format(correct / float(n_validation_data))

	# Export to csv
	csv_epoch.append([epoch, int(total_time / 60), correct / float(n_validation_data)])
	data = pd.DataFrame(csv_epoch)
	data.columns = ["epoch", "min", "accuracy"]
	data.to_csv("{}/epoch.csv".format(args.model_dir))

