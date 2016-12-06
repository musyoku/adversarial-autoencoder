import numpy as np
import os, sys, time
from chainer import cuda
from chainer import functions as F
from model import aae
from progress import Progress
from args import args
import dataset
import sampler

def main():
	# load MNIST images
	images, labels = dataset.load_train_images()

	# config
	config = aae.config

	# settings
	# _l -> labeled
	# _u -> unlabeled
	max_epoch = 1000
	num_trains_per_epoch = 5000
	batchsize_l = 100
	batchsize_u = 100
	alpha = 1

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# create semi-supervised split
	num_labeled_data = 10000
	num_types_of_label = 11		# additional label corresponds to unlabeled data
	training_images_l, training_labels_l, training_images_u, _, _ = dataset.create_semisupervised(images, labels, 0, num_labeled_data, num_types_of_label, seed=args.seed)

	# classification
	# 0 -> true sample
	# 1 -> generated sample
	class_true = aae.to_variable(np.zeros(batchsize_u, dtype=np.int32))
	class_fake = aae.to_variable(np.ones(batchsize_u, dtype=np.int32))

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_reconstruction = 0
		sum_loss_supervised = 0
		sum_loss_discriminator = 0
		sum_loss_generator = 0

		for t in xrange(num_trains_per_epoch):
			# sample from data distribution
			images_l, label_onehot_l, label_ids_l = dataset.sample_labeled_data(training_images_l, training_labels_l, batchsize_l, config.ndim_x, num_types_of_label)
			images_u = dataset.sample_unlabeled_data(training_images_u, batchsize_u, config.ndim_x)

			# reconstruction phase
			z_u = aae.encode_x_z(images_u)
			reconstruction_u = aae.decode_z_x(z_u)
			loss_reconstruction = F.mean_squared_error(aae.to_variable(images_u), reconstruction_u)
			aae.backprop_generator(loss_reconstruction)
			aae.backprop_decoder(loss_reconstruction)

			# adversarial phase
			z_fake_u = aae.encode_x_z(images_u)
			z_fake_l = aae.encode_x_z(images_l)
			onehot = np.zeros((1, num_types_of_label), dtype=np.float32)
			onehot[0, -1] = 1	# turn on the extra class
			label_onehot_u = np.repeat(onehot, batchsize_u, axis=0)
			z_true_l = sampler.supervised_gaussian_mixture(batchsize_l, config.ndim_z, label_ids_l, num_types_of_label - 1)
			z_true_u = sampler.gaussian_mixture(batchsize_u, config.ndim_z, num_types_of_label - 1)
			discrimination_z_true_l = aae.discriminate_z(label_onehot_l, z_true_l, apply_softmax=False)
			discrimination_z_true_u = aae.discriminate_z(label_onehot_u, z_true_u, apply_softmax=False)
			discrimination_z_fake_l = aae.discriminate_z(label_onehot_l, z_fake_l, apply_softmax=False)
			discrimination_z_fake_u = aae.discriminate_z(label_onehot_u, z_fake_u, apply_softmax=False)
			loss_discriminator = F.softmax_cross_entropy(discrimination_z_true_l, class_true) + F.softmax_cross_entropy(discrimination_z_true_u, class_true) + F.softmax_cross_entropy(discrimination_z_fake_l, class_fake) + F.softmax_cross_entropy(discrimination_z_fake_u, class_fake)
			aae.backprop_discriminator(loss_discriminator)

			# adversarial phase
			z_fake_u = aae.encode_x_z(images_u)
			z_fake_l = aae.encode_x_z(images_l)
			discrimination_z_fake_l = aae.discriminate_z(label_onehot_l, z_fake_l, apply_softmax=False)
			discrimination_z_fake_u = aae.discriminate_z(label_onehot_u, z_fake_u, apply_softmax=False)
			loss_generator = F.softmax_cross_entropy(discrimination_z_fake_l, class_true) + F.softmax_cross_entropy(discrimination_z_fake_u, class_true)
			aae.backprop_generator(loss_generator)

			sum_loss_reconstruction += float(loss_reconstruction.data)
			sum_loss_discriminator += float(loss_discriminator.data)
			sum_loss_generator += float(loss_generator.data)

			if t % 10 == 0:
				progress.show(t, num_trains_per_epoch, {})

		aae.save(args.model_dir)

		progress.show(num_trains_per_epoch, num_trains_per_epoch, {
			"loss_r": sum_loss_reconstruction / num_trains_per_epoch,
			"loss_d": sum_loss_discriminator / num_trains_per_epoch,
			"loss_g": sum_loss_generator / num_trains_per_epoch,
		})

if __name__ == "__main__":
	main()
