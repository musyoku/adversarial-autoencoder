import numpy as np
import os, sys, time
from chainer import cuda
from chainer import functions as F
import pandas as pd
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
	num_trains_per_epoch = 500
	batchsize_l = 100
	batchsize_u = 100
	alpha = 1

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# save validation accuracy per epoch
	csv_results = []

	# create semi-supervised split
	num_validation_data = 10000
	num_labeled_data = 100
	num_types_of_label = 10
	training_images_l, training_labels_l, training_images_u, validation_images, validation_labels = dataset.create_semisupervised(images, labels, num_validation_data, num_labeled_data, num_types_of_label)
	print training_labels_l

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
			images_l, label_onehot_l, label_ids_l = dataset.sample_labeled_data(training_images_l, training_labels_l, batchsize_l)
			images_u = dataset.sample_unlabeled_data(training_images_u, batchsize_u)

			# reconstruction phase
			qy_x_u, z_u = aae.encode_x_yz(images_u, apply_softmax=True)
			reconstruction_u = aae.decode_yz_x(qy_x_u, z_u)
			loss_reconstruction = F.mean_squared_error(aae.to_variable(images_u), reconstruction_u)
			aae.backprop_generator(loss_reconstruction)
			aae.backprop_decoder(loss_reconstruction)

			# adversarial phase
			y_fake_u, z_fake_u = aae.encode_x_yz(images_u, apply_softmax=True)
			z_true_u = sampler.gaussian(batchsize_u, config.ndim_z, mean=0, var=1)
			y_true_u = sampler.onehot_categorical(batchsize_u, config.ndim_y)
			dz_true = aae.discriminate_z(z_true_u, apply_softmax=False)
			dy_true = aae.discriminate_y(y_true_u, apply_softmax=False)
			dz_fake = aae.discriminate_z(z_fake_u, apply_softmax=False)
			dy_fake = aae.discriminate_y(y_fake_u, apply_softmax=False)
			loss_discriminator_z = F.softmax_cross_entropy(dz_true, class_true) + F.softmax_cross_entropy(dz_fake, class_fake)
			loss_discriminator_y = F.softmax_cross_entropy(dy_true, class_true) + F.softmax_cross_entropy(dy_fake, class_fake)
			loss_discriminator = loss_discriminator_z + loss_discriminator_y
			aae.backprop_discriminator(loss_discriminator)

			# adversarial phase
			y_fake_u, z_fake_u = aae.encode_x_yz(images_u, apply_softmax=True)
			dz_fake = aae.discriminate_z(z_fake_u, apply_softmax=False)
			dy_fake = aae.discriminate_y(y_fake_u, apply_softmax=False)
			loss_generator_z = F.softmax_cross_entropy(dz_fake, class_true)
			loss_generator_y = F.softmax_cross_entropy(dy_fake, class_true)
			loss_generator = loss_generator_z + loss_generator_y
			aae.backprop_generator(loss_generator)

			# supervised phase
			log_qy_x_l, z_l = aae.encode_x_yz(images_l, apply_softmax=False)
			loss_supervised = F.softmax_cross_entropy(log_qy_x_l, aae.to_variable(label_ids_l))
			aae.backprop_generator(loss_supervised)

			sum_loss_reconstruction += float(loss_reconstruction.data)
			sum_loss_supervised += float(loss_supervised.data)
			sum_loss_discriminator += float(loss_discriminator.data)
			sum_loss_generator += float(loss_generator.data)

			if t % 10 == 0:
				progress.show(t, num_trains_per_epoch, {})

		aae.save(args.model_dir)

		# validation
		images_v_segments = np.split(validation_images, num_validation_data // 1000)
		labels_v_segments = np.split(validation_labels, num_validation_data // 1000)
		sum_accuracy = 0
		for images_v, labels_v in zip(images_v_segments, labels_v_segments):
			qy = aae.encode_x_yz(images_v, apply_softmax=True, test=True)[0]
			accuracy = F.accuracy(qy, aae.to_variable(labels_v))
			sum_accuracy += float(accuracy.data)
		validation_accuracy = sum_accuracy / len(images_v_segments)
		
		progress.show(num_trains_per_epoch, num_trains_per_epoch, {
			"loss_r": sum_loss_reconstruction / num_trains_per_epoch,
			"loss_s": sum_loss_supervised / num_trains_per_epoch,
			"loss_d": sum_loss_discriminator / num_trains_per_epoch,
			"loss_g": sum_loss_generator / num_trains_per_epoch,
			"accuracy": validation_accuracy
		})

		# write accuracy to csv
		csv_results.append([epoch, validation_accuracy])
		data = pd.DataFrame(csv_results)
		data.columns = ["epoch", "accuracy"]
		data.to_csv("{}/result.csv".format(args.model_dir))

if __name__ == "__main__":
	main()
