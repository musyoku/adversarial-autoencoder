import numpy as np
import os, sys, time
from chainer import cuda
from chainer import functions as F
from model import aae
from progress import Progress
from args import args
import dataset
import sampler

# _l -> labeled
# _u -> unlabeled
def main():
	# load MNIST images
	images, labels = dataset.load_train_images()

	# config
	config = aae.config

	# settings
	max_epoch = 1000
	num_trains_per_epoch = 5000
	batchsize = 100
	alpha = 1

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# classification
	# 0 -> true sample
	# 1 -> generated sample
	class_true = aae.to_variable(np.zeros(batchsize, dtype=np.int32))
	class_fake = aae.to_variable(np.ones(batchsize, dtype=np.int32))

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_reconstruction = 0
		sum_loss_discriminator = 0
		sum_loss_generator = 0

		for t in xrange(num_trains_per_epoch):
			# sample from data distribution
			images_u = dataset.sample_unlabeled_data(images, batchsize, config.ndim_x)

			# reconstruction phase
			q_y_x_u, z_u = aae.encode_x_yz(images_u, apply_softmax=True)
			reconstruction_u = aae.decode_yz_x(q_y_x_u, z_u)
			loss_reconstruction = F.mean_squared_error(aae.to_variable(images_u), reconstruction_u)
			aae.backprop_generator(loss_reconstruction)
			aae.backprop_decoder(loss_reconstruction)

			# adversarial phase
			y_fake_u, z_fake_u = aae.encode_x_yz(images_u, apply_softmax=True)
			z_true_u = sampler.gaussian(batchsize, config.ndim_z, mean=0, var=1)
			y_true_u = sampler.onehot_categorical(batchsize, config.ndim_y)
			discrimination_z_true = aae.discriminate_z(z_true_u, apply_softmax=False)
			discrimination_y_true = aae.discriminate_y(y_true_u, apply_softmax=False)
			discrimination_z_fake = aae.discriminate_z(z_fake_u, apply_softmax=False)
			discrimination_y_fake = aae.discriminate_y(y_fake_u, apply_softmax=False)
			loss_discriminator_z = F.softmax_cross_entropy(discrimination_z_true, class_true) + F.softmax_cross_entropy(discrimination_z_fake, class_fake)
			loss_discriminator_y = F.softmax_cross_entropy(discrimination_y_true, class_true) + F.softmax_cross_entropy(discrimination_y_fake, class_fake)
			loss_discriminator = loss_discriminator_z + loss_discriminator_y
			aae.backprop_discriminator(loss_discriminator)

			# adversarial phase
			y_fake_u, z_fake_u = aae.encode_x_yz(images_u, apply_softmax=True)
			discrimination_z_fake = aae.discriminate_z(z_fake_u, apply_softmax=False)
			discrimination_y_fake = aae.discriminate_y(y_fake_u, apply_softmax=False)
			loss_generator_z = F.softmax_cross_entropy(discrimination_z_fake, class_true)
			loss_generator_y = F.softmax_cross_entropy(discrimination_y_fake, class_true)
			loss_generator = loss_generator_z + loss_generator_y
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
