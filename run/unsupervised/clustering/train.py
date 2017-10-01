from __future__ import division
from __future__ import print_function
import argparse, chainer, time, sys
import numpy as np
import chainer.functions as F
from chainer import cuda
from model import Model
from aae.optim import Optimizer, GradientClipping
from aae.utils import onehot, printr, clear_console
from aae.dataset.supervised import Dataset
import aae.sampler as sampler

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=64)
	parser.add_argument("--total-epochs", "-e", type=int, default=5000)
	parser.add_argument("--num-labeled-data", "-nl", type=int, default=100)
	parser.add_argument("--gpu-device", "-g", type=int, default=0)
	parser.add_argument("--grad-clip", "-gc", type=float, default=5)
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.0001)
	parser.add_argument("--momentum", "-mo", type=float, default=0.1)
	parser.add_argument("--optimizer", "-opt", type=str, default="adam")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--model", "-m", type=str, default="model.hdf5")
	args = parser.parse_args()

	np.random.seed(args.seed)

	model = Model()
	model.load(args.model)

	mnist_train, mnist_test = chainer.datasets.get_mnist()
	images_train, labels_train = mnist_train._datasets
	images_test, labels_test = mnist_test._datasets

	# normalize
	images_train = (images_train - 0.5) * 2
	images_test = (images_test - 0.5) * 2

	dataset = Dataset(train=(images_train, labels_train), test=(images_test, labels_test))

	total_iterations_train = len(images_train) // args.batchsize

	# optimizers
	optimizer_encoder = Optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer_encoder.setup(model.encoder)
	if args.grad_clip > 0:
		optimizer_encoder.add_hook(GradientClipping(args.grad_clip))

	optimizer_decoder = Optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer_decoder.setup(model.decoder)
	if args.grad_clip > 0:
		optimizer_decoder.add_hook(GradientClipping(args.grad_clip))

	optimizer_discriminator_z = Optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer_discriminator_z.setup(model.discriminator_z)
	if args.grad_clip > 0:
		optimizer_discriminator_z.add_hook(GradientClipping(args.grad_clip))

	optimizer_discriminator_y = Optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer_discriminator_y.setup(model.discriminator_y)
	if args.grad_clip > 0:
		optimizer_discriminator_y.add_hook(GradientClipping(args.grad_clip))

	using_gpu = False
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu()
		using_gpu = True
	xp = model.xp

	# 0 -> true sample
	# 1 -> generated sample
	class_true = np.zeros(args.batchsize, dtype=np.int32)
	class_fake = np.ones(args.batchsize, dtype=np.int32)
	if using_gpu:
		class_true = cuda.to_gpu(class_true)
		class_fake = cuda.to_gpu(class_fake)

	training_start_time = time.time()
	for epoch in range(args.total_epochs):

		sum_loss_generator 		= 0
		sum_loss_discriminator 	= 0
		sum_loss_autoencoder 	= 0
		sum_discriminator_z_confidence_true = 0
		sum_discriminator_z_confidence_fake = 0
		sum_discriminator_y_confidence_true = 0
		sum_discriminator_y_confidence_fake = 0

		epoch_start_time = time.time()
		dataset.shuffle()

		# training
		for itr in range(total_iterations_train):
			# update model parameters
			with chainer.using_config("train", True):
				# sample minibatch
				x_u, _, _ = dataset.sample_minibatch(args.batchsize, gpu=using_gpu)
				
				### reconstruction phase ###
				if True:
					y_onehot_u, z_u = model.encode_x_yz(x_u, apply_softmax_y=True)
					x_reconstruction_u = model.decode_yz_x(y_onehot_u, z_u)
					loss_reconstruction = F.mean_squared_error(x_u, x_reconstruction_u)

					model.cleargrads()
					loss_reconstruction.backward()
					optimizer_encoder.update()
					optimizer_decoder.update()

				### adversarial phase ###
				if True:
					y_onehot_fake_u, z_fake_u = model.encode_x_yz(x_u, apply_softmax_y=True)

					z_true = sampler.gaussian(args.batchsize, model.ndim_z, mean=0, var=1)
					y_onehot_true = sampler.onehot_categorical(args.batchsize, model.ndim_y)
					if using_gpu:
						z_true = cuda.to_gpu(z_true)
						y_onehot_true = cuda.to_gpu(y_onehot_true)

					dz_true = model.discriminate_z(z_true, apply_softmax=False)
					dz_fake = model.discriminate_z(z_fake_u, apply_softmax=False)
					dy_true = model.discriminate_y(y_onehot_true, apply_softmax=False)
					dy_fake = model.discriminate_y(y_onehot_fake_u, apply_softmax=False)

					discriminator_z_confidence_true = float(xp.mean(F.softmax(dz_true).data[:, 0]))
					discriminator_z_confidence_fake = float(xp.mean(F.softmax(dz_fake).data[:, 1]))
					discriminator_y_confidence_true = float(xp.mean(F.softmax(dy_true).data[:, 0]))
					discriminator_y_confidence_fake = float(xp.mean(F.softmax(dy_fake).data[:, 1]))

					loss_discriminator_z = F.softmax_cross_entropy(dz_true, class_true) + F.softmax_cross_entropy(dz_fake, class_fake)
					loss_discriminator_y = F.softmax_cross_entropy(dy_true, class_true) + F.softmax_cross_entropy(dy_fake, class_fake)
					loss_discriminator = loss_discriminator_z + loss_discriminator_y

					model.cleargrads()
					loss_discriminator.backward()
					optimizer_discriminator_z.update()
					optimizer_discriminator_y.update()

				### generator phase ###
				if True:
					y_onehot_fake_u, z_fake_u = model.encode_x_yz(x_u, apply_softmax_y=True)

					dz_fake = model.discriminate_z(z_fake_u, apply_softmax=False)
					dy_fake = model.discriminate_y(y_onehot_fake_u, apply_softmax=False)

					loss_generator = F.softmax_cross_entropy(dz_fake, class_true) + F.softmax_cross_entropy(dy_fake, class_true)

					model.cleargrads()
					loss_generator.backward()
					optimizer_encoder.update()

				sum_loss_discriminator 	+= float(loss_discriminator.data)
				sum_loss_generator 		+= float(loss_generator.data)
				sum_loss_autoencoder 	+= float(loss_reconstruction.data)
				sum_discriminator_z_confidence_true += discriminator_z_confidence_true
				sum_discriminator_z_confidence_fake += discriminator_z_confidence_fake
				sum_discriminator_y_confidence_true += discriminator_y_confidence_true
				sum_discriminator_y_confidence_fake += discriminator_y_confidence_fake

			printr("Training ... {:3.0f}% ({}/{})".format((itr + 1) / total_iterations_train * 100, itr + 1, total_iterations_train))

		model.save(args.model)
			
		clear_console()
		print("Epoch {} done in {} sec - loss: g={:.5g}, d={:.5g}, a={:.5g} - disc_z: true={:.1f}%, fake={:.1f}% - disc_y: true={:.1f}%, fake={:.1f}% - total {} min".format(
			epoch + 1, int(time.time() - epoch_start_time), 
			sum_loss_generator / total_iterations_train, 
			sum_loss_discriminator / total_iterations_train, 
			sum_loss_autoencoder / total_iterations_train, 
			sum_discriminator_z_confidence_true / total_iterations_train * 100, 
			sum_discriminator_z_confidence_fake / total_iterations_train * 100, 
			sum_discriminator_y_confidence_true / total_iterations_train * 100, 
			sum_discriminator_y_confidence_fake / total_iterations_train * 100, 
			int((time.time() - training_start_time) // 60)))


if __name__ == "__main__":
	main()
