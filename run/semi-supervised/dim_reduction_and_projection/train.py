from __future__ import division
from __future__ import print_function
import argparse, chainer, time, sys, math
import numpy as np
import chainer.functions as F
from chainer import cuda
from model import Model
from aae.optim import Optimizer, GradientClipping
from aae.utils import onehot, printr, clear_console
from aae.dataset.semi_supervised import Dataset
import aae.sampler as sampler

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=64)
	parser.add_argument("--total-epochs", "-e", type=int, default=5000)
	parser.add_argument("--num-labeled-data", "-nl", type=int, default=100)
	parser.add_argument("--gpu-device", "-g", type=int, default=0)
	parser.add_argument("--grad-clip", "-gc", type=float, default=5)
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

	dataset = Dataset(train=(images_train, labels_train), 
					  test=(images_test, labels_test), 
					  num_labeled_data=args.num_labeled_data, 
					  num_classes=model.ndim_y)
	print("#labeled:	{}".format(dataset.get_num_labeled_data()))
	print("#unlabeled:	{}".format(dataset.get_num_unlabeled_data()))
	_, labels = dataset.get_labeled_data()
	print("labeled data:", labels)

	total_iterations_train = len(images_train) // args.batchsize

	# optimizers
	optimizer_encoder = Optimizer("msgd", 0.01, 0.9)
	optimizer_encoder.setup(model.encoder)
	if args.grad_clip > 0:
		optimizer_encoder.add_hook(GradientClipping(args.grad_clip))

	optimizer_semi_supervised = Optimizer("msgd", 0.1, 0.9)
	optimizer_semi_supervised.setup(model.encoder)
	if args.grad_clip > 0:
		optimizer_semi_supervised.add_hook(GradientClipping(args.grad_clip))

	optimizer_generator = Optimizer("msgd", 0.1, 0.1)
	optimizer_generator.setup(model.encoder)
	if args.grad_clip > 0:
		optimizer_generator.add_hook(GradientClipping(args.grad_clip))

	optimizer_decoder = Optimizer("msgd", 0.01, 0.9)
	optimizer_decoder.setup(model.decoder)
	if args.grad_clip > 0:
		optimizer_decoder.add_hook(GradientClipping(args.grad_clip))

	optimizer_discriminator_z = Optimizer("msgd", 0.1, 0.1)
	optimizer_discriminator_z.setup(model.discriminator_z)
	if args.grad_clip > 0:
		optimizer_discriminator_z.add_hook(GradientClipping(args.grad_clip))

	optimizer_discriminator_y = Optimizer("msgd", 0.1, 0.1)
	optimizer_discriminator_y.setup(model.discriminator_y)
	if args.grad_clip > 0:
		optimizer_discriminator_y.add_hook(GradientClipping(args.grad_clip))

	optimizer_linear_transformation = Optimizer("msgd", 0.01, 0.9)
	optimizer_linear_transformation.setup(model.linear_transformation)
	if args.grad_clip > 0:
		optimizer_linear_transformation.add_hook(GradientClipping(args.grad_clip))

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

	# 2D circle
	# we use a linear transformation to map the 10D representation to a 2D space such that 
	# the cluster heads are mapped to the points that are uniformly placed on a 2D circle.
	rad = math.radians(360 / model.ndim_y)
	radius = 5
	mapped_cluster_head_2d_target = np.zeros((10, 2), dtype=np.float32)
	for n in range(model.ndim_y):
		x = math.cos(rad * n) * radius
		y = math.sin(rad * n) * radius
		mapped_cluster_head_2d_target[n] = (x, y)
	if using_gpu:
		mapped_cluster_head_2d_target = cuda.to_gpu(mapped_cluster_head_2d_target)

	# training loop
	training_start_time = time.time()
	for epoch in range(args.total_epochs):

		sum_loss_generator 		= 0
		sum_loss_discriminator 	= 0
		sum_loss_autoencoder 	= 0
		sum_loss_supervised 	= 0
		sum_loss_linear_transformation = 0
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
				x_u = dataset.sample_unlabeled_minibatch(args.batchsize, gpu=using_gpu)
				x_l, y_l, _ = dataset.sample_labeled_minibatch(args.batchsize, gpu=using_gpu)
				
				### reconstruction phase ###
				if True:
					y_onehot_u, z_u = model.encode_x_yz(x_u, apply_softmax_y=True)
					repr_u = model.encode_yz_representation(y_onehot_u, z_u)
					x_reconstruction_u = model.decode_representation_x(repr_u)
					loss_reconstruction_u = F.mean_squared_error(x_u, x_reconstruction_u)

					y_onehot_l, z_l = model.encode_x_yz(x_l, apply_softmax_y=True)
					repr_l = model.encode_yz_representation(y_onehot_l, z_l)
					x_reconstruction_l = model.decode_representation_x(repr_l)
					loss_reconstruction_l = F.mean_squared_error(x_l, x_reconstruction_l)

					loss_reconstruction = loss_reconstruction_u + loss_reconstruction_l

					model.cleargrads()
					loss_reconstruction.backward()
					optimizer_encoder.update()
					optimizer_decoder.update()

					sum_loss_autoencoder += float(loss_reconstruction.data)

				### adversarial phase ###
				if True:
					y_onehot_fake_u, z_fake_u = model.encode_x_yz(x_u, apply_softmax_y=True)

					z_true = sampler.gaussian(args.batchsize, model.ndim_y, mean=0, var=1)
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

					sum_loss_discriminator += float(loss_discriminator.data)
					sum_discriminator_z_confidence_true += discriminator_z_confidence_true
					sum_discriminator_z_confidence_fake += discriminator_z_confidence_fake
					sum_discriminator_y_confidence_true += discriminator_y_confidence_true
					sum_discriminator_y_confidence_fake += discriminator_y_confidence_fake

				### generator phase ###
				if True:
					y_onehot_fake_u, z_fake_u = model.encode_x_yz(x_u, apply_softmax_y=True)

					dz_fake = model.discriminate_z(z_fake_u, apply_softmax=False)
					dy_fake = model.discriminate_y(y_onehot_fake_u, apply_softmax=False)

					loss_generator = F.softmax_cross_entropy(dz_fake, class_true) + F.softmax_cross_entropy(dy_fake, class_true)

					model.cleargrads()
					loss_generator.backward()
					optimizer_generator.update()

					sum_loss_generator += float(loss_generator.data)

				### supervised phase ###
				if True:
					logit_l, _ = model.encode_x_yz(x_l, apply_softmax_y=False)
					loss_supervised = F.softmax_cross_entropy(logit_l, y_l)

					model.cleargrads()
					loss_supervised.backward()
					optimizer_semi_supervised.update()

					sum_loss_supervised += float(loss_supervised.data)

				### additional cost ###
				if True:
					identity = np.identity(model.ndim_y, dtype=np.float32)
					if using_gpu:
						identity = cuda.to_gpu(identity)
					mapped_head = model.linear_transformation(identity)
					loss_linear_transformation = F.mean_squared_error(mapped_cluster_head_2d_target, mapped_head)

					model.cleargrads()
					loss_linear_transformation.backward()
					optimizer_linear_transformation.update()

					sum_loss_linear_transformation	+= float(loss_linear_transformation.data)

			printr("Training ... {:3.0f}% ({}/{})".format((itr + 1) / total_iterations_train * 100, itr + 1, total_iterations_train))

		model.save(args.model)

		labeled_iter_train = dataset.get_iterator(args.batchsize * 20, train=True, labeled=True, gpu=using_gpu)
		unlabeled_iter_train = dataset.get_iterator(args.batchsize * 20, train=True, unlabeled=True, gpu=using_gpu)
		average_accuracy_l = 0
		average_accuracy_u = 0
		for x_l, true_label in labeled_iter_train:
			with chainer.no_backprop_mode() and chainer.using_config("train", False):
				y_onehot_l, _ = model.encode_x_yz(x_l, apply_softmax_y=True)
				accuracy = F.accuracy(y_onehot_l, true_label)
				average_accuracy_l += float(accuracy.data)

		for x_u, true_label in unlabeled_iter_train:
			with chainer.no_backprop_mode() and chainer.using_config("train", False):
				y_onehot_u, _ = model.encode_x_yz(x_u, apply_softmax_y=True)
				accuracy = F.accuracy(y_onehot_u, true_label)
				average_accuracy_u += float(accuracy.data)

		average_accuracy_l /= labeled_iter_train.get_total_iterations()
		average_accuracy_u /= unlabeled_iter_train.get_total_iterations()
			
		clear_console()
		print("Epoch {} done in {} sec - loss: g={:.5g}, d={:.5g}, a={:.5g}, s={:.5g}, l={:.5g} - disc_z: true={:.1f}%, fake={:.1f}% - disc_y: true={:.1f}%, fake={:.1f}% - acc: l={:.2f}%, u={:.2f}% - total {} min".format(
			epoch + 1, int(time.time() - epoch_start_time), 
			sum_loss_generator / total_iterations_train, 
			sum_loss_discriminator / total_iterations_train, 
			sum_loss_autoencoder / total_iterations_train, 
			sum_loss_supervised / total_iterations_train, 
			sum_loss_linear_transformation / total_iterations_train, 
			sum_discriminator_z_confidence_true / total_iterations_train * 100, 
			sum_discriminator_z_confidence_fake / total_iterations_train * 100, 
			sum_discriminator_y_confidence_true / total_iterations_train * 100, 
			sum_discriminator_y_confidence_fake / total_iterations_train * 100, 
			average_accuracy_l * 100,
			average_accuracy_u * 100,
			int((time.time() - training_start_time) // 60)))

	if epoch == 50:
		optimizer_encoder.set_learning_rate(0.001)
		optimizer_decoder.set_learning_rate(0.001)
		optimizer_semi_supervised.set_learning_rate(0.01)
		optimizer_generator.set_learning_rate(0.01)
		optimizer_discriminator_y.set_learning_rate(0.01)
		optimizer_discriminator_z.set_learning_rate(0.01)

	if epoch == 1000:
		optimizer_encoder.set_learning_rate(0.0001)
		optimizer_decoder.set_learning_rate(0.0001)
		optimizer_semi_supervised.set_learning_rate(0.001)
		optimizer_generator.set_learning_rate(0.001)
		optimizer_discriminator_y.set_learning_rate(0.001)
		optimizer_discriminator_z.set_learning_rate(0.001)


if __name__ == "__main__":
	main()
