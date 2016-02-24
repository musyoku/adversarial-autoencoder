# -*- coding: utf-8 -*-
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
import os, sys, time
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from config import config
from model import gen, dis, dec
from util import *

def sample_x_and_label_from_data_distribution(batchsize):
	shape = config.img_channel * config.img_width * config.img_width
	x_batch = np.zeros((batchsize, shape), dtype=np.float32)
	label_batch = np.zeros((batchsize, 1), dtype=np.int32)
	for j in range(batchsize):
		data_index = np.random.randint(len(dataset))
		img = dataset[data_index]
		x_batch[j] = img.reshape((shape,))
		label_batch[j] = labels[data_index]
	x_batch = Variable(x_batch)
	if config.use_gpu:
		x_batch.to_gpu()
	return x_batch, label_batch

def train(dataset, labels):
	if config.n_z != 2:
		raise Exception("Latent code dimension must be 2")
	batchsize = 100
	n_epoch = 10000
	n_train_each_epoch = 500
	total_time = 0

	xp = cuda.cupy if config.use_gpu else np

	# Discriminatorの学習回数
	## 詳細は[Generative Adversarial Networks](http://arxiv.org/abs/1406.2661)
	n_steps_to_optimize_dis = 2

	# Use Adam
	optimizer_dec = optimizers.Adam(alpha=0.0002, beta1=0.5)
	optimizer_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
	optimizer_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
	optimizer_gen.setup(gen)
	optimizer_dec.setup(dec)
	optimizer_dis.setup(dis)

	start_epoch = 1 if args.load_epoch == 0 else args.load_epoch + 1

	for epoch in xrange(start_epoch, n_epoch):
		# Adversarial Networksの誤差
		sum_loss_regularization = 0
		# 復号誤差
		sum_loss_reconstruction = 0

		start_time = time.time()
		
		for i in xrange(0, n_train_each_epoch):

			# Sample minibatch of examples
			x_batch, label_batch = sample_x_and_label_from_data_distribution(batchsize)

			# Reconstruction phase
			z_fake_batch = gen(x_batch)
			_x_batch = dec(z_fake_batch)

			## 復号誤差を最小化する
			loss_reconstruction = F.mean_squared_error(x_batch, _x_batch)
			sum_loss_reconstruction += loss_reconstruction.data

			optimizer_dec.zero_grads()
			optimizer_gen.zero_grads()
			loss_reconstruction.backward()
			optimizer_dec.update()
			optimizer_gen.update()

			# Adversarial phase
			for k in xrange(n_steps_to_optimize_dis):
				if k > 1:
					x_batch, label_batch = sample_x_and_label_from_data_distribution(batchsize)

				z_real_batch = sample_z_from_10_2d_gaussian_mixture(batchsize, label_batch, 10, config.use_gpu)

				## Discriminator loss
				p_real_batch = dis(z_real_batch)
				## p_real_batch[0] -> 本物である度合い
				## p_real_batch[1] -> 偽物である度合い
				loss_dis_real = F.softmax_cross_entropy(p_real_batch, Variable(xp.zeros(batchsize, dtype=np.int32)))

				## 上で一度z_fake_batchは計算しているため省く
				if k > 1:
					z_fake_batch = gen(x_batch)

				p_fake_batch = dis(z_fake_batch)
				## p_fake_batch[0] -> 本物である度合い
				## p_fake_batch[1] -> 偽物である度合い
				loss_dis_fake = F.softmax_cross_entropy(p_fake_batch, Variable(xp.ones(batchsize, dtype=np.int32)))

				loss_dis = loss_dis_fake + loss_dis_real
				sum_loss_regularization += loss_dis.data / float(k + 1)
				
				optimizer_dis.zero_grads()
				loss_dis.backward()
				optimizer_dis.update()


			## p_fake_batch[0] -> 本物である度合い
			## p_fake_batch[1] -> 偽物である度合い
			## generatorの学習では偽のデータを本物であると思い込ませる
			loss_gen = F.softmax_cross_entropy(p_fake_batch, Variable(xp.zeros(batchsize, dtype=np.int32)))
			sum_loss_regularization += loss_gen.data

			optimizer_gen.zero_grads()
			loss_gen.backward()
			optimizer_gen.update()

		# Saving the models
		print "epoch", epoch
		print "	reconstruction loss", (sum_loss_reconstruction / n_train_each_epoch)
		print "	regularization loss", (sum_loss_regularization / n_train_each_epoch)
		serializers.save_hdf5("%s/gen_epoch_%d.model" % (args.model_dir, epoch), gen)
		serializers.save_hdf5("%s/dis_epoch_%d.model" % (args.model_dir, epoch), dis)
		serializers.save_hdf5("%s/dec_epoch_%d.model" % (args.model_dir, epoch), dec)
		elapsed_time = time.time() - start_time
		print "	time", elapsed_time
		total_time += elapsed_time
		print "	total_time", total_time


try:
	os.mkdir(args.model_dir)
except:
	pass

dataset, labels = load_mnist_dataset(args)
train(dataset, labels)