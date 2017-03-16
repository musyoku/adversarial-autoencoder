# -*- coding: utf-8 -*-
import os
import numpy as np
import mnist_tools

def load_train_images():
	return mnist_tools.load_train_images()

def load_test_images():
	return mnist_tools.load_test_images()

def create_semisupervised(images, labels, num_validation_data=10000, num_labeled_data=100, num_types_of_label=10):
	assert len(images) >= num_validation_data + num_labeled_data
	training_labeled_x = []
	training_unlabeled_x = []
	validation_x = []
	validation_labels = []
	training_labels = []
	indices_for_label = {}
	num_data_per_label = int(num_labeled_data / num_types_of_label)
	num_unlabeled_data = len(images) - num_validation_data - num_labeled_data

	indices = np.arange(len(images))
	np.random.shuffle(indices)

	def check(index):
		label = labels[index]
		if label not in indices_for_label:
			indices_for_label[label] = []
			return True
		if len(indices_for_label[label]) < num_data_per_label:
			for i in indices_for_label[label]:
				if i == index:
					return False
			return True
		return False

	for n in xrange(len(images)):
		index = indices[n]
		if check(index):
			indices_for_label[labels[index]].append(index)
			training_labeled_x.append(images[index])
			training_labels.append(labels[index])
		else:
			if len(training_unlabeled_x) < num_unlabeled_data:
				training_unlabeled_x.append(images[index])
			else:
				validation_x.append(images[index])
				validation_labels.append(labels[index])

	return training_labeled_x, training_labels, training_unlabeled_x, validation_x, validation_labels
	
def sample_labeled_data(images, labels, batchsize, ndim_y=10):
	ndim_x = 28 ** 2
	image_batch = np.zeros((batchsize, ndim_x), dtype=np.float32)
	label_onehot_batch = np.zeros((batchsize, ndim_y), dtype=np.float32)
	label_id_batch = np.zeros((batchsize,), dtype=np.int32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=False)
	for j in range(batchsize):
		data_index = indices[j]
		image_batch[j] = images[data_index].astype(np.float32)
		label_onehot_batch[j, labels[data_index]] = 1
		label_id_batch[j] = labels[data_index]
	return image_batch, label_onehot_batch, label_id_batch

def sample_unlabeled_data(images, batchsize):
	ndim_x = 28 ** 2
	image_batch = np.zeros((batchsize, ndim_x), dtype=np.float32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=False)
	for j in range(batchsize):
		data_index = indices[j]
		image_batch[j] = images[data_index].astype(np.float32)
	return image_batch