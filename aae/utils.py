import sys
import numpy as np

def clear_console():
	printr("")

def printr(string):
	sys.stdout.write("\r\033[2K")
	sys.stdout.write(string)
	sys.stdout.flush()

def onehot(labels, ndim=10):
	vec = np.zeros((len(labels), ndim), dtype=np.float32)
	vec[np.arange(len(labels)), labels] = 1
	return vec