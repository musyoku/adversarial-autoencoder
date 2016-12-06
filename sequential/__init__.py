import links
import layers
import functions
import util
import chain
from sequential import *

def from_json(str):
	seq = Sequential()
	seq.from_json(str)
	return seq

def from_dict(dict_array):
	seq = Sequential()
	seq.from_dict(dict_array)
	return seq