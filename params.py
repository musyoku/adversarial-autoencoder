class Params():
	def __init__(self, dict=None):
		if dict:
			self.from_dict(dict)

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			if hasattr(self, attr):
				setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			if hasattr(value, "to_dict"):
				dict[attr] = value.to_dict()
			else:
				dict[attr] = value
		return dict

	def dump(self):
		for attr, value in self.__dict__.iteritems():
			print "	{}: {}".format(attr, value)
