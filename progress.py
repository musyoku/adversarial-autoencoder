import sys, time, math

class Progress(object):

	def __init__(self):
		self.start_time = 0
		self.epoch_start_time = 0

	def start_epoch(self, current_epoch, total_epoch):
		current = time.time()
		if self.start_time == 0:
			self.start_time = current
		self.epoch_start_time = current
		print "Epoch {}/{}".format(current_epoch, total_epoch)

	def get_progress_bar(self, current_step, total_steps, num_segments=30):
		str = "["
		base = total_steps / float(num_segments)
		for seg in xrange(num_segments):
			if base * (seg + 1) < current_step:
				str += "="
			else:
				if str[-1] == "=":
					str = str + ">"
				else:
					str += "."
		str = str[:num_segments] + "]"
		return str

	def get_total_time(self):
		return int((time.time() - self.start_time) / 60)

	def get_args(self, args):
		str = ""
		for key, value in args.iteritems():
			if isinstance(value, float):
				str += " - {}: {:.3f}".format(key, value)
			else:
				str += " - {}: {}".format(key, value)
		return str

	def get_elapsed_minute(self):
		return int((time.time() - self.epoch_start_time) / 60)

	def show(self, current_step, total_steps, args):
		digits = int(math.log10(total_steps)) + 1
		progress_bar = self.get_progress_bar(current_step, total_steps)
		prefix = "{0:>{1}}/{2} {3}".format(current_step, digits, total_steps, progress_bar)
		args = self.get_args(args)
		if current_step == total_steps:
			sys.stdout.write("\r")
			print "{} - {}m{}".format(prefix, self.get_elapsed_minute(), args)
		else:
			sys.stdout.write("\r{} - {}m{}".format(prefix, self.get_elapsed_minute(), args))
			sys.stdout.flush()
