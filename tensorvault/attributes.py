import sys


def _UndefinedAttribute(x):
	return AttributeError('Attempted to access undefined dataset attribute {}'.format(x))

class UndefinedAttribute(object):
	"""
	"""
	def __init__(self, attr):
		self.value_ = attr

	def __getattribute__(self, name):
		if name == 'value_':
			return object.__getattribute__(self, name)
		else:
			raise _UndefinedAttribute(self.value_)

	def _raise_error(self, *args, **kwargs):
		raise _UndefinedAttribute(self.value_)

	def __getitem__(self, *args, **kwargs):
		raise _UndefinedAttribute(self.value_)

	def __setitem__(self, *args, **kwargs):
		raise _UndefinedAttribute(self.value_)

	def __truediv__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __floordiv__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __add__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __sub__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __mul__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __mod__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __pow__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __lt__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __gt__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __le__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __ge__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __ne__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __neg__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __pos__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __invert__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __eq__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __isub__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __iadd__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __imul__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __idiv__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __imod__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __ipow__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __ifloordiv__(self, o):
		raise _UndefinedAttribute(self.value_)

	def __str__(self, *args, **kwargs):
		raise _UndefinedAttribute(self.value_)

	def __repr__(self, *args, **kwargs):
		raise _UndefinedAttribute(self.value_)
