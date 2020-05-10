import sys
import numpy as np

from .lib import backend

class NumPyWrap(object):
	"""
	NumPy wrapper to mimic module-level functions with safe conversions
	of tensor args/kwargs to ndarrays
	"""
	def __init__(self, np):
		self.__np = np

	def __tensor_fn(self, np_fn):
		"""
		Dynamic decorator method for NumPy functions
		"""
		def fn(*args, **kwargs):
			B = backend.get()
			args = list(args)
			self.__safe_unwrap(B, 0, list(kwargs.keys()), args, kwargs)
			return np_fn(*args, **kwargs)

		return fn

	def __safe_unwrap(self, B, idx, keys, args, kwargs):
		"""
		Iterate through args and kwargs and convert tensors to ndarrays
		"""
		if idx < len(args) and B.is_raw_tensor(args[idx]):
			args[idx] = B.wrap(args[idx]).to_numpy()
		if keys:
			kwarg = keys.pop()
			if B.is_raw_tensor(kwargs[kwarg]):
				kwargs[kwarg] = B.wrap(kwargs[kwarg]).to_numpy()
		if idx < len(args) or keys:
			self.__safe_unwrap(B, idx+1, keys, args, kwargs)

	def __getattr__(self, k):
		try:
			return object.__getattr__(self, k)
		except AttributeError:
			np_attr = getattr(self.__np, k)
			if callable(np_attr):
				return self.__tensor_fn(np_attr)
			else:
				return NumPyWrap(np_attr)
