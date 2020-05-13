import os
import sys
import json
import types
import numpy as np

from os.path import join, exists, isfile

from collections import defaultdict, OrderedDict
from itertools import chain

from . import backend
from .attrs import UndefinedAttribute, NullAssignment
from .index import Indexer, default_indices
from .utils import all_slice, exception_message, I_

class InvalidFeatureAccessType(Exception): pass
class InvalidFeatureIdentifier(Exception): pass
class InvalidFeatureAddress(Exception): pass
class InvalidDeletionShape(Exception): pass
class UnsupportedDataType(Exception): pass
class InvalidTensorType(Exception): pass


def _check(*args):
	if (len(args)==3 and args[0] != args[1]) or (len(args)==2 and not args[0]):
		raise args[-1]


class DataEngine(object):
	"""
	Data class for operations on abstract tensors under a typical dataset-like interface
	"""
	def __init__(self, **features):
		"""
		Build a dataset from numpy arrays as features (see 'load' for initialization from files)
		Features of > 2 dimensions are flattened along the inner axis resulting in a 2d feature
		for consistency. Features of one dimension are expanded into two (i.e. 1 column).
		"""
		global B
		B = backend.get()
		self._indexer = Indexer(*default_indices())
		self._features = OrderedDict()
		_outer_shape = 0
		for k,v in features.items():
			_check(B.is_tensor(v), InvalidTensorType)
			assert v.shape()[0] == (_outer_shape or v.shape()[0]), 'Feature arrays must have equal length'
			if not v.shape()[0]:
				v = B.from_numpy(np.ndarray((0,v.shape()[1]), dtype=v.dtype()))
			if len(v.shape()) > 2:
				v = v.reshape(v.shape()[0], -1)
			if len(v.shape()) == 1:
				v = v.reshape(-1, 1)
			assert v.to_numpy().dtype == features[k].to_numpy().dtype, f'{v.to_numpy().dtype}, {features[k].to_numpy().dtype}'
			self.set_feature(k,v)


	@classmethod
	def load(cls, fileset, callback, lazy_max):
		"""
		"""
		fileset = fileset if isinstance(fileset, list) else [fileset]
		out_data = []

		if lazy_max > 0:
			raise NotImplementedError

		for file in fileset:
			if file.endswith('json'):
				with open(file, 'r') as f:
					data = json.loads(f.read())
					data = callback(data)
				if isinstance(data, list) and isinstance(data[0], dict):
					instance_count, feature_ids = len(data), list(data[0].keys())
					out_data.append(
						{f:B.from_list([data[i][f] for i in range(instance_count)])
						 for f in feature_ids})

				elif isinstance(data, dict) and isinstance(data[0], list):
					out_data.append({f:B.from_list(v) for f,v in data.items()})
				else:
					raise Exception('Unsupported data input format.')

			elif file.endswith('csv'):
				raise NotImplementedError

			elif file.endswith('txt'):
				raise NotImplementedError

			else:
				raise Exception('Unsupported file input format.')

		if len(out_data) > 1:
			return cls(**{k:B.concat([d[k] for d in out_data], axis=0) for k in out_data[0].keys()})
		else:
			return cls(**out_data[0])

	@property
	def data(self):
		"""
		Homogenized data as an array/tensor backing the Dataset
		"""
		return self.coalesce(*self.tensors)

	@property
	def features(self):
		"""
		List of all feature names in the dataset
		"""
		return list(self._features.keys())

	@property
	def tensors(self):
		"""
		List of all feature tensors backing the Dataset
		"""
		return list(self._features.values())

	@property
	def size(self):
		"""
		The abstract dimensions of the dataset (num_instances, num_features)
		"""
		return len(self), len(self.features)

	@property
	def shape(self):
		"""
		The underlying dimensions of the dataset (num_instances, sum of each feature width)
		"""
		return len(self), sum(self.structure[-1])

	@property
	def structure(self):
		"""
		The underlying dimension structure of the dataset (num_instances, (feature_width_1, feature_width_2...))
		"""
		return len(self), tuple(t.shape()[-1] for t in self.tensors)

	@property
	def dtypes(self):
		"""
		The native datatypes in the dataset (dtype_feature_1, dtype_feature_2...)
		"""
		return tuple(t.dtype() for t in self.tensors)

	@property
	def copy(self):
		"""
		Make a copy of the DataEngine
		"""
		return DataEngine(**{f:B.make_copy(self.get_feature(f)) for f in self.features})

	def get_feature(self, f, default=InvalidFeatureIdentifier):
		"""
		"""
		if isinstance(f, str):
			try:
				return self._features[f]
			except KeyError:
				if isinstance(default, Exception):
					raise default()
				else:
					return default
		elif B.is_tensor(f):
			addr = f.address()
			for f in self.features:
				faddr = self.get_feature(f).address()
				if addr == faddr:
					return f
			raise InvalidFeatureAddress()
		raise InvalidFeatureAccessType(type(f))

	def set_feature(self, f, v):
		"""
		"""
		assert isinstance(f, str), 'Feature ID must be a string'
		assert B.is_tensor(v) or v is None, 'Feature must be tensor or None'
		if v is not None:
			if v.address() in [self.get_feature(id).address() for id in self._features]:
				v = B.make_copy(v)
			self._features[f] = v
		else:
			del self._features[f]

	def coalesce(self, *data, axis=1):
		"""
		"""
		ranks = [str, float, int]
		dtypes = set([d.dtype() for d in data])

		if len(data) == 1:
			return data[0] if isinstance(data, tuple) else data

		for dtype in ranks:
			if dtype in dtypes:
				return B.concat(tuple(d.type(dtype) for d in data), axis=axis)

		invalid = [str(t) for t in dtypes if t not in ranks]
		raise UnsupportedDataType('Array(s) with type {}'.format(' or '.join(invalid)))

	def __delete_items(self, rows, items):
		"""
		"""
		_check(len(items) == self.size[-1] or rows == all_slice,
			   InvalidDeletionShape('Row deletion must include all features'))
		if len(items) == self.size[-1]:
			for f in self.features:
				arr = self.get_feature(f)
				self.set_feature(f, B.delete(arr, rows, axis=0))
		else:
			for item in items:
				feature = self.get_feature(item)
				self.set_feature(feature, None)

	def __apply(self, v, method, inplace=True):
		"""
		"""
		new = {f:B.apply(self.get_feature(f), method, v) for f in self.features}
		if any(v is not None for _,v in new.items()):
			return DataEngine(**new)

	def __getitem__(self, i):
		"""
		"""
		idx0, idx1, _ = self._indexer(i, NullAssignment, all_columns=self.tensors)
		if B.is_tensor(idx0):
			idx0 = idx0.reshape(-1)
			_check(idx0.shape()[0], len(self), Exception(exception_message(f"""
			Bad tensor indexer shape {idx0.shape()}; dimension 0 should equal
			dataset length {len(self)} (the number of instances in dataset)""")))
		return DataEngine(**{self.get_feature(tensor):tensor.index(I_[idx0,:])
					   		 for tensor in idx1})

	def __setitem__(self, i, val):
		idx0, idx1, val = self._indexer(i, val, all_columns=self.tensors)
		if val is None:
			self.__delete_items(idx0, idx1)
		else:
			curr_length = len(self)
			for i,a in enumerate(idx1):
				# vstack - add new feature(s)
				if isinstance(a, UndefinedAttribute):
					self.set_feature(a.value_, val[i])
				# hstack - add new instances
				elif idx0 == slice(curr_length, None, None):
					f = self.get_feature(a)
					self.set_feature(f, self.coalesce(a, val[i], axis=0))
				# value assignments
				else:
					a.index(I_[idx0,:], v=val[i])

	def __truediv__(self, o):
		return self.__apply(o, '__truediv__')

	def __floordiv__(self, o):
		return self.__apply(o, '__floordiv__')

	def __add__(self, o):
		return self.__apply(o, '__add__')

	def __sub__(self, o):
		return self.__apply(o, '__sub__')

	def __mul__(self, o):
		return self.__apply(o, '__mul__')

	def __mod__(self, o):
		return self.__apply(o, '__mod__')

	def __pow__(self, o):
		return self.__apply(o, '__pow__')

	def __lt__(self, o):
		return self.__apply(o, '__lt__')

	def __gt__(self, o):
		return self.__apply(o, '__gt__')

	def __le__(self, o):
		return self.__apply(o, '__le__')

	def __ge__(self, o):
		return self.__apply(o, '__ge__')

	def __ne__(self, o):
		return self.__apply(o, '__ne__')

	def __neg__(self, o):
		return self.__apply(o, '__neg__')

	def __pos__(self, o):
		return self.__apply(o, '__pos__')

	def __invert__(self, o):
		return self.__apply(o, '__invert__')

	def __eq__(self, o):
		if B.is_tensor(o):
			return self.__apply(o, '__eq__')
		elif isinstance(o, DataEngine):
			if o.shape() != self.shape: # structure may differ
				return False
			else:
				s = [x for _,x in sorted(list(zip(self.features, self.tensors)))]
				o = [x for _,x in sorted(list(zip(o.features, o.tensors)))]
				return all([B.equal(a,b) for a,b in zip(s,o)])
		else:
			return False

	def __isub__(self, o):
		self.__apply(o, '__isub__')
		return self.data

	def __iadd__(self, o):
		self.__apply(o, '__iadd__')
		return self.data

	def __imul__(self, o):
		self.__apply(o, '__imul__')
		return self.data

	def __idiv__(self, o):
		self.__apply(o, '__idiv__')
		return self.data

	def __imod__(self, o):
		self.__apply(o, '__imod__')
		return self.data

	def __ipow__(self, o):
		self.__apply(o, '__ipow__')
		return self.data

	def __ifloordiv__(self, o):
		self.__apply(o, '__ifloordiv__')
		return self.data

	def __str__(self):
		print_context_v = 15
		print_context_h = 6
		maxlens = [11, 5, 5] # str, float, int
		idx_head = 'index'
		sep = '|'

		if len(self) == 0:
			return 'Empty Dataset with features {}'.format(self.features)

		data = self.data

		dtypes, widths = self.dtypes, self.structure[-1]
		dtypes = list(chain(*[[dtypes[i]]*widths[i] for i,f in enumerate(self.features)]))
		header = list(chain(*[[f]*widths[i] for i,f in enumerate(self.features)]))


		indices = list(range(len(self)))
		idx_col_w = max(len(str(data.shape()[0])), len(idx_head))

		if self.shape[0] > print_context_v*2:
			data = B.concat((data.index(I_[:print_context_v,:]), data.index(I_[-print_context_v:,:])), axis=0)
			indices = indices[:print_context_v] + indices[-print_context_v:]
		if self.shape[1] > print_context_h*2:
			data = B.concat((data.index(I_[:,:print_context_h]), data.index(I_[:,-print_context_h:])), axis=1)
			dtypes = dtypes[:print_context_h] + dtypes[-print_context_h:]
			header = header[:print_context_h] + header[-print_context_h:]

		data = data.to_numpy().astype(str)
		data = np.vstack((np.array(header).reshape(1,-1), data))

		out = ''

		for i in range(data.shape[0]):
			if i == print_context_v+1:
				width = (out.find('\n')-2)
				out += ('\n {0:^'+str(width)+'} \n').format('.....')

			out += '\n '+sep if i else ' '+sep
			out += (' {0:<'+str(idx_col_w)+'} |').format(indices[i-1] if i else idx_head)

			for j,x in enumerate(data[i]):
				if j == print_context_h and self.shape[1] > print_context_h*2:
					out += ' ... '+(sep if header[j-1] != header[j] else '')
				if dtypes[j] == float:
					s = '{:.3f}'.format(float(x)) if i else x
					maxl = maxlens[1]
				elif dtypes[j] == int:
					s = str(int(float(x))) if i else x
					maxl = maxlens[2]
				else:
					maxl = maxlens[0]
					s = x.strip() if i else x
				if len(s) > maxl:
					s = s[:(maxl-2)]+'..'

				out += (' {0:<'+str(maxl+1)+'}').format(s)
				out += sep if (j+1) == len(header) or header[j] != header[j+1] else ''

		return '\n'+out


	def __len__(self):
		return self.get_feature(self.features[0]).shape()[0] if self.features else 0
