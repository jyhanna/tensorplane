import os
import sys
import json
import types
import numpy as np

from os.path import join, exists, isfile
import matplotlib.pyplot as plt

from collections import defaultdict
from itertools import chain

import tensorplane.backend
_B = os.getenv('DATAFLOW_BACKEND')
B = getattr(backend, _B)()

from tensorplane.backend import AbstractTensor
from attributes import UndefinedAttribute

from utils import all_slice, I_

from index import (
	IndexEngine,
	default_indices,
	NullAssignment,
	UndefinedAttribute,
)

class InvalidFeatureAccessType(Exception): pass
class InvalidFeatureIdentifier(Exception): pass
class InvalidFeatureAddress(Exception): pass
class InvalidDeletionShape(Exception): pass
class UnsupportedDataType(Exception): pass
class InvalidTensorType(Exception): pass

def _check(*args):
	if (len(args)==3 and args[0] != args[1]) or (len(args)==2 and not args[0]):
		raise args[-1]


class Dataset(object):
	"""
	A class for managing and encapsulating heterogeneous data from various sources.
	A Dataset provides uniform and consistent access and manipulation of complex
	subsets of data using expressive and powerful NumPy-inspired indexing conventions.
	Datasets currently support NumPy ndarray and PyTorch tensor datatypes and
	will support TensorFlow tensors in the near future (among other scientific
	computation library datatypes). A specific backend can be chosen from the
	backends.backends dictionary keys by setting the 'DATAFLOW_BACKEND' environment
	variable to one of the implemented backend datatype options.

	Datasets are similar to R (or Pandas) dataframes, but designed specifically for
	machine learning applications in Python and offer a high degree of flexibility,
	support, and efficiency by virtue of very common datatype backends like NumPy ndarrays.

	Crucially, a Dataset object does not entirely hide its backend data management.
	Instead, it exposes and encourages underlying array/tensor access through an
	advanced data indexing interface. It also supports various high-level common
	dataset functions, like splitting and batching data in addition to its low-
	level management. Ultimately, a Dataset is nothing but a lightweight wrapper
	over arrays or tensors, but with a machine-learning-oriented layer of abstraction
	that matches common dataset use cases for various deep learning and other
	machine learning and data mining applications.

	NOTE: This library is still in its very early phases of development, and
	has not yet been comprehensively tested. Production deployment of software
	using Dataflow is strongly discouraged.
	"""
	def __init__(self, **features):
		"""
		Build a dataset from numpy arrays as features (see 'load' for initialization from files)
		Features of > 2 dimensions are flattened along the inner axis resulting in a 2d feature
		for consistency. Features of one dimension are expanded into two (i.e. 1 column).
		"""
		self._indexer = IndexEngine(*default_indices())
		self._features = []
		_outer_shape = 0
		for k,v in features.items():
			if B.is_concrete(v):
				v = B.from_concrete(v)
			else:
				_check(B.is_abstract(v), InvalidTensorType)
			assert v.shape()[0] == (_outer_shape or v.shape()[0]), 'Feature arrays must have equal length'
			if len(v.shape()) > 2:
				v = v.reshape(v.shape()[0], -1)
			if len(v.shape()) == 1:
				v = v.reshape(-1, 1)
			self.__set_feature(k, v if v.shape()[0] else B.from_list([]))


	@classmethod
	def load(cls, fileset, callback=(lambda x: x), lazy_max=0):
		"""
		Class constructor to load a Dataset from given file(s) (json, csv, txt)
		TODO: dynamic lazy loading of a subset of instances to conserve memory
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
		return self.__aggregate(*self.features)

	@property
	def features(self):
		"""
		List of all AbstractTensor features backing the Dataset
		"""
		return [self.__get_feature(f) for f in self.feature_ids]

	@property
	def feature_ids(self):
		"""
		List of all feature names in the dataset
		"""
		return self._features

	@property
	def size(self):
		"""
		The abstract dimensions of the dataset (num_instances, num_features)
		"""
		return len(self), len(self.feature_ids)

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
		return len(self), tuple(self.__get_feature(f).shape()[-1] for f in self.feature_ids)

	@property
	def dtypes(self):
		"""
		The native datatypes in the dataset (dtype_feature_1, dtype_feature_2...)
		"""
		return tuple(self.__get_feature(f).dtype() for f in self.feature_ids)

	@property
	def summary(self):
		"""
		Summary of the structure and composition of the Dataset + metadata
		"""
		s, d = self.structure[-1], self.dtypes
		features = {
			f:{'width':s[i],
			   'type':d[i],
			   'index':i}
		for i,f in enumerate(self.feature_ids)}
		meta = {'instances': len(self),
				'features':  self.size[-1],
				'columns':   self.shape[-1],
				'types':     len(set(d))}
		return {'metadata': meta,
				'features': features}

	def batch(self, size):
		"""
		Simple batch generator function (to be expanded with lazy loading)
		"""
		assert size <= len(self), 'Batch size must not exceed number of instances'
		while i < len(self):
			yield self[i:i+size]
			i += size

	def copy(self):
		"""
		Make a copy of the dataset (copy each feature array/tensor)
		"""
		return Dataset(**{f:B.make_copy(self.__get_feature(f)) for f in self.feature_ids})

	def split(self, ratio, shuffle=False, seed=42):
		"""
		Split Dataset into multiple datasets (e.g. train/test or train/test/val)
		An 80/20 split would be (0.8,0.2). An 85/10/5 split would be (0.85,0.1/0.05)
		"""
		assert ratio and isinstance(ratio, tuple), 'Must have at least 1 split'
		assert sum(ratio) == 1, 'Splits must sum to 1'

		if shuffle:
			shuffle_idxs = np.arange(len(dataset))
			np.random.shuffle(shuffle_idxs)
			dataset = self[B.from_numpy(shuffle_idxs.reshape(-1)).data,:]
		else:
			dataset = self

		nsplits = len(ratio)
		fsplits = []
		indices = [0]

		for i, r in enumerate(ratio):
			indices.append(int(len(dataset)*r) + indices[-1])

		if indices[-1] == len(self)-1:
			indices.pop(-1)
		indices.append(len(dataset))

		return [dataset[indices[i]:indices[i+1]] for i in range(nsplits)]

	###### Private methods for handling indexing, aggregation, etc. ######

	def __get_feature(self, f):
		"""
		"""
		if isinstance(f, str):
			try:
				return B.from_concrete(getattr(self, f))
			except AttributeError:
				raise InvalidFeatureIdentifier()
		elif B.is_abstract(f):
			addr = f.address()
			for f in self.feature_ids:
				faddr = self.__get_feature(f).address()
				if addr == faddr:
					return f
			raise InvalidFeatureAddress()
		raise InvalidFeatureAccessType(type(f))

	def __set_feature(self, f, v):
		"""
		"""
		assert isinstance(f, str), 'Feature ID must be a string'
		assert B.is_abstract(v) or v is None, 'Feature must be tensor or None'
		if v is not None:
			#_check(v.shape[0], len(self), Exception('Invalid feature length'))
			if v.address() in [self.__get_feature(id).address() for id in self._features]:
				v = B.make_copy(v)
			if f not in self._features:
				self._features.append(f)
			setattr(self, f, v) # must go after
		else:
			self._features = [id for id in self._features if id != f]
			delattr(self, f)

	def __delete_items(self, rows, items):
		"""
		"""
		_check(len(items) == self.size[-1] or rows == all_slice,
			   InvalidDeletionShape('Row deletion must include all features'))
		if len(items) == self.size[-1]:
			for f in self.feature_ids:
				arr = self.__get_feature(f)
				self.__set_feature(f, B.delete(arr, rows, axis=0))
		else:
			for item in items:
				feature = self.__get_feature(item)
				self.__set_feature(feature, None)

	def __aggregate(self, *data, axis=1):
		"""
		"""
		ranks = [str, float, int]
		if data and data[0].shape()[0] == 0:
			return B.from_list([])
		dtypes = set([d.dtype() for d in data])

		if len(data) == 1:
			return data[0] if isinstance(data, tuple) else data

		for dtype in ranks:
			if dtype in dtypes:
				return B.concat(tuple(d.type(dtype) for d in data), axis=axis)

		invalid = [str(t) for t in dtypes if t not in ranks]
		raise UnsupportedDataType('Array(s) with type {}'.format(' or '.join(invalid)))

	def __apply(self, v, method, inplace=True):
		"""
		"""
		new = {f:B.apply(self.__get_feature(f), method, v) for f in self.feature_ids}
		if any(v is not None for _,v in new.items()):
			return Dataset(**new)

	def __getitem__(self, i):
		"""
		"""
		i = B.abstractify(i, deep=True)
		idx0, idx1, _ = self._indexer(i, NullAssignment, all_columns=self.features)
		if B.is_abstract(idx0):
			idx0 = idx0.reshape(-1)
			_check(idx0.shape()[0], len(self), Exception())
		return Dataset(**{self.__get_feature(tensor):tensor.index(I_[idx0,:])
					   for tensor in idx1})

	def __setitem__(self, i, val):
		i, val = B.abstractify((i, val), deep=True)
		idx0, idx1, val = self._indexer(i, val, all_columns=self.features)
		if val is None:
			self.__delete_items(idx0, idx1)
		else:
			curr_length = len(self)
			for i,a in enumerate(idx1):
				# hstack - add new feature(s)
				if isinstance(a, UndefinedAttribute):
					self.__set_feature(a.value_, val[i])
				# vstack - add new instances
				elif idx0 == slice(curr_length, None, None):
					f = self.__get_feature(a)
					self.__set_feature(f, self.__aggregate(a, val[i], axis=0))
				# value assignments
				else:
					a.index(I_[idx0,:], v=val[i])

	def __getattribute__(self, k):
		try:
			attr = object.__getattribute__(self, k)
			if B.is_abstract(attr):
				return attr.data
			return attr
		except AttributeError:
			return UndefinedAttribute(k)

	def __setattr__(self, k, v):
		if isinstance(k, UndefinedAttribute):
			self.__set_feature(k, v.value_)
		elif B.is_concrete(v) and k not in self.feature_ids:
			self.__set_feature(k, B.abstractify(v))
		elif v is None and k in self.feature_ids:
			self.__set_feature(k, v)
		else:
			object.__setattr__(self, k, v)

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
		if B.is_concrete(o):
			return self.__apply(o, '__eq__')
		elif isinstance(o, Dataset):
			if o.shape() != self.shape: # structure may differ
				return False
			else:
				s = [x for _,x in sorted(list(zip(self.feature_ids, self.features)))]
				o = [x for _,x in sorted(list(zip(o.feature_ids, o.features)))]
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
			return 'Empty Dataset with features {}'.format(self.feature_ids)

		data = B.from_concrete(self.data)

		dtypes, widths = self.dtypes, self.structure[-1]
		dtypes = list(chain(*[[dtypes[i]]*widths[i] for i,f in enumerate(self.feature_ids)]))
		header = list(chain(*[[f]*widths[i] for i,f in enumerate(self.feature_ids)]))


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
		return self.__get_feature(self.feature_ids[0]).shape()[0]
