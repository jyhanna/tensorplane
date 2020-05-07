import os
import sys
import json
import types
import numpy as np

from os.path import join, exists, isfile
import matplotlib.pyplot as plt

from collections import defaultdict, OrderedDict
from itertools import chain

from .lib import backend
from .lib.engine import DataEngine
from .lib.attrs import UndefinedAttribute, NullAssignment

class Dataset(object):

	def __init__(self, **features):
		global B
		B = backend.get()
		self._source = DataEngine(**B.wrap(features))

	@staticmethod
	def load(fileset, callback=(lambda x: x), lazy_max=0):
		"""
		"""
		return Dataset._from_source(DataEngine.load(fileset, callback, lazy_max))

	@classmethod
	def _from_source(cls, src):
		"""
		"""
		inst = cls()
		inst._source = src
		return inst

	@property
	def data(self):
		"""
		Homogenized data as a single array/tensor backing the Dataset
		"""
		return B.unwrap(self._source.data)

	@property
	def tensors(self):
		"""
		List of all feature tensors backing the Dataset
		"""
		return B.unwrap(self._source.tensors)

	@property
	def features(self):
		"""
		List of all feature names in the dataset
		"""
		return self._source.features

	@property
	def size(self):
		"""
		The abstract dimensions of the dataset (num_instances, num_features)
		"""
		return self._source.size

	@property
	def shape(self):
		"""
		The underlying dimensions of the dataset (num_instances, sum of each feature width)
		"""
		return self._source.shape

	@property
	def structure(self):
		"""
		The underlying dimension structure of the dataset (num_instances, (feature_width_1, feature_width_2...))
		"""
		return self._source.structure

	@property
	def dtypes(self):
		"""
		The native datatypes in the dataset (dtype_feature_1, dtype_feature_2...)
		"""
		return self._source.dtypes

	@property
	def copy(self):
		"""
		Make a copy of the dataset (copy each feature array/tensor)
		"""
		return Dataset._from_source(self._source.copy)

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
		for i,f in enumerate(self.features)}
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

	def __getitem__(self, i):
		return self._source[B.wrap(i)]

	def __setitem__(self, i, val):
		self._source[B.wrap(i)] = B.wrap(val)

	def __getattribute__(self, k):
		tensor = self._source.get_feature(k, None) if k != '_source' else None
		if tensor is None:
			try:
				return object.__getattribute__(self, k)
			except AttributeError:
				return UndefinedAttribute(k)
		else:
			return B.unwrap(tensor)

	def __setattr__(self, k, v):
		if isinstance(v, UndefinedAttribute):
			self._source.set_feature(k, v.value_)
		elif B.is_raw_tensor(v):
			self._source.set_feature(k, B.wrap(v))
		else:
			object.__setattr__(self, k, v)

	def __str__(self):
		return str(self._source)

	def __len__(self):
		return len(self._source)
