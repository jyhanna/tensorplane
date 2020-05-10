import os
import sys
import time
import json
import math
import hashlib
import random
import pytest
import decorator

import torch
import numpy as np
from tensorplane.core.wrap import NumPyWrap

np = NumPyWrap(np)

from itertools import product

from tensorplane.core.lib import backend
from tensorplane.core.lib.utils import I_, all_slice
from tensorplane.core.data import Dataset

from .utils import (
	assert_all_eq,
	assert_false,
	assert_true,
	assert_none,
	assert_eq,
	np_convert,
	array,
	tensor,
	lrange,
	lprod,
	rand_str,
)

def configure():
	"""
	Automatically repeat testing on all implemented backends
	"""
	for b in list(backend.backends.keys()):
		globals()['TestDataset'+b] = type(b, (Template,), {'B': b})


def build_dataset(structure, dtypes):
	"""
	Create a random dataset from a given structure and dtype(s)
	"""
	ldim, fdims = structure
	ids = 'abcdefghijklmnopqrstuvwxyz'
	dtypes = dtypes if isinstance(dtypes, list) else [dtypes]*len(fdims)
	kwargs = {}

	for i, dtype in enumerate(dtypes):
		fdim = fdims[i]
		if ldim == 0:
			t = np.ndarray((0,fdim))
		elif dtype == float:
			t = np.random.randn(ldim,fdim)
		elif dtype == int:
			t = np.random.randint(10, size=(ldim,fdim))
		elif dtype == str:
			s = 'abcdefghijklmnopqrstuvwxyz0123456789 []{}()*&^%$#@!,./"\'|\\'
			t = np.array([[rand_str(s) for _ in range(fdim)] for _ in range(ldim)])
		else:
			raise Exception('Bad datatype given to test func')
		kwargs[ids[i]] = tensor(t)

	return Dataset(**kwargs)


def generate_datasets():
	"""
	Generate a variety of testing datasets
	"""
	dtypes = [int, float, str, [int, float, int, float]]
	structures = [
		(1, (1,)),
		(3, (2,)),
		(0, (1, 2)),
		(1, (1, 2)),
		(2, (2, 2)),
		(1, (3, 4, 1)),
		(1, (1, 1, 10)),
		(2, (3, 4, 1)),
		(2, (1, 1, 10)),
		(3, (3, 4, 1)),
		(3, (1, 1, 10)),
		#(1000, (3, 1, 1)),
		#(1000, (1, 2, 10)),
		#(1000, (1, 100, 1)),
		#(1000, (1, 2, 1, 2))
	]
	for t in dtypes:
		for s in structures:
			_,fdim = s
			try:
				if not isinstance(t, list):
					B.from_numpy(np.array([t()]))
			except TypeError:
				assert_eq(t, str, 'Type error for tensor other than string')
				break
			x = t[:len(fdim)] if isinstance(t, list) else ([t]*len(fdim))
			yield build_dataset(s, x), s, t


def dataset_parameterize(foreach_feature=False,
						 with_structure=False,
						 with_type=False):
	"""
	Parameterize a dataset from setup generator
	"""
	def parameterize_dec(func):
		def test_wrapper(func, *args, **kwargs):
			for d,s,t in generate_datasets():
				d_args = (d,s) if with_structure else (d,)
				d_args = d_args+(t,) if with_type else d_args
				if foreach_feature:
					for f in np_convert(d.tensors, copy=False):
						func(args[0], *(d_args+(f,)), **kwargs)
				else:
					func(args[0], *d_args, **kwargs)
		return decorator.decorator(test_wrapper, func)
	return parameterize_dec


# fixtures overwritten by custom decorator
@pytest.fixture
def d(): return
@pytest.fixture
def s(): return
@pytest.fixture
def f(): return
@pytest.fixture
def t(): return

class Template:

	@classmethod
	def setup_class(cls):
		global B
		backend.set(cls.B)
		B = backend.get()
		np.random.seed(0)
		random.seed(0)

	### Dimension properties tests

	@dataset_parameterize(with_structure=True)
	def test_structure(self, d, s):
		assert_eq(d.structure, s, 'Incorrect dataset structure')

	@dataset_parameterize(with_structure=True)
	def test_shape(self, d, s):
		assert_eq(d.shape, (s[0],sum(s[1])), 'Incorrect dataset shape')

	@dataset_parameterize(with_structure=True)
	def test_size(self, d, s):
		assert_eq(d.size, (s[0],len(s[1])), 'Incorrect dataset size')

	@dataset_parameterize(with_structure=True)
	def test_len(self, d, s):
		assert_eq(len(d), s[0], 'Incorrect dataset shape')
		assert_eq(len(set(len(f) for f in d.tensors)), 1, 'Inconsistent feature tensor lengths')
		assert_eq(len(d.tensors[0]), s[0], 'Feature tensor lengths do not match dataset length')

	### General property and method tests

	def test_tensors(self):
		pass

	def test_data(self):
		pass

	def test_batch(self):
		pass

	def test_split(self):
		pass

	### Attribute tests

	def test_new_feature_assignment(self):
		pass

	def test_new_feature_deletion(self):
		pass

	### Indexing tests

	def _index_check(self, d, idx1, idx2, other_ts, msg):
		np_fn = lambda a: a[(array(idx1) if B.is_raw_tensor(idx1) else idx1),:]
		np_idx = idx2 if idx2 != all_slice else d.tensors
		tgt = (np_convert(np_idx, fn=np_fn, copy=True) +
			   np_convert(other_ts, copy=True))
		res = np_convert(d[idx1, idx2].tensors) + np_convert(other_ts)
		assert_all_eq(zip(res, tgt), msg)

	def _complete_index_check(self, d, idx1, msg):
		if not len(d): return
		self._index_check(d, idx1, I_[:], [],
		f'Incorrect index full dataset {msg}')

		if d.size[1] < 2: return

		self._index_check(d, idx1, d.tensors[:1], d.tensors[1:],
		f'Incorrect index subset ')

		if d.size[1] < 3: return

		self._index_check(d, idx1, d.tensors[:2], d.tensors[2:],
		f'Incorrect index subset (first) {msg}')
		self._index_check(d, idx1, d.tensors[1:2], d.tensors[:1]+d.tensors[2:],
		f'Incorrect index subset (middle) {msg}')
		self._index_check(d, idx1, d.tensors[2:], d.tensors[:2],
		f'Incorrect index subset (last) {msg}')

	@dataset_parameterize(foreach_feature=True)
	def test_index_instance_sorting(self, d, f):
		if f.shape[1] != 1: return
		ids = np.argsort(f).reshape(-1)
		self._complete_index_check(d, ids, 'sorting using argsort')

	@dataset_parameterize(foreach_feature=True, with_type=True)
	def test_index_instance_boolean(self, d, f, t):
		print(t)
		self._complete_index_check(d, f[:, 0]<1, 'boolean indexing')

	@dataset_parameterize()
	def test_index_instance_shuffling(self, d):
		shuffle_idxs = np.arange(len(d))
		np.random.shuffle(shuffle_idxs)
		ids = tensor(shuffle_idxs)
		self._complete_index_check(d, ids, 'shuffling')

	@dataset_parameterize()
	def test_index_instance_slices(self, d):
		self._complete_index_check(d, I_[::-1], 'reverse slicing')
		self._complete_index_check(d, I_[::-2], 'reverse slicing skip')
		self._complete_index_check(d, I_[:1], 'simple one-sided slicing')
		self._complete_index_check(d, I_[1:], 'simple one-sided slicing')

		if len(d) < 3: return

		self._complete_index_check(d, I_[1:-1], 'inner rows slicing')
		self._complete_index_check(d, I_[:1], 'single element slice')

		if len(d) < 4: return

		self._complete_index_check(d, I_[1:-1:-1], 'inner rows reverse slicing')

		self._complete_index_check(d, I_[2:3], 'simple two-sided slicing')
		self._complete_index_check(d, I_[2:-1], 'simple two-sided slicing')


	def test_index_instance_creation(self):
		pass

	def test_index_feature_creation(self):
		pass

	def test_index_instance_assignment(self):
		pass

	def test_index_feature_assignment(self):
		pass

	def test_index_instance_deletion(self):
		pass

	def test_index_feature_deletion(self):
		pass

	def test_index_scalar_arithmetic(self):
		pass

	def test_index_scalar_assignment(self):
		pass

	### Loading and saving tests

	def test_load_json(self):
		pass

	def test_load_csv(self):
		pass

	def test_load_txt(self):
		pass

	def test_save_json(self):
		pass

	def test_save_csv(self):
		pass

	def test_save_txt(self):
		pass


configure()
