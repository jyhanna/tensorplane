import os
import sys
import types
import numpy as np
from collections import defaultdict
from itertools import chain, product

import numpy as np

from .utils import is_class, is_subclass, split_tensor, list_flatten, all_slice, I_

from .backend import AbstractTensor
from .attrs import UndefinedAttribute, NullAssignment


def default_indices():
	return [IndexRowSubset,
			IndexRowXORColumnSubset,
			IndexColumnNew,
			IndexRowXORColumnSubset,
			IndexRowORColumnSubset]


def index(*vals, is_compact=False):
	"""
	Index wrapper for indexing grammars
	"""
	if is_compact:
		return list(product(*vals))
	else:
		return vals


def Map(index, fn):
	"""
	Map indices to a function
	"""
	return (index, fn)


def SimpleParse(x, top=True):
	"""
	Debugging and Exception message index grammar parser (needs refactoring)
	"""
	x = x if isinstance(x, (list, tuple)) else [x]
	res = type(x)([(type(v) if (not isinstance(v, (list, tuple)) and not type(v)==type) else
	(v if type(v)==type else type(v)(SimpleParse(v[0] if v else None, top=False)))) for v in x])
	return res if len(res)>1 or not top else res[0]


class Indexer(object):
	"""
	Main indexing class
	"""
	def __init__(self, *indices, **kwargs):
		self.__indices = [I(**kwargs) for I in indices]

	def __call__(self, i, v, **kwargs):
		for index in self.__indices:
			res = index(i, v, **kwargs)
			if res is not None:
				return res

		msg = 'No indexing grammars match the given index structure {} and assignment {}'
		raise Exception(msg.format(SimpleParse(i), SimpleParse(v)))


class AbstractIndex(object):
	"""
	"""
	def __init__(self, **kwargs):
		self.define(**kwargs)
		self.__outputs = self.__parse(self.outputs)
		self.__inputs = self.__parse(self.inputs)
		self.__assign = self.__parse([g for g,_ in self.assign])
		self.__assignment_map = [fn for _,fn in self.assign]

	def define(self):
		raise NotImplementedError('Must override "define" method of AbstractIndex in a concrete subclass')

	def morph_index(self, i):
		return i # no default changes

	def __parse(self, grammar):
		def __parse_opt(o):
			if isinstance(o, (tuple, list)):
				opts = [__parse_opt(x) for x in o]
				if isinstance(o, tuple):
					opts = [x if isinstance(x,list) and len(x) > 1 else [x] for x in opts]
					return list(product(*opts))
				return opts
			elif is_subclass(o, AbstractIndex):
				return o().__parse(o().inputs)
			else:
				return o

		opts = [__parse_opt(o) for o in grammar]
		if len(opts) == 1 and (not isinstance(opts[0], list) or len(opts[0]) == 1):
			return opts[0]
		else:
			return list_flatten(opts)

	def __match(self, candidate, grammar):
		if isinstance(grammar, tuple):
			if isinstance(candidate, tuple) and len(grammar)==len(candidate):
				return all(self.__match(candidate[i],n) for i,n in enumerate(grammar))
			if len(grammar) == 1:
				return self.__match(candidate,grammar[0])
		elif isinstance(grammar, list):
			if isinstance(candidate, list) and candidate:
				return self.__match(candidate[0],grammar[0])
			elif isinstance(candidate, list) and not candidate:
				return True
		elif is_class(grammar):
			if is_class(candidate) and grammar == candidate == NullAssignment:
				return True
			else:
				return isinstance(candidate, grammar)
		elif isinstance(candidate, AbstractTensor):
			return self.__match(AbstractTensor,grammar)
		elif isinstance(candidate, types.FunctionType):
			return self.__match(candidate(),grammar)
		else:
			is_equal = candidate == grammar
			return type(is_equal) == bool and is_equal
		return False

	def __call__(self, i, v, **kwargs):
		for conf in self.__inputs:
			assign_matches = [self.__match(v, g) for g in self.__assign]
			#print(assign_matches, self.__assign, SimpleParse(v))
			if self.__match(i, conf) and sum(assign_matches):
				if sum(assign_matches) > 1:
					msg = 'Could not resolve assignment (ambiguous grammar) {}'
					raise Exception(msg.format(self.__assign))
				i = self.morph_index(i, **kwargs)
				args = (list(i) if isinstance(i, tuple) else [i]) + [v]
				func = self.__assignment_map[np.argmax(assign_matches)]
				resp = func(*args)
				if not any([self.__match(resp, g) for g in self.__outputs]):
					msg = 'Bad format {} for array index {} and assigned value {}'
					raise Exception(msg.format(resp, SimpleParse(i), SimpleParse(v)))
				return resp


class AbstractDatasetIndex(AbstractIndex):
	outputs = index([int, slice, AbstractTensor, np.ndarray],
					[[AbstractTensor], [UndefinedAttribute]],
					[[AbstractTensor], None, NullAssignment],
					is_compact=True)

	def identity(self, *x):
		return x

	def wrap_array(self, *x):
		return tuple([v for v in x[:-1]] + [[x[-1]]])

	def split_tensor(self, r, c, v):
		return (r, c, split_tensor(c, v))


class IndexRowSubset(AbstractDatasetIndex):
	def define(self, **kwargs):
		self.inputs = index(int, slice, AbstractTensor, np.ndarray)
		self.assign = [
			Map(index(AbstractTensor),   self.split_tensor),
			Map(index([AbstractTensor]), self.identity),
			Map(index(NullAssignment),   self.identity),
			Map(index(None),             self.identity),
		]

	def morph_index(self, i, **kwargs):
		return i, kwargs['all_columns']


class IndexColumnNew(AbstractDatasetIndex):
	def define(self, **kwargs):
		self.inputs = index([UndefinedAttribute],
							(all_slice, [UndefinedAttribute]))
		self.assign = [
			Map(index(AbstractTensor),   self.wrap_array),
			Map(index([AbstractTensor]), self.identity),
		]

	def morph_index(self, i, **kwargs):
		return i if isinstance(i, tuple) else (all_slice, i)


class IndexRowXORColumnSubset(AbstractDatasetIndex):
	def define(self, **kwargs):
		self.inputs = index([AbstractTensor],
							(all_slice, [AbstractTensor]),
							(IndexRowSubset, all_slice))
		self.assign = [
			Map(index(AbstractTensor),   self.split_tensor),
			Map(index([AbstractTensor]), self.identity),
			Map(index(NullAssignment),   self.identity),
			Map(index(None),             self.identity),
		]

	def morph_index(self, i, **kwargs):
		r, c = i if isinstance(i, tuple) else (kwargs['all_columns'], i)
		return r, (kwargs['all_columns'] if c == all_slice else c)


class IndexRowORColumnSubset(AbstractDatasetIndex):
	def define(self, **kwargs):
		self.inputs = index((IndexRowSubset, [AbstractTensor]))
		self.assign = [
			Map(index(AbstractTensor),   self.split_tensor),
			Map(index([AbstractTensor]), self.identity),
			Map(index(NullAssignment),   self.identity)
	  	]

	def morph_index(self, i, **kwargs):
		return i[0], (kwargs['all_columns'] if i[1] == all_slice else i[1])
