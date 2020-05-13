import os
import sys
import time
import functools

from .utils import exception_message, list_type, list_dims, rev_slice, slice_to_range

import numpy as np

set_type = set # overridden by backend set function

backends = {
	'NumPyBackend': ('numpy', 'np'),
	'PyTorchBackend': ('torch', 'torch'),
	#'TensorFlowBackend': ('tensorflow', 'tf')
}


def get():
	"""
	Get the backend, which must be first set using backend.set(name) with a valid
	backend name or by calling it without args after setting the 'TENSORPLANE_BACKEND'
	environment variable to a valid backend. An exception will be raised otherwise.
	"""
	if 'B' in globals():
		return globals()['B']
	else:
		raise Exception(exception_message("""No backend set. Please specify a
		backend tensor library by setting environment variable TENSORPLANE_BACKEND
		to one of {}""", list(backends.keys())))


def set(backend_name=''):
	"""
	Set the current backend. Do this before importing any other modules. If backend_name
	is not provided, the 'TENSORPLANE_BACKEND' environment variable will be used.
	"""
	_B = backend_name or os.getenv('TENSORPLANE_BACKEND')
	if 'B' not in globals():
		global B
	try:
		if _B in backends:
			# import necessary backend library dependency
			globals()[backends[_B][1]] = __import__(backends[_B][0])
			# set global backend
			B = globals()[_B]()
		elif _B is None:
			get() # raise 'no backend' exception
		else:
			raise Exception(exception_message("""Unsupported backend {}. Please select
			from the available backend tensor libraries: {}""", _B, list(backends.keys())))

		print(f'Backend successfully set to {type(B)}')

	except ModuleNotFoundError:
		raise ModuleNotFoundError(exception_message("""Backend DataFlow type {}
		could not be imported. Please install the necessary dependencies to
		use this backend, or try installing another backend package from the
		following options: {}""", _B, list(backends.keys())))

	return B


def tensor_property(func):
	"""
	Decorator for backend functions accessible as instance methods of AbstractTensor
	Determines if function is called with AbstractTensor instance. If the backend
	property/method is only available through the backend class use @backend_only.
	"""
	@functools.wraps(func)
	def property_wrap(*args, **kwargs):
		if len(args) > 1 and isinstance(args[1], AbstractTensor.Binding):
			args = args[:1] + (args[1].abstract_tensor,) + args[2:]
		return func(*args, **kwargs)
	return property_wrap


def backend_only(func):
	"""
	Decorator for backend functions that are not convertible to instance methods
	of AbstractTensor. Mutually exclusive usage with @tensor_property. This decorator
	MUST be used AFTER @unwrapped_tensor, otherwise an exception will be raised.
	"""
	@functools.wraps(func)
	def property_wrap(*args, **kwargs):
		if len(args) > 1 and isinstance(args[1], AbstractTensor.Binding):
			raise Exception(exception_message("""AbstractTensor does not have attibute
			{}. Use the AbstractBackend interface if you are trying to access non
			tensor library functions with an AbstractTensor.""", func.__name__))
		return func(*args, **kwargs)
	return property_wrap


def unwrapped_tensor(func):
	"""
	Decorator for unwrapping abstract tensors into their backing tensor library type
	"""
	@functools.wraps(func)
	def tensor_wrap(*args, **kwargs):
		kwargs = {k:v for k,v in B.unwrap(tuple(kwargs.items()))}
		args = B.unwrap(tuple(args))
		a = func(*args, **kwargs)
		return B.wrap(a)
	return tensor_wrap


class AbstractTensor(object):
	"""
	Main tensor class for wrapping backing tensor types
	"""
	class Binding(object):
		def __init__(self, t):
			self.abstract_tensor = t

	def __init__(self, v):
		self.__data = v

	@property
	def data(self):
		return self.__data

	def __is_method(self, attr):
		try:
			return attr(self) == self
		except TypeError:
			return False
		else:
			return False

	def __getattribute__(self, name):
		try:
			return object.__getattribute__(self, name)
		except AttributeError:
			pass
		try:
			backend_func = getattr(B, name)
			return functools.partial(backend_func, AbstractTensor.Binding(self))
		except AttributeError:
			pass
		raise AttributeError(exception_message('Backend has no attribute {}', name))

	def __str__(self):
		return f'Tensorplane.AbstractTensor: shape: {self.shape()}, data:\n{self.data}'

	def __repr__(self):
		return str(self)


class AbstractBackend(object):
	"""
	The abstract backend interface for using common scientific computation
	and machine learning library functions.
	"""
	def __init__(self, *opts):
		pass

	@unwrapped_tensor
	@backend_only
	def apply(self, d, method, *args, **kwargs):
		"""
		Apply an instance method on an initialized object of the abstract tensor's
		concrete type. Use this method with caution and ONLY on methods that
		are defined on all supported concrete data storage types with
		IDENTICAL parameters (including positional argument orderings) and
		IDENTICAL behaviors and return types. Although this is a useful shortcut for
		simple, uniformly defined methods (such as arithmetic operations)
		on many tensor/array-like types, it can break the layer of abstraction
		and information hiding provided by the AbstractBackend class. This
		may be replaced/refactored in future more stable versions of DataFlow.
		"""
		return getattr(self.unwrap(d), method)(*args, **kwargs)

	@tensor_property
	def wrap(self, stuff):
		"""
		Recursively wraps an AbstractTensor object. Descends into
		lists, tuples, sets, and dicts.
		"""
		if self.is_raw_tensor(stuff):
			return AbstractTensor(stuff)
		if isinstance(stuff, (tuple, set_type, list)):
			return type(stuff)([self.wrap(s) for s in stuff])
		if isinstance(stuff, dict):
			return {k:self.wrap(v) for k,v in stuff.items()}
		return stuff

	@tensor_property
	def unwrap(self, stuff):
		"""
		Recursively unwraps an AbstractTensor object. Descends into
		lists, tuples, sets, and dicts.
		"""
		if self.is_tensor(stuff):
			return stuff.data
		if self.is_raw_tensor(stuff):
			return stuff
		if isinstance(stuff, (tuple, set_type, list)):
			return type(stuff)([self.unwrap(s) for s in stuff])
		if isinstance(stuff, dict):
			return {k:self.unwrap(v) for k,v in stuff.items()}
		return stuff

	@tensor_property
	def is_tensor(self, x):
		"""
		Return if x is an AbstractTensor object
		"""
		return isinstance(x, AbstractTensor)

	@tensor_property
	def is_raw_tensor(self, x):
		"""
		Return if x is an instance of the concrete tensor/array-like object
		"""
		return self._is_raw_tensor(x)

	@property
	@backend_only
	def tensor(self):
		"""
		The underlying concrete tensor/array-like datatype for an abstract tensor
		"""
		return self._tensor()

	### Data properties

	@unwrapped_tensor
	@tensor_property
	def dtype(self, d):
		"""
		The datatype of the elements in an abstract tensor
		"""
		return self._dtype(d)

	@unwrapped_tensor
	@tensor_property
	def shape(self, d):
		"""
		The shape of the given abstract tensor, i.e. (dim_1, dim_2, ...)
		"""
		return self._shape(d)

	@unwrapped_tensor
	@tensor_property
	def size(self, d):
		"""
		The number of elements in the abstract tensor
		"""
		return self._size(d)

	@unwrapped_tensor
	@tensor_property
	def device(self, d):
		"""
		The current device of the tensor (one of backend.Device)
		"""
		return self._device(d)

	@unwrapped_tensor
	@tensor_property
	def address(self, d):
		"""
		Address in memory of abstract tensor's backing object
		"""
		return self._address(d)

	@unwrapped_tensor
	@tensor_property
	def type(self, d, dtype):
		"""
		Change datatypes for abstract tensor
		"""
		return self._type(d, dtype)

	@unwrapped_tensor
	@tensor_property
	def to_list(self, d):
		"""
		Convert an abstract tensor to a native Python list
		"""
		return self._to_list(d)

	@tensor_property
	def to_numpy(self, d):
		"""
		Convert an abstract tensor to a NumPy ndarray (not rewrapped!)
		"""
		return self._to_numpy(d.data)

	### Construction methods

	@unwrapped_tensor
	@backend_only
	def from_concrete(self, d):
		"""
		Convert a concrete tensor to an abstract tensor
		"""
		return d

	@unwrapped_tensor
	@backend_only
	def from_list(self, l):
		"""
		Convert a native python list to an abstract tensor
		"""
		return self._from_list(l)

	@unwrapped_tensor
	@backend_only
	def from_numpy(self, a):
		"""
		Convert a NumPy ndarray to an abstract tensor
		"""
		return self._from_numpy(a)

	@unwrapped_tensor
	@backend_only
	def from_range(self, min, max, step=1):
		"""
		Make an abstract tensor from the given range parameters
		"""
		return self._from_range(min, max, step)

	@unwrapped_tensor
	@backend_only
	def make_random_ints(self, min=0, max=1, shape=(1,1)):
		"""
		Make an abstract tensor with random integer values in the given range
		"""
		return self._make_random_ints(min, max, shape)

	@unwrapped_tensor
	@backend_only
	def make_random_floats(self, shape=(1,1)):
		"""
		Make an abstract tensor from a random uniform distribution in (0, 1)
		"""
		return self._make_random_floats(shape)

	@unwrapped_tensor
	@backend_only
	def make_ones(self, shape):
		"""
		Make an abstract tensor of ones from the given shape
		"""
		return self._make_ones(shape)

	@unwrapped_tensor
	@backend_only
	def make_zeros(self, shape):
		"""
		Make an abstract tensor of zeros from the given shape
		"""
		return self._make_zeros(shape)

	@unwrapped_tensor
	@backend_only
	def make_copy(self, d):
		"""
		Make and return a copy of the abstract tenosr in memory
		"""
		return self._make_copy(d)

	### Manipulation methods

	@unwrapped_tensor
	@tensor_property
	def reshape(self, d, *shape):
		"""
		The shape of the given abstract tensor, i.e. (dim_1, dim_2, ...)
		"""
		return self._reshape(d, *shape)

	@unwrapped_tensor
	@tensor_property
	def delete(self, d, idx, axis):
		"""
		Delete entries in abstract tensor along the given axis
		"""
		return self._delete(d, idx, axis)

	@unwrapped_tensor
	@tensor_property
	def index(self, d, i, v=None):
		"""
		Indexing interface for abstract tensors. Indexing behavior is defined
		by NumPy indexing syntax. See NumPy docs for details.
		"""
		return self._index(d, i, v=v)

	@unwrapped_tensor
	@backend_only
	def concat(self, ds, axis):
		"""
		Concatenate a tuple of abstract tensors along an existing axis
		"""
		return self._concat(ds, axis)

	### Access and comparison methods

	@unwrapped_tensor
	@tensor_property
	def argmax(self, d, axis=None):
		"""
		Get indices of maximum values in abstract tensor
		"""
		return self._argmax(d, axis=axis)

	@unwrapped_tensor
	@tensor_property
	def argsort(self, d, axis=-1):
		"""
		Get sorted indices along axis of abstract tensor
		"""
		return self._argsort(d, axis=axis)

	@unwrapped_tensor
	@backend_only
	def equal(self, d1, d2):
		"""
		Element-wise equality of abstract tensors, returning a Boolean value.
		"""
		return self._equal(d1, d2)


class NumPyBackend(AbstractBackend):

	def _tensor(self):
		return np.ndarray

	def _dtype(self, d):
		return type(d.flat[0].item()) if all(d.shape) else d.dtype

	def _shape(self, d):
		return d.shape

	def _size(self, d):
		return d.size

	def _device(self, d):
		return 'cpu'

	def _address(self, d):
		return d.__array_interface__['data'][0]

	def _is_raw_tensor(self, x):
		return isinstance(x, np.ndarray)

	def _type(self, d, dtype):
		return d.astype(dtype)

	def _reshape(self, d, *shape):
		return d.reshape(*shape)

	def _from_list(self, l):
		return np.array(l)

	def _from_numpy(self, a):
		return a

	def _from_range(self, min, max, step):
		return np.arange(min, max, step)

	def _make_random_ints(self, min, max, shape):
		return np.random.randint(min, max+1, size=shape)

	def _make_random_floats(self, shape):
		return np.random.uniform(size=shape)

	def _make_ones(self, shape):
		return np.ones(shape=shape)

	def _make_zeros(self, shape):
		return np.zeros(shape=shape)

	def _make_copy(self, d):
		return np.copy(d)

	def _to_list(self, d):
		return d.tolist()

	def _to_numpy(self, d):
		return np.array(d)

	def _concat(self, ds, axis):
		for i,_ in enumerate(ds):
			if ds[i].ndim == 1 and axis == 1:
				ds[i] = ds[i].reshape(-1, 1)
		return np.concatenate(ds, axis=axis)

	def _delete(self, d, idx, axis):
		return np.delete(d, idx, axis=axis)

	def _index(self, d, i, v):
		if v is None:
			return d[i]
		else:
			d[i] = v

	def _argmax(self, d, axis):
		return np.argmax(d, axis=axis)

	def _argsort(self, d, axis):
		return np.argsort(d, axis=axis)

	def _equal(self, d1, d2):
		raise np.array_equal(d1, d2)


class PyTorchBackend(AbstractBackend):

	def _type_map(self, t):
		return {
			float: torch.float64,
			int: torch.int64,
		}[t]

	def _tensor(self):
		return torch.tensor

	def _dtype(self, d):
		return type(d[(0,)*len(d.size())].item()) if all(d.size()) else d.numpy().dtype

	def _shape(self, d):
		return tuple(dim for dim in d.size())

	def _size(self, d):
		return d.numel()

	def _device(self, d):
		return d.get_device()

	def _address(self, d):
		return d.data_ptr()

	def _is_raw_tensor(self, x):
		return torch.is_tensor(x)

	def _type(self, d, dtype):
		return d.type(self._type_map(dtype))

	def _reshape(self, d, *shape):
		return d.view(*shape)

	def _from_list(self, l):
		if list_type(l) == str:
			print('WARNING: str tensors are not supported by PyTorch, defaulting to zeros')
			return torch.zeros(*list_dims(l))
		else:
			return torch.tensor(l)

	def _from_numpy(self, a):
		return torch.from_numpy(a)

	def _from_range(self, min, max, step):
		return torch.arange(min, max, step)

	def _make_random_ints(self, min, max, shape):
		return torch.randint(min, max, shape)

	def _make_random_floats(self, shape):
		return torch.rand(*shape)

	def _make_ones(self, shape):
		return torch.ones(*shape)

	def _make_zeros(self, shape):
		return torch.zeros(*shape)

	def _make_copy(self, d):
		return d.clone()

	def _to_list(self, d):
		return d.tolist()

	def _to_numpy(self, d):
		return d.cpu().numpy()

	def _concat(self, ds, axis):
		return torch.cat(ds, dim=axis)

	def _delete(self, d, idx, axis):
		if axis != 0:
			raise Exception('Non-zero axis deletion is not supported by PyTorch backend')
		if isinstance(idx,int):
			idx = slice(idx,idx+1,1)
		idx1, idx2 = slice_to_range(idx, d.size(0), fn=torch.arange, invert=True)
		idx = torch.cat((idx1,idx2))
		return  d[idx]

	def _index(self, d, i, v=None):
		# Handle reverse indexing quirks
		if isinstance(i,tuple) and rev_slice(i[0]):
			i_0 = slice_to_range(i[0], d.size(0), fn=torch.arange)
			i = (i_0,) + i[1:]
		if v is None:
			return d[i]
		else:
			d[i] = v

	def _argmax(self, d, axis):
		return torch.argmax(d, dim=axis)

	def _argsort(self, d, axis):
		return torch.argsort(d, dim=axis)

	def _equal(self, d1, d2):
		return torch.equal(d1, d2)


class TensorFlowBackend(AbstractBackend):
	def __init__(self, *opts):
		raise NotImplementedError('TensorFlow backend is not yet implemented.')
