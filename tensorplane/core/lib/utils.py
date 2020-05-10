import os
import sys
import time


all_slice = slice(None, None, None)


def split_tensor(col_list, array):
	"""
	Split an AbstractTensor vertically w.r.t a list of tensors or np.ndarrays
	"""
	new_arr = [0]*len(col_list)
	prev_idx = 0
	for i,x in enumerate(col_list):
		x_dim1 = x.shape[-1] if isinstance(x, np.ndarray) else x.shape()[-1]
		new_arr[i] = array.index(I_[:,prev_idx:prev_idx+x_dim1])
		prev_idx += x_dim1
	return new_arr


def rev_slice(i):
	"""
	"""
	return isinstance(i, slice) and i.step is not None and i.step < 0


def slice_to_range(i, hi, fn=range, invert=False):
	"""
	"""
	strt, stop, step = i.start, i.stop, i.step
	step = 1 if step is None else step
	strt = hi+strt if strt is not None and strt < 0 else strt
	stop = hi+stop if stop is not None and stop < 0 else stop
	if rev_slice(i):
		mn, mx = hi-1, -1
		strt = mn if strt is None else strt
		stop = mx if stop is None else stop
	else:
		mn, mx = 0, hi
		strt = mn if strt is None else strt
		stop = mx if stop is None else stop
	if invert:
		return fn(mn,strt,step), fn(stop,mx,step)
	return fn(strt,stop,step)


def exception_message(x, *args):
	"""
	"""
	x = x.replace('\n', ' ').replace('\t', '')
	return x.format(*args) if args else x


def is_class(T):
	"""
	"""
	try:
		return issubclass(T, T)
	except TypeError:
		return False


def is_subclass(T1, T2):
	"""
	"""
	try:
		return issubclass(T1, T2)
	except TypeError:
		return False


def list_type(l):
	"""
	"""
	return list_type(l[0]) if l and isinstance(l, list) else type(l)


def list_dims(l):
	"""
	"""
	return [len(l)] + list_dims(l[0]) if l and isinstance(l, list) else []


def list_flatten(l):
	"""
	"""
	out = []
	for o in l:
		if isinstance(o, list):
			for x in o:
				out.append(x)
		else:
			out.append(o)
	return out


class _PrettyIndex(object):
	"""
	"""
	def __getitem__(self, i):
		return i

I_ = _PrettyIndex()
