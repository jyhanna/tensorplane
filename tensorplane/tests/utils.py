import os
import sys
import random
from itertools import product
import numpy as np

from tensorplane.core.lib import backend as b

tensor = lambda ndarray: b.get().from_numpy(ndarray).data
pylist = lambda tensor: b.get().to_list(b.get().wrap(tensor))
array =  lambda tensor: b.get().to_numpy(b.get().wrap(tensor))
copy = lambda tensor: b.get().wrap(tensor).make_copy()
rand_str = lambda s: "".join(random.sample(s, random.randint(1,len(s))))
is_arr = lambda a: isinstance(a, np.ndarray)

lrange = lambda *args: list(range(*args))
lprod =  lambda *args: list(product(*args))


def np_convert(ts, fn=(lambda x: x), copy=False):
    return [fn(array(c).copy() if copy else array(c)) for c in ts]

def coalesced_type(d):
    

def assert_all_eq(iter, msg):
    failures = []
    iter = list(iter)
    for v1, v2 in iter:
        try:
            assert_eq(v1, v2, msg)
        except AssertionError as e:
            failures.append(f'\n{"-"*6}\n{e}')
    failure_msg = ''.join(failures)
    if failure_msg:
        raise AssertionError(f'{len(failures)}/{len(iter)} subtests failed: {msg}{failure_msg}')


def assert_eq(a, b, msg):
    out = (f"{msg}\nResult ({type(a)}{(a.shape,a.dtype) if is_arr(a) else ''}) = {a}" +
                f"\nTarget ({type(b)}{(b.shape,b.dtype) if is_arr(b) else ''}) = {b}")
    if is_arr(a):
        assert np.array_equal(a, b), out
    else:
        assert a == b, out


def assert_false(cond, msg):
    assert_eq(cond, False, msg)


def assert_true(cond, msg):
    assert_eq(cond, True, msg)


def assert_none(val, msg):
    assert val is None, f'{msg}\nValue={val}'
