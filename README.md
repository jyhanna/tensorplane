<div align="center">
  <img src="/.github/logo.svg"><br>
</div>

-----------------

# Tensorplane: Unified data management in Python

[![Build Status](https://travis-ci.org/jyhanna/tensorplane.svg?branch=master)](https://travis-ci.org/jyhanna/tensorplane)
[![codecov](https://codecov.io/gh/jyhanna/tensorplane/branch/master/graph/badge.svg)](https://codecov.io/gh/jyhanna/tensorplane)
[![PyPI version](https://badge.fury.io/py/tensorplane.svg)](https://badge.fury.io/py/tensorplane)

Tensorplane is a Python library for high-level dataset management built upon powerful array and tensor frameworks like NumPy and PyTorch. While other dataset-oriented Python libraries exist, Tensorplane is modularly designed to leverage these advanced and well-supported backend frameworks to provide a single data access and manipulation interface for common data operations. Tensorplane is further designed for modern deep learning data processing pipelines by supporting a collection of multidimensional feature tensors encapsulated in a single object. Each tensor can also be a different type, allowing uniform manipulation of potentially heterogeneous datasets.

**NOTE: Tensorplane is in early phases of pre-alpha development. While many basic features have been implemented, significant expansions in the codebase and modifications to the API are forthcoming. It is not recommended to use the current version or any previous versions or derivations of this software in a production environment until a future more stable distribution of Tensorplane is released.**

## About

The main interface for working with data in Tensorplane is the `Dataset` class. A `Dataset` object supports expressive indexing which allows for reordering (sorting, shuffling), creating (vertical or horizontal adding or stacking), deleting, and performing various other operations on index-defined subsets of data. Other high-level dataset operations like splitting and batching are also supported through `Dataset` methods.

<div align="center">
  <img src="/.github/tp_info.svg"><br>
</div>


Notably, the Tensorplane API is almost entirely object-oriented. All data manipulation is performed on the `Dataset` instance. Beyond the `Dataset`, other abstract classes such as `AbstractIndex` and `AbstractBackend` can be subclassed to provide custom indexing grammars or concrete backend tensor library implementations, respectively. Ultimately, the key motivations behind Tensorplane's design are threefold: portability, expressivity, and extensibility.


## Installation

The latest version of Tensorplane can be installed using `pip`:

`pip install tensorplane`

## Quickstart

### Configure your environment

First things first, choose you backend tensor framework. Begin by installing the necessary backend (e.g. `pip install torch`). Then, before initializing any classes, add the following code:

```python
from tensorplane.core.lib import backend
backend.set('YOUR-BACKEND')
```

Or just set the environment variable `TENSORPLANE_BACKEND`:

```python
os.environ['TENSORPLANE_BACKEND'] = 'YOUR-BACKEND'
```

Note that `'YOUR-BACKEND'` must be one of the keys in the dictionary of available backends `tensorplane.core.lib.backend.backends`. Backends can be dynamically set/switched at runtime using `backend.set`, but data from existing `Dataset`s will not be copied and must be re-initialized.

### Start doing stuff with data

To get started, load a dataset from a file using `Dataset.load` or initialize it using your tensor/array types of choice, with feature names as keyword arguments:

```python
from tensorplane.core.lib import backend
from tensorplane.core.data import Dataset
from tensorplane.core.wrap import NumPyWrap
backend.set('PyTorchBackend')

import torch
import numpy as np

# Ensure NumPy functions properly handle arbitrary backend tensor types
np = NumPyWrap(np)

# Load data from a file (eg. JSON)
ds = Dataset.load('YOUR-DATAFILE.json')

# Alternatively, load data from arrays/tensors
ds = Dataset(x=torch.randint(1, (100, 3)),
             y=torch.randint(1, (100, 2)),
             z=torch.randint(1, (100, 1)))

```

It is important to note here that once you've chosen your backend tensor library, **all attributes (feature tensors) of the Dataset will be instances of the library's tensor objects**. You can manipulate these tensors using the library's native tensor-consuming functions OR using `numpy` functions after wrapping the `numpy` module, in `NumPyWrap`, but note that the latter option will yield `np.ndarray`s, not necessarily the tensor objects native to the chosen backend.

Now, you're ready to start manipulating data. Use `Dataset`s powerful indexing syntax to do stuff with, or get stuff from your data. Below are some examples of what you can do with `Dataset` instances.

The basic syntax for indexing `Dataset`s is:

- First Index (dimension 0): Rows/Instances
  - Backend tensor type (e.g. `np.ndarray` or `torch.tensor`) or simply `np.ndarray` of boolean values or indices
  - An `int` for a specific row / instance location
  - A `slice` object, with the typical slice behavior
- Second Index (dimension 1): Columns/features
  - A `list` of backend tensor types, specifically the attributes of the dataset or a new attribute (e.g. `ds.labels`)
  - An empty slice `:` indicating all features in the dataset are to be accessed or manipulated
- Optional assignment values
  - A `list` of backend tensor types to assign
  - A backend tensor type (splittable into the indexed feature tensor dimensions)
  - `None` for deleting a subset of data (instances xor features)

```python
# sort all instances by a feature
ds = ds[np.argsort(ds.x),:]

# sort a subset of features by a feature
ds = ds[np.argsort(ds.x),[ds.x, ds.y]]

# shuffle instances for all features
shuffle_idxs = np.arange(len(ds))
np.random.shuffle(shuffle_idxs)
ds = ds[shuffle_idxs,:]

# reverse all instances for all features
ds = ds[::-1,:]

# reverse instances on a subset of features
ds = ds[::-1,[ds.x, ds.y]]

# delete some instances
ds[:20,:] = None

# delete some features
ds[:,[ds.y, ds.x]] = None

# create some features (~hstack)
ds[:,[ds.w, ds.v]] = [np.zeros((len(ds), 1)), np.zeros((len(ds), 1))]
ds[:,[ds.q]] = np.ones((len(ds), 2))
ds[:,[ds.r]] = np.ones((len(ds), 2))

# create some instances (~vstack)
ds[len(ds):,:] = np.ones((100, ds.shape[-1]))

# reassign some features
ds[:,[ds.w, ds.v]] = [np.ones((len(ds), 1)), np.zeros((len(ds), 1))]
ds[:,[ds.q, ds.r]] = np.random.randint(3, size=(len(ds), 4))

# do some scalar element-wise arithmetic
ds = ds[:,[ds.r, ds.v, ds.q]] + 1

# do some scalar arithmetic in-place
ds[:,[ds.r, ds.v, ds.q]] -= 1

# do some boolean indexing
ds = ds[ds.q[:,0]<2, [ds.q, ds.r]]

```

**Is this code really portable between tensor / machine learning libraries?**

**Yes**. Although each feature tensor of `ds` is a tensor object specific to the current backend (e.g. `torch.tensor` for the PyTorch backend), The `Dataset` class abstracts (or *wraps*) tensors of the current backend type into a common tensor interface when they are internally processed by an instance, and then de-abstracts (or *unwraps*) them when results of computations such as indexing, shuffling, or arithmetic operations are returned or assigned. Furthermore, the `NumPyWrap` class allows you to use `numpy` functions on tensor objects from any implemented backend tensor library, so you can use the common array manipulation functions you know and love in your data processing pipeline.

Full documentation coming soon, but if you have any questions, concerns, or suggestions, see the email listed below or start an issue.

## Contributing

Contributions are more than welcome, especially in these early phases of development. Feel free to email `jonathanyussefhanna(at: gmail).com` for more info.
