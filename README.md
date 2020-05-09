<div align="center">
  <img src="/.github/logo.svg"><br>
</div>

-----------------

# Tensorplane: Unified data management in Python

Tensorplane is a Python library for high-level dataset management built upon powerful array and tensor frameworks like NumPy and PyTorch. While other dataset-oriented Python libraries exist, Tensorplane is modularly designed to leverage these advanced and well-supported backend frameworks to provide a single data access and manipulation interface for common data operations. Tensorplane is further designed for modern deep learning data processing pipelines by supporting a collection of multidimensional feature tensors encapsulated in a single object. Each tensor can also be a different type, allowing uniform manipulation of potentially heterogeneous datasets.

**NOTE: Tensorplane is in early phases of pre-alpha development. While most (but by no means all) basic features have been implemented, significant expansions in the codebase and modifications to the API are forthcoming. It is not recommended to use the current version or any previous versions or derivations of this software in a production environment until a future more stable distribution of Tensorplane is released.**

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
backend.set('NumPyBackend')

import numpy as np

# Load data from a file
ds = Dataset.load('YOUR-DATAFILE.json')

# Load data from arrays/tensors
ds = Dataset(x=np.randint(1, (100, 3)),
             y=np.randint(1, (100, 2)),
             z=np.randint(1, (100, 1)))

```

Now, you're ready to start manipulating data. Use `Dataset`s powerful indexing syntax to do stuff with, or get stuff from your data. Below are some examples of what you can do (using the NumPy backend, but just replace your inputs and indices with tensors instead if using PyTorch).

The basic syntax for indexing `Datasets` is:

- First Index (dimension 0): Rows/Instances
  - Backend tensor type (e.g. `np.ndarray` or `torch.tensor`) of boolean values or indices
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

# shuffle all instances
shuffle_idxs = np.arange(len(ds))
np.random.shuffle(shuffle_idxs)
ds = ds[shuffle_idxs,:]

# reverse all instances
ds = ds[::-1,:]

# reverse a subset of features
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

Full documentation coming soon, but if you have any questions, concerns, or suggestions, see the email listed below or start an issue.

## Contributing

Contributions are more than welcome, especially in these early phases of development. Feel free to email `jonathanyussefhanna(at: gmail).com` for more info.

## To-do

A lot
