<div align="center">
  <img src="/.github/logo.svg"><br>
</div>

-----------------

# Tensorplane: Unified data management in Python

Tensorplane is a Python library for high-level dataset management built upon powerful array and tensor frameworks like NumPy and PyTorch. While other dataset-oriented Python libraries exist, Tensorplane is modularly designed to interface with these advanced and well-supported backend frameworks to provide a single data access and manipulation interface for common data operations. Tensorplane is further designed for modern deep learning data processing pipelines by supporting a collection of multidimensional feature tensors encapsulated in a single object. Each tensor can also be a different type, allowing uniform manipulation of potentially heterogeneous datasets.

NOTE: Tensorplane if in early phases of pre-alpha development. While most basic features have been implemented, significant expansions in the codebase and modifications to the API are forthcoming. It is not recommended to use the current version or any previous versions or derivations of this software in a production environment until a future more stable distribution of Tensorplane is released.

## About

The main interface for working with data in Tensorplane is the `Dataset` class. A `Dataset` object supports expressive indexing which allows for reordering (sorting, shuffling), creating (vertical or horizontal adding or stacking), deleting, and performing various other operations on index-defined subsets of data. Other high-level dataset operations like splitting and batching are also supported through `Dataset` methods.

<div align="center">
  <img src="/.github/tp_info.svg"><br>
</div>


Notably, the Tensorplane API is almost entirely object-oriented. All data manipulation is performed on the `Dataset` instance. Beyond the `Dataset`, other abstract classes such as `AbstractIndex` and `AbstractBackend` can be subclassed to provide custom indexing grammars or concrete backend tensor library implementations, respectively. Ultimately, the key motivations behind Tensorplane's design are threefold: portability, expressivity, and extensibility.


## Installation

...

## Usage

...

## To-do

...
