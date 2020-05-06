import os
import numpy as np

os.environ["TENSORPLANE_BACKEND"] = 'NumPyBackend'

from core import data

def main():
    data.Dataset(x=np.arange(10))

if __name__ == '__main__':
    main()
