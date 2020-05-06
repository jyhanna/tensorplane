import os
import numpy as np
from core import data

os.environ["TENSORPLANE_BACKEND"] = 'NumPyBackend'

def main():
    data.Dataset(x=np.arange(10))

if __name__ == '__main__':
    main()
