import numpy as np
from metrics import *

def test_volume():
    for dim in range(0,7):
        print(dim,get_hypersphere_volume(dim,1))

if __name__ == "__main__":
    test_volume()
