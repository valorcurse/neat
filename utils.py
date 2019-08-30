import pickle
import numpy as np

def find(f, seq):
    """Return first item in sequence where f(item) == True."""
    for item in seq:
        if f(item): 
            return item

def fastCopy(object):
    return pickle.loads(pickle.dumps(object, -1))

def cartesian(x, y, out=None):
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 4)