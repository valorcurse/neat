import pickle
import os
from inspect import currentframe, getframeinfo, stack

def find(f, seq):
    """Return first item in sequence where f(item) == True."""
    for item in seq:
        if f(item): 
            return item

def fastCopy(object):
    return pickle.loads(pickle.dumps(object, -1))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def debug_print(message):
    caller = getframeinfo(stack()[1][0])
    base = os.path.basename(caller.filename)
    print("{}:{} - {}".format(base, caller.lineno, message))