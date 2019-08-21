def find(f, seq):
	"""Return first item in sequence where f(item) == True."""
	for item in seq:
		if f(item): 
			return item
			
def fastCopy(object):
	return pickle.loads(pickle.dumps(object, -1))