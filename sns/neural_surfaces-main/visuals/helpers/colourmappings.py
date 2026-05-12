import numpy as np

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def inv_sigmoid(x):
	return -np.log((1 - x) / x)


def mapping12(field, invert=False):
	if not invert:
		return sigmoid(0.1 * field)
	else:
		return 10 * inv_sigmoid(field)

def mapping13(field, invert=False):
	if not invert:
		return sigmoid(0.1 * np.sign(field) * (abs(field) ** 0.5))
	else:
		# inv_sigmoid gives 0.1 * sign(data) * sqrt(|data|)
		v = inv_sigmoid(field)
		return np.sign(v) * (10 * np.abs(v)) ** 2


def scaled_normals_cmap(x, invert=False):
	if not invert:
		return np.clip(0.02 * np.abs(x), 0.0, 1.0)
	else:
		return x / 0.02   # returns magnitude; sign is lost

def scaled_normals_cmap2(x, invert=False):
	if not invert:
		return np.clip(0.02 * x + 0.5, 0.0, 1.0)
	else:
		return (x - 0.5) / 0.02


def linear(x, invert=False):
	if not invert:
		return np.clip(x / 20, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) * 20

def linear2(x, invert=False):
	if not invert:
		return np.clip(10 * x, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) / 10

def linear3(x, invert=False):
	if not invert:
		return np.clip(0.005 * x, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) / 0.005


def linear5(x, invert=False):
	if not invert:
		return np.clip(x / 20, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) * 20
	

def linear10(x, invert=False):
	if not invert:
		return np.clip(x / 20, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) * 20
	
def linear100(x, invert=False):
	if not invert:
		return np.clip(x / 200, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) * 200

def linear500(x, invert=False):
	if not invert:
		return np.clip(x / 1000, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) * 1000

def linear1000(x, invert=False):
	if not invert:
		return np.clip(x / 2000, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) * 2000

def linear10000(x, invert=False):
	if not invert:
		return np.clip(x / 20000, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) * 20000

def linear6(x, invert=False):
	if not invert:
		return np.clip(0.0015 * x, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) / 0.0015

def linear7(x, invert=False):
	if not invert:
		return np.clip(0.0004 * x, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) / 0.0004

def linear8(x, invert=False):
	if not invert:
		return np.clip(x / 10, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) * 10

def linear9(x, invert=False):
	if not invert:
		return np.clip(x / 5, -0.5, 0.5) + 0.5
	else:
		return (x - 0.5) * 5


def positive_only_linear1(x, invert=False):
	if not invert:
		return np.clip(0.05 * x, 0, 1)
	else:
		return x / 0.05


def quadratic(x, invert=False):
	if not invert:
		return np.clip(np.sign(x) * np.sqrt(abs(x)) / 20, -0.5, 0.5) + 0.5
	else:
		v = x - 0.5
		return np.sign(v) * (20 * np.abs(v)) ** 2


def logmap(x, invert=False):
	if not invert:
		return linear(10 * np.log(x))
	else:
		# linear(10*log(x)) = clip(0.5*log(x), -0.5, 0.5) + 0.5
		# inverse: x = exp(2*(val - 0.5))
		return np.exp(2 * (x - 0.5))
