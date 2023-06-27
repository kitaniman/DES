import numpy as np
from numpy.typing import NDArray


from key_schedule import key_schedule
from permutations import ip, ip_inv
from f_function import f


def des(x: NDArray, k: NDArray):
    y = x[ip]

    l, r = np.split(y, 2)

    for ki in key_schedule(k):
        l, r = r, l ^ f(r, ki)

    y = np.concatenate((l, r))
    y = y[ip_inv]
    return y
