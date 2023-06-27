import numpy as np
from numpy.typing import NDArray


from key_schedule import key_schedule
from permutations import ip, ip_inv
from f_function import f


def des_encrypt(x: NDArray, k: NDArray):
    y = x[ip]

    l, r = np.split(y, 2)

    for ki in key_schedule(k):
        l, r = r, l ^ f(r, ki)

    y = np.concatenate((r, l))
    return y[ip_inv]


def des_decrypt(y: NDArray, k: NDArray):
    x = y[ip]

    l, r = np.split(x, 2)

    for ki in reversed(tuple(key_schedule(k))):
        l, r = r, l ^ f(r, ki)

    x = np.concatenate((r, l))
    return x[ip_inv]
