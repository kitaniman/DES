import numpy as np
from numpy.typing import NDArray


pc_1, = np.array([
    57, 49, 41, 33, 25, 17, 9, 1,
    58, 50, 42, 34, 26, 18, 10, 2,
    59, 51, 43, 35, 27, 19, 11, 3,
    60, 52, 44, 36, 63, 55, 47, 39,
    31, 23, 15, 7, 62, 54, 46, 38,
    30, 22, 14, 6, 61, 53, 45, 37,
    29, 21, 13, 5, 28, 20, 12, 4
], np.uint8) - 1


pc_2, = np.array([
    14, 17, 11, 24, 1, 5, 3, 28,
    15, 6, 21, 10, 23, 19, 12, 4,
    26, 8, 16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55, 30, 40,
    51, 45, 33, 48, 44, 49, 39, 56,
    34, 53, 46, 42, 50, 36, 29, 32
], np.uint8) - 1


one_shift_rounds = {1, 2, 9, 16}


def key_schedule(k: NDArray):
    stripped_k = k[pc_1]
    c, d = np.split(stripped_k, 2)

    for i in range(1, 17):
        if i in one_shift_rounds:
            c, d = np.roll(c, -1), np.roll(d, -1)
        else:
            c, d = np.roll(c, -2), np.roll(d, -2)

        yield np.concatenate((c, d))[pc_2]
