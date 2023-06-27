import numpy as np
from numpy.typing import NDArray


from s_boxes import s_boxes


expansion_permuataion = np.array([
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
], np.uint8) - 1


permutation_p = np.array([
    16, 7, 20, 21, 29, 12, 28, 17,
    1, 15, 23, 26, 5, 18, 31, 10,
    2, 8, 24, 14, 32, 27, 3, 9,
    19, 13, 30, 6, 22, 11, 4, 25
], np.uint8) - 1


powers_of_2 = np.array([8, 4, 2, 1], np.uint8)


def apply_s_box(b: NDArray, s_box: NDArray):
    row = b[0] << 1 | b[-1]
    col = np.dot(powers_of_2, b[1:-1])

    binary_string = np.binary_repr(s_box[row, col], 4)
    return np.array([int(bit) for bit in binary_string])


def f(ri: NDArray, ki: NDArray):
    expanded_ri = ri[expansion_permuataion]
    xored = expanded_ri ^ ki

    nonlinearized = np.concatenate([
        apply_s_box(b, s_box)
        for b, s_box in zip(np.split(xored, 8), s_boxes)
    ])

    permuted = nonlinearized[permutation_p]
    return permuted
