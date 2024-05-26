import numpy as np

def factorized_square_Euclidean(xs, xt):
    square_norm_s = np.sum(xs**2, axis=1)
    square_norm_t = np.sum(xt**2, axis=1)
    A_1 = np.zeros((np.shape(xs)[0], 2 + np.shape(xs)[1]))
    A_1[:, 0] = square_norm_s
    A_1[:, 1] = np.ones(np.shape(xs)[0])
    A_1[:, 2:] = -2 * xs

    A_2 = np.zeros((2 + np.shape(xs)[1], np.shape(xt)[0]))
    A_2[0, :] = np.ones(np.shape(xt)[0])
    A_2[1, :] = square_norm_t
    A_2[2:, :] = xt.T

    return A_1, A_2
