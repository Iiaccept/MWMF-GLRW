import numpy as np


def fNorm(A):
    """
    The normalization function of a similarity matrix

    Parameters:
    A (numpy.ndarray): Input similarity matrix

    Returns:
    B (numpy.ndarray): Normalized similarity matrix
    """
    num1, num2 = A.shape
    rnM = np.zeros(num1)
    cnM = np.zeros(num2)
    B = np.zeros((num1, num2))

    for ii in range(num1):
        rnM[ii] = np.sum(A[ii, :])

    for ij in range(num2):
        cnM[ij] = np.sum(A[:, ij])

    for i in range(num1):
        rsum = rnM[i]
        for j in range(num2):
            csum = cnM[j]
            if rsum == 0 or csum == 0:
                B[i, j] = 0
            else:
                B[i, j] = A[i, j] / np.sqrt(rsum * csum)

    return B

