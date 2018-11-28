"""
Simple matrix inversion functions and tools for demonstrating pure-python
performance relative to better tools such as numpy.

The LUP decomposition and inversion functions are translated from C code
in the wikipedia article: https://en.wikipedia.org/wiki/LU_decomposition
The contributing author of this code appears to be anonymous.
"""
import random
from copy import deepcopy


def invert_pure(A):
    """
    Return the inverse of matrix A using pure python lists and
    interpreter code.

    Matrix A is assumed to be a list of lists of 
    python floats (double precision)
    """
    LU = deepcopy(A)
    P = _lup_decompose_pure(LU)
    return _lup_invert_pure(LU, P)


def _lup_decompose_pure(A, tol=0.000001):
    """
    Perform an in place LUP decomposition of square matrix A.
    The result is stored in A as a single matrix which contains both
    matrices L-E and U as A=(L-E)+U such that P*A=L*U.

    The permutation matrix is not stored as a matrix, but in an integer
    vector P of size N+1 containing column indexes where the
    permutation matrix has "1". This vector is returned as a list.
    """
    n = len(A)

    # Unit permutation matrix, P[n] initialized with n
    P = list(range(n+1))

    for i in range(n):
        max_A = 0.0
        imax = i

        for k in range(i, n):
            if abs(A[k][i]) > max_A:
                max_A = abs(A[k][i])
                imax = k

        if max_A < tol:
            raise ValueError('Failure, input matrix is degenerate')

        if imax != i:
            # pivoting P
            P[i], P[imax] = P[imax], P[i]
            # pivoting rows of A
            A[i], A[imax] = A[imax], A[i]
            #counting pivots starting from N (for determinant)            
            P[n] += 1

        for j in range(i + 1, n):
            A[j][i] /= A[i][i]

            for k in range(i + 1, n):
                A[j][k] -= A[j][i] * A[i][k]

    return P


def _lup_invert_pure(LU, P):
    """
    Calculates the inverse of the LUP decomposed matrix LU, stores it in
    matrix IA and returns IA.
    """
    n = len(LU)
    IA = [[0.0] * n for _ in range(n)]
        
    for j in range(n):
        for i in range(n):
            if P[i] == j:
                IA[i][j] = 1.0
            else:
                IA[i][j] = 0.0

            for k in range(i):
                IA[i][j] -= LU[i][k] * IA[k][j]

        for i in range(n - 1, -1, -1):
            for k in range(i + 1, n):
                IA[i][j] -= LU[i][k] * IA[k][j]

            IA[i][j] = IA[i][j] / LU[i][i]

    return IA


#
# Inversion using numpy arrays and numba jit
#


import numpy as np
from numba import jit


def invert_numba(A):
    """
    Return the inverse of matrix A using numpy arrays and
    machine code.

    Matrix A is assumed to be a list of lists of 
    python floats (double precision)
    """
    LU = np.array(A)
    P = _lup_decompose_numba(LU)
    IA =  _lup_invert_numba(LU, P)
    return np.ndarray.tolist(IA)


@jit(nopython=True, nogil=True)
def _lup_decompose_numba(A, tol=0.000001):
    """
    Perform an in place LUP decomposition of square matrix A.
    The result is stored in A as a single matrix which contains both
    matrices L-E and U as A=(L-E)+U such that P*A=L*U.

    The permutation matrix is not stored as a matrix, but in an integer
    vector P of size N+1 containing column indexes where the
    permutation matrix has "1". This vector is returned as a list.
    """
    n = len(A)

    # Unit permutation matrix, P[n] initialized with n
    P = np.arange(n+1)

    for i in range(n):
        max_A = 0.0
        imax = i

        for k in range(i, n):
            if abs(A[k, i]) > max_A:
                max_A = abs(A[k, i])
                imax = k

        if max_A < tol:
            raise ValueError('Failure, input matrix is degenerate')

        if imax != i:
            # pivoting P
            P[i], P[imax] = P[imax], P[i]
            # pivoting rows of A
            for j in range(n):
                A[i, j], A[imax, j] = A[imax, j], A[i, j]
            #counting pivots starting from N (for determinant)            
            P[n] += 1

        for j in range(i + 1, n):
            A[j, i] /= A[i, i]

            for k in range(i + 1, n):
                A[j, k] -= A[j, i] * A[i, k]

    return P


@jit(nopython=True, nogil=True)
def _lup_invert_numba(LU, P):
    """
    Calculates the inverse of the LUP decomposed matrix LU, stores it in
    matrix IA and returns IA.
    """
    n = len(LU)
    IA = np.zeros((n, n))
        
    for j in range(n):
        for i in range(n):
            if P[i] == j:
                IA[i, j] = 1.0
            else:
                IA[i, j] = 0.0

            for k in range(i):
                IA[i, j] -= LU[i, k] * IA[k, j]

        for i in range(n - 1, -1, -1):
            for k in range(i + 1, n):
                IA[i, j] -= LU[i, k] * IA[k, j]

            IA[i, j] = IA[i, j] / LU[i, i]

    return IA



def invert_numpy_linalg(A):
    """
    Return the inverse of matrix A using numpy arrays and the
    compiled linalg routine in the numpy library

    Matrix A is assumed to be a list of lists of 
    python floats (double precision)
    """
    IA =  np.linalg.inv(np.array(A))
    return np.ndarray.tolist(IA)
