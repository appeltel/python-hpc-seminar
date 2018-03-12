"""
Simple matrix inversion functions and tools for demonstrating pure-python
performance relative to better tools such as numpy.

The LUP decomposition and inversion functions are translated from C code
in the wikipedia article: https://en.wikipedia.org/wiki/LU_decomposition
The contributing author of this code appears to be anonymous.
"""


def lup_decompose(A, tol=0.000001):
    """
    Perform an in place LUP decomposition of square matrix A.

    The result is stored in A as a single matrix which contains both
    matrices L-E and U as A=(L-E)+U such that P*A=L*U.

    The permutation matrix is not stored as a matrix, but in an integer
    vector P of size N+1 containing column indexes where the
    permutation matrix has "1". This vector is returned as a list.

    :param list(list(float)) A: Invertible matrix
    :param float tol: small tolerance number to detect failure when the
        matrix is near degenerate

    :returns: The permutation matrix P
    :rtype: list(int)
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


def lup_invert(LU, P, IA=None):
    """
    Calculates the inverse of the LUP decomposed matrix LU, stores it in
    matrix IA and returns IA.

    :param list(list(float)) LU: LUP-decomposed invertible matrix, i.e. the
        result of the lup_decompose function
    :param list(int) P: The permutation matrix P as an integer vector
    :param list(list(float)) IA: Matrix to store the inverse. If None,
        a new matrix will be initialized.
    :returns: Inverted matrix
    :rtype: list(list(float))
    """
    n = len(LU)
    if IA is None:
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


if __name__ == '__main__':

    A = [
        [9., 2., 3.],
        [4., 5., 6.],
        [7., 8., 1.]
    ]
    print('Input test matrix:')
    for row in A:
        print(row)

    P = lup_decompose(A)
    IA = lup_invert(A, P)

    print('Output inverse matrix:')
    for row in IA:
        print(row)
