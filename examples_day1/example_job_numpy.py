"""
Invert a set of 4 300x300 matrices serially
"""
import datetime

import numpy as np


# target function to invert matrices
def invert(matrix):
    return np.linalg.inv(matrix)

# Generate the list of matrices
input_matrices = [
    np.random.uniform(low=-100., high=100., size=(300,300)) for _ in range(4)
]

# Timestamp when we begin
before = datetime.datetime.utcnow()

# run the inversions
inverses = []
for matrix in input_matrices:
    inverses.append(invert(matrix))

# Add the element (50,50) of all inverses, call that our result
result = sum(M[50][50] for M in inverses)

# Calculate elapsed time
after = datetime.datetime.utcnow()
elapsed = (after - before).total_seconds()

print('Inverted 4 300x300 matrices.')
print('Sum of M^-1[50][50] is {}'.format(result))
print('Time elapsed: {0:.3f}s'.format(elapsed))
