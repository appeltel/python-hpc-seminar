"""
Invert a set of 4 300x300 matrices using multiprocessing and
a queue.

This runs in parallel, avoids the GIL, and communicates output
via the queue.
"""
import datetime
import random
import multiprocessing

from matrix_invert import *


random.seed(42)

# target function to invert matrices
def invert(matrix, queue):
    P = lup_decompose(matrix)
    queue.put(lup_invert(matrix, P))

# Generate the list of matrices
input_matrices = [generate_matrix(300) for _ in range(4)]

# Timestamp when we begin
before = datetime.datetime.utcnow()

# run the inversions
result_q = multiprocessing.Queue()
tasks = []
for matrix in input_matrices:
    task = multiprocessing.Process(target=invert, args=(matrix, result_q))
    task.start()

inverses = [result_q.get() for _ in range(4)]

# Add the element (50,50) of all inverses, call that our result
result = sum(M[50][50] for M in inverses)

# Calculate elapsed time
after = datetime.datetime.utcnow()
elapsed = (after - before).total_seconds()

print('Inverted 4 300x300 matrices.')
print('Sum of M^-1[50][50] is {}'.format(result))
print('Time elapsed: {0:.3f}s'.format(elapsed))
