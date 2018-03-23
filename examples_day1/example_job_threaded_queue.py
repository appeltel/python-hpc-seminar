"""
Invert a set of 4 300x300 matrices concurrently using threads,
and use a queue to communicate between tasks.

Note that this does not execute in parallel due to the CPython
Global Interpreter Lock (GIL)
"""
import datetime
from queue import Queue
import random
import threading

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
result_q = Queue()
for matrix in input_matrices:
    task = threading.Thread(target=invert, args=(matrix, result_q))
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
