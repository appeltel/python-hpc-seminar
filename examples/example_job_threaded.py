"""
Invert a set of 4 300x300 matrices serially
"""
import datetime
import random
import threading

from matrix_invert import *


random.seed(42)

# target function to invert matrices
def invert(matrix, output_list):
    P = lup_decompose(matrix)
    output_list.append(lup_invert(matrix, P))

# Generate the list of matrices
input_matrices = [generate_matrix(300) for _ in range(4)]

# Timestamp when we begin
before = datetime.datetime.utcnow()

# run the inversions
tasks = []
inverses = []
for matrix in input_matrices:
    task = threading.Thread(target=invert, args=(matrix, inverses))
    task.start()
    tasks.append(task)

for task in tasks:
    task.join()

# Add the element (50,50) of all inverses, call that our result
result = sum(M[50][50] for M in inverses)

# Calculate elapsed time
after = datetime.datetime.utcnow()
elapsed = (after - before).total_seconds()

print('Inverted 4 300x300 matrices.')
print('Sum of M^-1[50][50] is {}'.format(result))
print('Time elapsed: {0:.3f}s'.format(elapsed))
