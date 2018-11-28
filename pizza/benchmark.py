"""
This benchmark will print the time taken to invert several large
random matrices and print the sum of the diagonals of the inverses.
"""
import argparse
import datetime
import random
import sys


# Use a fixed seed for reproducibility
random.seed(42)


def main():
    """
    Run benchmark based on user specifications
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='serial', type=str)
    parser.add_argument('--size', default=300, type=int)
    parser.add_argument('--number', default=4, type=int)
    args = parser.parse_args()

    print(f'Inverting {args.number} {args.size}x{args.size} matrices.')
    print(f'Using method {args.method}')

    matrices = [generate_matrix(args.size) for _ in range(args.number)]

    before = datetime.datetime.utcnow()
    if args.method == 'serial':
        inverses = run_serial(matrices)
    elif args.method == 'threading':
        inverses = run_threaded(matrices)
    elif args.method == 'multiprocessing':
        inverses = run_multiprocess(matrices)
    elif args.method == 'serial_numba':
        inverses = run_serial_numba(matrices)
    elif args.method == 'threading_numba':
        inverses = run_threaded_numba(matrices)
    elif args.method == 'numpy_linalg':
        inverses = run_numpy_linalg(matrices)
    else:
        print(f'Method {args.method} is invalid')
        sys.exit(1)
    after = datetime.datetime.utcnow()
    elapsed = (after - before).total_seconds()

    result = sum(sum(M[i][i] for i in range(args.size)) for M in inverses)
    print(f'The sum of the diagonals of the inverses is {result}')
    print(f'Time elapsed: {elapsed:.3f}s')
    sys.exit(0)


def generate_matrix(n, minval=-100., maxval=100.):
    """
    Generate a random square matrix of dimension n x n with values
    taken from a uniform distribution
    """
    result = []
    for _ in range(n):
        result.append([random.uniform(minval, maxval) for _ in range(n)])
    return result


from invert import invert_pure


def run_serial(matrices):
    """
    Calculate the inverses of the matrices serially using pure python
    """
    results = []
    for M in matrices:
        results.append(invert_pure(M))
    return results


import threading
import queue


def run_threaded(matrices):
    """
    Calculate the inverses of the matrices using one thread per matrix
    in pure python
    """

    def target_func(matrix, queue):
        result = invert_pure(matrix)
        queue.put(result)

    results = queue.Queue()
    for M in matrices:
        task = threading.Thread(target=target_func, args=(M, results))
        task.start()

    return [results.get() for _ in range(len(matrices))]
    

import multiprocessing


def run_multiprocess(matrices):
    """
    Calculate the inverses of the matrices using one process per matrix
    in pure python
    """

    def target_func(matrix, queue):
        result = invert_pure(matrix)
        queue.put(result)

    results = multiprocessing.Queue()
    for M in matrices:
        task = multiprocessing.Process(target=target_func, args=(M, results))
        task.start()

    return [results.get() for _ in range(len(matrices))]


from invert import invert_numba


def run_serial_numba(matrices):
    """
    Calculate the inverses of the matrices serially using machine
    instructions on numpy arrays
    """
    results = []
    for M in matrices:
        results.append(invert_numba(M))
    return results


def run_threaded_numba(matrices):
    """
    Calculate the inverses of the matrices using one thread per matrix
    using machine instructions on numpy arrays
    """

    def target_func(matrix, queue):
        result = invert_numba(matrix)
        queue.put(result)

    results = queue.Queue()
    for M in matrices:
        task = threading.Thread(target=target_func, args=(M, results))
        task.start()

    return [results.get() for _ in range(len(matrices))]


from invert import invert_numpy_linalg


def run_numpy_linalg(matrices):
    """
    Calculate the inverses of the matrices using the numpy linalg library
    which lauches threads automatically
    """
    results = []
    for M in matrices:
        results.append(invert_numpy_linalg(M))
    return results

if __name__ == '__main__':
    main()
