"""
Compare time to invert 300x300 matrices between a simple pure-python
LUP-decomposition algorithm and the numpy invert routine on a
numpy array.
"""
import textwrap
import timeit

import numpy as np

from matrix_invert import *


setup = textwrap.dedent("""\
    from __main__ import generate_matrix, lup_decompose, lup_invert
    M = generate_matrix(300)
    """)
statement = 'P = lup_decompose(M); MI = lup_invert(M, P);'
t = timeit.Timer(statement, setup=setup)
execution_time = t.timeit(number=1)
print(
    'Time to invert 300x300 matrix, LUP-pure-python: {0:.3f}s'
    .format(execution_time)
)

setup = textwrap.dedent("""\
    import numpy as np
    M = np.random.uniform(low=-100., high=100., size=(300,300))
    """)
statement = 'np.linalg.inv(M)'
t = timeit.Timer(statement, setup=setup)
execution_time = t.timeit(number=1)
print(
    'Time to invert 300x300 matrix, numpy-linalg: {0:.3f}s'
    .format(execution_time)
)

