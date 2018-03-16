# python-hpc-seminar

Note: This version of the seminar is specifically tailored to the Vanderbilt
[Scientific Computing 3260](https://sc3260s18.github.io/) course and some
slides/materials may pertain specifically to this course or the
[ACCRE](https://www.vanderbilt.edu/accre/) facility.

This repository contains materials for a two-part seminar on introducing
python and its use in HPC. They are intended to be delivered in a seminar,
and so not all materials may be coherently readable on their own.

## Overview

These are materials for a two-part seminar introducing python to an
audience generally familiar with high-performance computing and
parallel programming, but with potentially little to no experience in
python.

The first part introduces some features of the python language and
multiprocessing tools available in the standard library of the
reference implementation. Issues of general performance and unsuitability
for scientific computing are discussed.

The second part introduces some commonly used scientific libraries starting
with numpy, and explores how these extensions provide a framework
for parallel, high-performance computation. Some examples of machine
learning libraries are also presented. The seminar concludes with an
introduction to Jupyter notebooks and some plotting libraries.

* [Slides PDF](/python_hpc_seminar.pdf)
* [Part One Examples](/examples_day1)
* [Part Two Examples](/examples_day2)

## Part One: The Python Language and Standard Library

**Outline:**

* Benefits of python / why use python?
* General features of the python language
* Inner workings and performance
* Multithreading and multiprocessing examples
* A few words on concurrency

**Resource Links:**

* [Official Python Tutorial](https://docs.python.org/3/tutorial/)
* [Python Standard Library Reference](https://docs.python.org/3/library/index.html) _keep this under your pillow_
* [Dive into Python](http://www.diveintopython3.net/)
* [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
* [Python Essential Reference](http://www.dabeaz.com/per.html) (book, not free)
* [Fluent Python](http://shop.oreilly.com/product/0636920032519.do) (book, not free, advanced)
* [PyNash Python User Group](http://pynash.org/connect/)
* [PyTN Conference](https://www.pytennessee.org/)
* [PyOhio Conference](https://www.pyohio.org/2018/)
* [Django Girls Tutorial (en)](https://tutorial.djangogirls.org/en/)

## Part Two: Scientific Libraries and Tools for Python

**Outline:**

* Packaging and distributing libraries, pip, PyPI, Anaconda
* Introduction to numpy
* Revisiting part one multiprocessing examples with numpy
* A quick tour of some scientific libraries
* Jupyter notebooks and literate programming
* Making pretty and interactive plots

**Resource Links:**

* [Python on ACCRE](https://www.vanderbilt.edu/accre/documentation/python/)
* [PyPI / Warehouse](https://pypi.org/) (the cheese shop)
* [Downloading conda](https://conda.io/docs/user-guide/install/download.html)
* [SciPy.org](https://www.scipy.org/) (numpy, scipy, pandas, matplotlib, ...)
* [Numpy Tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
* [Auto-Scaling Scikit-learn](https://databricks.com/blog/2016/02/08/auto-scaling-scikit-learn-with-apache-spark.html) (capstone idea)
* [Tensorflow Playground](http://playground.tensorflow.org/)
