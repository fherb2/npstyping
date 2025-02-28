# npstyping

NumPy s_hape / s_trict typing – Module for special Numpy use cases.

NumPy is intended for mathematical calculations. For this purpose, a special restriction of the shape of arrays does not appear to be necessary. In the context of application and module programming for special libraries or entire applications, it can be useful for quality assurance purposes to restrict NumPy arrays as far as possible in terms of shape and data typing.

There are two ways to implement typing in Python:

* static typing (works offline as a development tool (e.g. mypy and Pyright))
* asserts and other individial solutions for data content and type checks of data during the runtime

The module npstyping has two aims:

* Observe the shape of arrays during processing
* Extend dtype checks to ndarray-dtype 'object' cases (dtype = 'O') and unclear checks of the return value from numpy.datetime64 and numpy.timedelta64 arrays.

Numpy array data typing is very well implemented. Right now, the datetime64/timedelta64 data types are fuzzy when the values are passed to non-Numpy types. If the data in the array are not from a directly by NumPy supported format, so it is not possible to chech these data type by the dtype attribute. But also a really good shape restriction is not in sight. The nptyping module was a possible solution for NumPy 1.x. It is no longer maintained with 2.0. Perhaps a many-to-one solution is on the way, as the NumPy team and the creators of mypy and pyright are thinking about it. But both of these solutions are static type checkers. It's good to keep processing time for this outside of program run time. But this task is difficult because the static checkers must follow the mathematical path of the array processing to recognise the shape, starting with the shape at the beginning of the process.

This module is intended as an interim solution until a truly fully implemented mechanism for shape testing and tracking is implemented in common static type checkers.

What about the processing time for this work alongside the data processing itself?

To save processing time for type and form checks, the module implements the Python optimisation argument ‘-O’. This turns off the shape and type checks. However, all other functions remain active. This means that you can decide which checks should remain active in this case: those that are explicitly called in the code.

– Work in progress –
